"""pyGIMLi travel-time loading, meshing, forward modeling, and residuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .data_io import load_numeric_table


def _require_pygimli() -> tuple[Any, Any]:
    import pygimli as pg
    from pygimli.physics import traveltime as tt

    return pg, tt


def _sensor_table(data: Any) -> pd.DataFrame:
    rows = []
    for i in range(data.sensorCount()):
        pos = data.sensorPosition(i)
        x = float(pos.x())
        y = float(pos.y())
        z = float(pos.z())
        elevation = z if abs(z) > 1e-12 else y
        rows.append({"sensor": i, "x": x, "y": y, "z": z, "elevation": elevation})
    return pd.DataFrame(rows)


def _data_array(data: Any, key: str) -> np.ndarray | None:
    try:
        return np.asarray(data[key], dtype=float)
    except Exception:
        return None


def _sensor_indices(
    source: np.ndarray,
    receiver: np.ndarray,
    n_sensors: int,
    *,
    sensor_index_base: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-based source and receiver indices with validation."""
    s = np.asarray(source, dtype=float)
    g = np.asarray(receiver, dtype=float)
    combined = np.r_[s, g]
    if not np.all(np.isfinite(combined)):
        raise ValueError("sensor indices must be finite.")
    rounded = np.rint(combined)
    if not np.allclose(combined, rounded, atol=1.0e-8, rtol=0.0):
        raise ValueError("sensor indices must be integer-valued.")
    if sensor_index_base is None:
        if np.nanmin(combined) == 0:
            sensor_index_base = 0
        elif np.nanmax(combined) == n_sensors:
            sensor_index_base = 1
        else:
            sensor_index_base = 0
    if sensor_index_base not in {0, 1}:
        raise ValueError("sensor_index_base must be 0, 1, or None.")
    indices = rounded.astype(int) - sensor_index_base
    if np.any(indices < 0) or np.any(indices >= n_sensors):
        raise ValueError("sensor index is outside the available sensor table.")
    n_source = len(s)
    return indices[:n_source], indices[n_source:]


def _observations(
    data: Any,
    sensors: pd.DataFrame,
    auto_scale: bool = True,
    sensor_index_base: int | None = None,
) -> pd.DataFrame:
    s = _data_array(data, "s")
    g = _data_array(data, "g")
    t = _data_array(data, "t")
    if s is None or g is None or t is None:
        return pd.DataFrame()
    t = t.astype(float)
    if auto_scale and np.nanmedian(t) > 1.0:
        warnings.warn("Observed travel times look like milliseconds; converting to seconds.", RuntimeWarning)
        t = t * 1e-3
    err = _data_array(data, "err")
    if err is None:
        err = _data_array(data, "error")

    si, gi = _sensor_indices(
        s,
        g,
        len(sensors),
        sensor_index_base=sensor_index_base,
    )
    sx = sensors.loc[si, "x"].to_numpy()
    gx = sensors.loc[gi, "x"].to_numpy()
    offset = np.abs(gx - sx)
    out = pd.DataFrame({"s": s, "g": g, "t": t, "offset": offset})
    if err is not None:
        out["err"] = np.asarray(err, dtype=float)
    return out


def _manual_pygimli_parse(path: Path) -> tuple[Any, int]:
    pg, _ = _require_pygimli()
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    n_sensors = int(lines[0].split()[0])
    idx = 1
    sensor_header = []
    if lines[idx].startswith("#"):
        sensor_header = lines[idx].lstrip("#").split()
        idx += 1
    sensors = []
    for _ in range(n_sensors):
        sensors.append([float(v) for v in lines[idx].split()])
        idx += 1
    n_data = int(lines[idx].split()[0])
    idx += 1
    data_header = []
    if lines[idx].startswith("#"):
        data_header = lines[idx].lstrip("#").split()
        idx += 1
    if not data_header:
        data_header = ["s", "g", "t"]
    rows = [[float(v) for v in lines[idx + i].split()] for i in range(n_data)]
    arr = np.asarray(rows, dtype=float)
    try:
        source_column = data_header.index("s")
        receiver_column = data_header.index("g")
        raw_indices = np.r_[arr[:, source_column], arr[:, receiver_column]]
        sensor_index_base = 0 if np.nanmin(raw_indices) == 0 else 1
    except ValueError:
        sensor_index_base = 1

    dc = pg.DataContainer()
    dc.registerSensorIndex("s")
    dc.registerSensorIndex("g")
    dc.setSensorIndexOnFileFromOne(True)
    for row in sensors:
        if len(row) >= 3:
            dc.createSensor(row[:3])
        elif "z" in sensor_header:
            dc.createSensor([row[0], row[1]])
        else:
            dc.createSensor(row[:2])
    dc.resize(n_data)
    for i, key in enumerate(data_header[: arr.shape[1]]):
        dc.set(key, arr[:, i])
    if "valid" not in dc.dataMap():
        dc.set("valid", np.ones(n_data))
    return dc, sensor_index_base


def load_pygimli_traveltime_data(path: str | Path, auto_scale: bool = True) -> dict[str, Any]:
    """Load a pyGIMLi travel-time file, including near-compatible text formats."""
    path = Path(path)
    pg, tt = _require_pygimli()
    data = None
    manager = None
    load_method = "manual"
    sensor_index_base = 0
    try:
        manager = tt.TravelTimeManager(str(path))
        data = getattr(manager, "data", None) or getattr(manager, "dataContainer", None)
        if data is not None and data.size() > 0:
            load_method = "TravelTimeManager"
    except Exception:
        data = None
    if data is None:
        try:
            data = pg.DataContainer(str(path))
            load_method = "DataContainer"
        except Exception:
            data, sensor_index_base = _manual_pygimli_parse(path)
            load_method = "manual"
    sensors = _sensor_table(data)
    observations = _observations(
        data,
        sensors,
        auto_scale=auto_scale,
        sensor_index_base=sensor_index_base,
    )
    return {
        "path": path,
        "name": path.stem,
        "data": data,
        "manager": manager,
        "sensors": sensors,
        "observations": observations,
        "load_method": load_method,
        "keys": list(data.dataMap().keys()),
        "n_sensors": int(data.sensorCount()),
        "n_picks": int(data.size()),
    }


def load_line_topography(path: str | Path) -> pd.DataFrame:
    """Load a line topography or map-view trace file."""
    df = load_numeric_table(path)
    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ["x", "y", "z", "distance"]
    elif df.shape[1] == 3:
        vals = df.iloc[:, :3].to_numpy(dtype=float)
        if np.nanmedian(np.abs(vals[:, :2])) > 1.0e4:
            df.columns = ["x", "y", "distance"]
        else:
            df.columns = ["distance", "z", "extra"]
    elif df.shape[1] == 2:
        df.columns = ["distance", "z"]
        df["x"] = df["distance"]
        df["y"] = 0.0
    else:
        raise ValueError(f"Cannot parse line topography columns in {path}")
    if "distance" not in df:
        dx = np.diff(df["x"], prepend=df["x"].iloc[0])
        dy = np.diff(df["y"], prepend=df["y"].iloc[0])
        df["distance"] = np.cumsum(np.hypot(dx, dy))
    return df


def straighten_line_topography(
    topography: pd.DataFrame,
    distances: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Represent a map-view seismic trace by its first-to-last straight line.

    The returned third column is distance along line. When sensor distances are
    supplied, the straight trace is sampled at those positions and is linearly
    extended beyond the topography endpoints when needed.
    """
    if not {"x", "y"}.issubset(topography.columns):
        raise ValueError("straight-line geometry requires x and y columns.")
    xy = topography[["x", "y"]].to_numpy(dtype=float)
    if len(xy) < 2 or not np.all(np.isfinite(xy)):
        raise ValueError("straight-line geometry requires two finite endpoints.")
    if "distance" in topography:
        original_distance = topography["distance"].to_numpy(dtype=float)
        original_distance = original_distance - original_distance[0]
    else:
        step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        original_distance = np.r_[0.0, np.cumsum(step)]
    length_coordinate = float(original_distance[-1])
    endpoint_vector = xy[-1] - xy[0]
    endpoint_length = float(np.linalg.norm(endpoint_vector))
    if length_coordinate <= 0 or endpoint_length <= 0:
        raise ValueError("straight-line geometry has coincident endpoints.")

    requested_distance = (
        original_distance
        if distances is None
        else np.asarray(distances, dtype=float)
    )
    fraction = requested_distance / length_coordinate
    straight_xy = xy[0] + fraction[:, None] * endpoint_vector
    relative = xy - xy[0]
    deviation = np.abs(
        endpoint_vector[0] * relative[:, 1]
        - endpoint_vector[1] * relative[:, 0]
    ) / endpoint_length
    diagnostics = {
        "endpoint_length_m": endpoint_length,
        "topography_distance_m": length_coordinate,
        "maximum_plan_deviation_m": float(np.max(deviation)),
        "rms_plan_deviation_m": float(np.sqrt(np.mean(deviation**2))),
    }
    return (
        np.column_stack((straight_xy, requested_distance)),
        diagnostics,
    )


def create_mesh_for_data(data_scheme: Any, bottom_depth: float, cell_size: float) -> Any:
    """Create a pyGIMLi travel-time mesh directly from a data scheme."""
    _, tt = _require_pygimli()
    manager = tt.TravelTimeManager()
    return manager.createMesh(data=data_scheme, paraDepth=float(bottom_depth), paraMaxCellSize=float(cell_size))


def map_velocity_section_to_mesh(mesh: Any, velocity_section: dict[str, np.ndarray]) -> np.ndarray:
    """Map a distance-depth velocity section onto pyGIMLi mesh cells."""
    distance = np.asarray(velocity_section["distance"], dtype=float)
    depth = np.asarray(velocity_section.get("depth", velocity_section.get("z")), dtype=float)
    vp = np.asarray(velocity_section.get("vp", velocity_section.get("values")), dtype=float)
    surface = velocity_section.get("surface_elevation")
    if surface is None:
        surface_values = np.zeros_like(distance)
    else:
        surface_values = np.asarray(surface, dtype=float)
    if vp.shape != (len(depth), len(distance)):
        raise ValueError("velocity_section vp must have shape (n_depth, n_distance)")

    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((depth, distance), vp, bounds_error=False, fill_value=None)
    cell_vel = np.empty(mesh.cellCount(), dtype=float)
    for i, cell in enumerate(mesh.cells()):
        center = cell.center()
        x = float(center.x())
        y = float(center.y())
        surf = np.interp(x, distance, surface_values)
        d = max(surf - y, 0.0)
        cell_vel[i] = float(interp([[d, x]])[0])
    bad = ~np.isfinite(cell_vel)
    if np.any(bad):
        cell_vel[bad] = float(np.nanmedian(vp))
    return np.clip(cell_vel, 150.0, 8000.0)


def simulate_travel_times(mesh: Any, data_scheme: Any, cell_velocities: np.ndarray, sec_nodes: int = 3) -> dict[str, Any]:
    """Run pyGIMLi first-arrival forward modeling."""
    _, tt = _require_pygimli()
    manager = tt.TravelTimeManager()
    simulated = manager.simulate(
        mesh=mesh,
        scheme=data_scheme,
        vel=np.asarray(cell_velocities, dtype=float),
        secNodes=int(sec_nodes),
        noiseLevel=0.0,
        noiseAbs=0.0,
    )
    return {"predicted": np.asarray(simulated["t"], dtype=float), "data": simulated, "manager": manager}


def compute_travel_time_residuals(t_obs: np.ndarray, t_pred: np.ndarray, err: np.ndarray | None = None) -> dict[str, Any]:
    """Compute residual statistics for observed and predicted travel times."""
    obs = np.asarray(t_obs, dtype=float)
    pred = np.asarray(t_pred, dtype=float)
    n = min(len(obs), len(pred))
    residual = pred[:n] - obs[:n]
    rmse = float(np.sqrt(np.nanmean(residual**2)))
    out: dict[str, Any] = {"residual": residual, "rmse": rmse, "bias": float(np.nanmean(residual)), "n": int(n)}
    if err is not None:
        err_arr = np.asarray(err, dtype=float)[:n]
        ok = np.isfinite(err_arr) & (err_arr > 0)
        if np.any(ok):
            out["chi_square"] = float(np.nanmean((residual[ok] / err_arr[ok]) ** 2))
    return out


