"""DEM loading, terrain metrics, and line sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DemGrid:
    """DEM array plus grid coordinates."""

    data: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    crs: Any | None = None
    path: Path | None = None
    nodata: float | None = None


def load_dem(path: str | Path) -> DemGrid:
    """Load a georeferenced DEM with rasterio."""
    path = Path(path)
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        rows, cols = data.shape
        xs = np.arange(cols) + 0.5
        ys = np.arange(rows) + 0.5
        x = np.array([src.transform * (col, 0.5) for col in xs], dtype=float)[:, 0]
        y = np.array([src.transform * (0.5, row) for row in ys], dtype=float)[:, 1]
        dx = abs(float(src.transform.a))
        dy = abs(float(src.transform.e))
        dem = DemGrid(data=data, x=x, y=y, dx=dx, dy=dy, crs=src.crs, path=path, nodata=nodata)

    if dem.nodata is not None:
        dem.data = np.where(np.isclose(dem.data, dem.nodata), np.nan, dem.data)
    return dem


def fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    """Fill NaN cells using nearest valid neighbors."""
    arr = np.asarray(a, dtype=float)
    if not np.isnan(arr).any():
        return arr
    from scipy import ndimage

    mask = np.isnan(arr)
    indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(indices)]


def crop_or_resample_dem(dem: DemGrid, target_resolution: float | None = None) -> DemGrid:
    """Optionally resample a DEM to a target resolution."""
    if target_resolution is None or np.isclose(target_resolution, dem.dx):
        return dem
    from scipy.ndimage import zoom

    scale_x = dem.dx / float(target_resolution)
    scale_y = dem.dy / float(target_resolution)
    data = zoom(fill_nan_nearest(dem.data), (scale_y, scale_x), order=1)
    rows, cols = data.shape
    x = dem.x[0] + np.arange(cols) * float(target_resolution)
    y_step = -float(target_resolution) if dem.y[0] > dem.y[-1] else float(target_resolution)
    y = dem.y[0] + np.arange(rows) * y_step
    return DemGrid(data=data, x=x, y=y, dx=float(target_resolution), dy=float(target_resolution), crs=dem.crs, path=dem.path)


def compute_hand(
    dem: DemGrid | np.ndarray,
    drainage_area: np.ndarray,
    threshold: float,
    flow_receivers: np.ndarray,
    *,
    channel_mask: np.ndarray | None = None,
    include_terminal_nodes: bool = True,
    normalized: bool = True,
) -> np.ndarray:
    """Compute height above nearest downstream drainage."""
    z = fill_nan_nearest(dem.data if isinstance(dem, DemGrid) else np.asarray(dem, dtype=float))
    drainage = np.asarray(drainage_area, dtype=float)
    if drainage.shape != z.shape:
        raise ValueError(f"drainage_area shape {drainage.shape} does not match DEM shape {z.shape}.")

    receivers = np.asarray(flow_receivers).reshape(-1).astype(int)
    z_flat = z.reshape(-1)
    if receivers.size != z_flat.size:
        raise ValueError(f"flow_receivers length {receivers.size} does not match DEM node count {z_flat.size}.")

    terminal_nodes = receivers == np.arange(receivers.size)
    if channel_mask is None:
        channels = drainage.reshape(-1) >= float(threshold)
    else:
        channels = np.asarray(channel_mask, dtype=bool).reshape(-1)
        if channels.size != z_flat.size:
            raise ValueError(
                f"channel_mask length {channels.size} does not match DEM node count {z_flat.size}."
            )
    if include_terminal_nodes:
        channels = channels | terminal_nodes
    channels = channels.reshape(z.shape)
    if not np.any(channels):
        raise ValueError("HAND requires at least one channel node; lower the drainage-area threshold.")

    channel_flat = channels.reshape(-1)
    downstream_elev = np.full(z_flat.shape, np.nan, dtype=float)
    for i in range(z_flat.size):
        node = i
        seen: set[int] = set()
        while 0 <= node < z_flat.size and node not in seen:
            seen.add(node)
            if channel_flat[node]:
                downstream_elev[i] = float(z_flat[node])
                break
            next_node = int(receivers[node])
            if next_node == node:
                break
            node = next_node

    unresolved = np.isnan(downstream_elev)
    if np.any(unresolved):
        raise ValueError(
            f"HAND could not trace {int(np.sum(unresolved))} nodes to a downstream channel; "
            "lower channel_drainage_area_threshold or check flow routing."
        )
    channel_elev = downstream_elev.reshape(z.shape)
    hand = np.maximum(z - channel_elev, 0.0)
    hand[channels] = 0.0
    return normalize_array(hand) if normalized else hand


def normalize_array(a: np.ndarray) -> np.ndarray:
    """Scale an array to 0-1 while ignoring NaNs."""
    arr = np.asarray(a, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if np.isclose(amax, amin):
        return np.zeros_like(arr, dtype=float)
    return (arr - amin) / (amax - amin)


def sample_dem_along_line(x: np.ndarray, y: np.ndarray, dem_grid: DemGrid) -> np.ndarray:
    """Sample DEM values at line coordinates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    data = fill_nan_nearest(dem_grid.data)
    grid_y = dem_grid.y
    grid_x = dem_grid.x
    if grid_y[0] > grid_y[-1]:
        grid_y = grid_y[::-1]
        data = data[::-1, :]
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((grid_y, grid_x), data, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([y, x]))


def line_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute cumulative distance along a polyline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return np.array([], dtype=float)
    step = np.hypot(np.diff(x), np.diff(y))
    return np.r_[0.0, np.cumsum(step)]


