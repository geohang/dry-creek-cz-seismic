"""Drainage-controlled critical-zone geometry and rock-physics velocity models.

``compute_rd_geometry_from_zs`` applies the relief-ratio model, in which fresh
bedrock sits at relief ``Zb = r Zs`` above the hydrologically connected channel
datum, so depth to fresh bedrock is ``Df = (1 - r) Zs``.
``build_cz_unit_grid`` assigns each cell to mobile regolith, weathered/fractured
bedrock, or fresh bedrock. Velocity comes from a Hertz-Mindlin lower bound in the
mobile regolith and a differential effective medium in granite below it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dem_tools import DemGrid, crop_or_resample_dem, load_dem
from .velocity_model import _patch_pyhydrogeophysx_root


DEFAULT_RPV_PARAMS: dict[str, Any] = {
    "soil": {
        "phi": 0.42,
        "Sw": 0.45,
        "Km": 15.0,
        "Gm": 10.0,
        "rho": 2300.0,
        "critical_porosity": 0.4,
        "hertz_mindlin_bound": "mean",
        "hertz_mindlin_depth_m": 1.0,
    },
    "weathered_bedrock": {
        "phi_top": 0.22,
        "phi_bottom": 0.05,
        "Sw_top": 0.65,
        "Sw_bottom": 0.90,
        "alpha_top": 0.03,
        "alpha_bottom": 0.12,
        "Km": 55.0,
        "Gm": 33.0,
        "rho": 2650.0,
    },
    "fresh_bedrock": {
        "phi": 0.015,
        "Sw": 0.95,
        "alpha": 0.20,
        "Km": 62.0,
        "Gm": 38.0,
        "rho": 2700.0,
    },
    "vp_min": 300.0,
    "vp_max": 6000.0,
    "weathered_lookup_samples": 512,
}


UNIT_LABELS = {
    0: "soil",
    1: "weathered_bedrock",
    2: "fresh_bedrock",
}


def load_dry_creek_dem(
    config: dict[str, Any],
    root: str | Path,
    *,
    target_resolution_m: float | None = None,
) -> DemGrid:
    """Load the configured Dry Creek DEM and optionally resample it."""
    root = Path(root)
    dem_path = root / str(config.get("dem_file", ""))
    if not dem_path.exists():
        raise FileNotFoundError(
            f"Dry Creek DEM was not found at {dem_path}. "
            "Update config.yaml dem_file or place the DEM at that path."
        )
    dem = load_dem(dem_path)
    if target_resolution_m is None:
        target_resolution_m = config.get("target_dem_resolution")
    return crop_or_resample_dem(dem, target_resolution_m)


def compute_rd_geometry_from_zs(
    zs: np.ndarray,
    h_soil: np.ndarray,
    r: float,
    *,
    smooth_sigma: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Build strict RD relief-ratio geometry from relief to channel and soil depth.

    The only depth adjustment retained here is the physical layer-ordering
    constraint ``D_fresh >= H_soil``.
    """
    if not 0.0 <= float(r) <= 1.0:
        raise ValueError("r must lie in [0, 1].")
    relief = np.maximum(np.asarray(zs, dtype=float), 0.0)
    soil = np.asarray(h_soil, dtype=float)
    if relief.shape != soil.shape:
        raise ValueError("zs and h_soil must have the same shape.")

    zb = float(r) * relief
    d_fresh_raw = np.maximum(relief - zb, 0.0)
    d_fresh_geometric = d_fresh_raw
    if smooth_sigma is not None and float(smooth_sigma) > 0.0:
        from scipy.ndimage import gaussian_filter

        d_fresh_geometric = gaussian_filter(
            d_fresh_geometric,
            sigma=float(smooth_sigma),
            mode="nearest",
        )
        d_fresh_geometric = np.maximum(d_fresh_geometric, 0.0)
    soil_exceeds = soil > d_fresh_geometric
    d_fresh = np.maximum(d_fresh_geometric, soil)
    h_weathered_raw = d_fresh_geometric - soil
    h_weathered = np.maximum(d_fresh - soil, 0.0)
    return {
        "Zs": relief,
        "Zb": zb,
        "Zb_relief": zb,
        "H_soil": soil,
        "H_weathered": h_weathered,
        "H_weathered_raw": h_weathered_raw,
        "D_fresh": d_fresh,
        "D_fresh_raw": d_fresh_raw,
        "D_fresh_geometric": d_fresh_geometric,
        "soil_exceeds_rd_depth_fraction": float(np.mean(soil_exceeds)),
        "r": float(r),
        "smooth_sigma": None if smooth_sigma is None else float(smooth_sigma),
    }


compute_rd_geometry_from_Zs = compute_rd_geometry_from_zs


def build_cz_unit_grid(
    h_soil: np.ndarray,
    d_fresh: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Build a 3D unit grid: 0 soil, 1 weathered bedrock, 2 fresh bedrock."""
    soil = np.maximum(np.asarray(h_soil, dtype=float), 0.0)
    fresh = np.maximum(np.asarray(d_fresh, dtype=float), soil)
    if soil.shape != fresh.shape:
        raise ValueError("h_soil and d_fresh must have the same shape.")
    z = np.asarray(depth, dtype=float)[:, None, None]
    return np.where(z <= soil, 0, np.where(z <= fresh, 1, 2)).astype(np.int16)


def _dem_velocity(phi: np.ndarray, sw: np.ndarray, alpha: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from PyHydroGeophysX.petrophysics import velocity_models as vm

    _patch_pyhydrogeophysx_root(vm)
    _, _, vp = vm.velDEM(
        np.asarray(phi, dtype=float),
        float(params["Km"]),
        float(params["Gm"]),
        float(params["rho"]),
        np.asarray(sw, dtype=float),
        np.asarray(alpha, dtype=float),
    )
    return np.asarray(vp, dtype=float)


def _vel_porous_with_critical_porosity(
    phi: np.ndarray,
    km: float,
    gm: float,
    rho_b: float,
    saturation: np.ndarray,
    *,
    depth: float,
    critical_porosity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hertz-Mindlin high/low Vp bounds with configurable critical porosity."""
    from PyHydroGeophysX.petrophysics import velocity_models as vm

    phi_c = float(critical_porosity)
    if not 0.0 < phi_c < 1.0:
        raise ValueError("critical_porosity must lie between 0 and 1.")

    phi_arr, sat_arr = np.broadcast_arrays(
        np.asarray(phi, dtype=float),
        np.asarray(saturation, dtype=float),
    )
    out_shape = phi_arr.shape
    phi_flat = phi_arr.ravel()
    sat_flat = sat_arr.ravel()

    v = (3.0 * km - 2.0 * gm) / (2.0 * (3.0 * km + gm))
    pressure_gpa = (rho_b - 1000.0) * 9.8 * float(depth) / 1.0e9
    k_hm = (
        4.0**2
        * (1.0 - phi_c) ** 2
        * gm**2
        / (18.0 * np.pi**2 * (1.0 - v) ** 2)
        * pressure_gpa
    ) ** (1.0 / 3.0)
    g_hm = (5.0 - 4.0 * v) / (10.0 - 2.0 * v) * (
        (3.0 * 4.0**2 * (1.0 - phi_c) ** 2 * gm**2)
        * pressure_gpa
        / (2.0 * np.pi**2 * (1.0 - v) ** 2)
    ) ** (1.0 / 3.0)

    vp_high = np.empty_like(phi_flat, dtype=float)
    vp_low = np.empty_like(phi_flat, dtype=float)
    for index, (phi_value, sat_value) in enumerate(zip(phi_flat, sat_flat)):
        if phi_value < phi_c:
            k_eff_low = (
                phi_value / phi_c / (k_hm + 4.0 / 3.0 * g_hm)
                + (1.0 - phi_value / phi_c) / (km + 4.0 / 3.0 * g_hm)
            ) ** (-1.0) - 4.0 / 3.0 * g_hm
            one_d_low = (
                g_hm
                / 6.0
                * (9.0 * k_hm + 8.0 * g_hm)
                / (k_hm + 2.0 * g_hm)
            )
            g_eff_low = (
                phi_value / phi_c / (g_hm + one_d_low)
                + (1.0 - phi_value / phi_c) / (gm + one_d_low)
            ) ** (-1.0) - one_d_low

            k_eff_high = (
                phi_value / phi_c / (k_hm + 4.0 / 3.0 * gm)
                + (1.0 - phi_value / phi_c) / (km + 4.0 / 3.0 * gm)
            ) ** (-1.0) - 4.0 / 3.0 * gm
            one_d_high = (
                gm
                / 6.0
                * (9.0 * km + 8.0 * gm)
                / (km + 2.0 * gm)
            )
            g_eff_high = (
                phi_value / phi_c / (g_hm + one_d_high)
                + (1.0 - phi_value / phi_c) / (gm + one_d_high)
            ) ** (-1.0) - one_d_high

            sat_high = vm.satK(k_eff_high, km, phi_value, sat_value)
            sat_low = vm.satK(k_eff_low, km, phi_value, sat_value)
        else:
            k_eff = (
                (1.0 - phi_value)
                / (1.0 - phi_c)
                / (k_hm + 4.0 / 3.0 * g_hm)
                + (phi_value - phi_c)
                / (1.0 - phi_c)
                / (4.0 / 3.0 * g_hm)
            ) ** (-1.0) - 4.0 / 3.0 * g_hm
            one_d = (
                g_hm
                / 6.0
                * (9.0 * k_hm + 8.0 * g_hm)
                / (k_hm + 2.0 * g_hm)
            )
            g_eff = (
                (1.0 - phi_value) / (1.0 - phi_c) / (g_hm + one_d)
                + (phi_value - phi_c) / (1.0 - phi_c) / one_d
            ) ** (-1.0) - one_d

            sat_high = vm.satK(k_eff, km, phi_value, sat_value)
            sat_low = sat_high
            g_eff_high = g_eff
            g_eff_low = g_eff

        rho_air = 1.225
        rho_water = 1000.0
        rho_total = rho_b * (1.0 - phi_value) + (
            sat_value * rho_water + (1.0 - sat_value) * rho_air
        ) * phi_value
        vp_high[index] = np.sqrt((sat_high + 4.0 / 3.0 * g_eff_high) * 1.0e9 / rho_total)
        vp_low[index] = np.sqrt((sat_low + 4.0 / 3.0 * g_eff_low) * 1.0e9 / rho_total)

    return vp_high.reshape(out_shape), vp_low.reshape(out_shape)


def _hertz_mindlin_velocity(phi: np.ndarray, sw: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from PyHydroGeophysX.petrophysics import velocity_models as vm

    phi_arr = np.asarray(phi, dtype=float)
    sw_arr = np.asarray(sw, dtype=float)
    depth = float(params.get("hertz_mindlin_depth_m", 1.0))
    critical_porosity = float(
        params.get("critical_porosity", params.get("phi_c", 0.4))
    )
    if np.isclose(critical_porosity, 0.4):
        high, low = vm.vel_porous(
            phi_arr,
            float(params["Km"]),
            float(params["Gm"]),
            float(params["rho"]),
            sw_arr,
            depth=depth,
        )
    else:
        high, low = _vel_porous_with_critical_porosity(
            phi_arr,
            float(params["Km"]),
            float(params["Gm"]),
            float(params["rho"]),
            sw_arr,
            depth=depth,
            critical_porosity=critical_porosity,
        )
    bound = str(params.get("hertz_mindlin_bound", "mean")).lower()
    if bound == "upper":
        return np.asarray(high, dtype=float)
    if bound == "lower":
        return np.asarray(low, dtype=float)
    if bound == "mean":
        return 0.5 * (np.asarray(high, dtype=float) + np.asarray(low, dtype=float))
    raise ValueError(f"Unknown Hertz-Mindlin bound: {bound}")


def make_vp_model_dict(
    dem: DemGrid,
    depth: np.ndarray,
    vp: np.ndarray,
    cz_unit: np.ndarray,
    **extra_fields: np.ndarray,
) -> dict[str, Any]:
    """Create a model dict compatible with velocity_model section extraction."""
    model: dict[str, Any] = {
        "x": np.asarray(dem.x, dtype=float),
        "y": np.asarray(dem.y, dtype=float),
        "z": np.asarray(depth, dtype=float),
        "vp": np.asarray(vp, dtype=float),
        "unit": np.asarray(cz_unit, dtype=np.int16),
        "topographic_elevation": np.asarray(dem.data, dtype=float),
        "conversion_method": "RD_RPV_voxel_rock_physics",
    }
    model.update(extra_fields)
    return model


def sample_2d_field_along_line(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    line_xy: np.ndarray,
) -> np.ndarray:
    """Sample a 2D grid along x-y line coordinates."""
    from scipy.interpolate import RegularGridInterpolator

    grid_x = np.asarray(x, dtype=float)
    grid_y = np.asarray(y, dtype=float)
    values = np.asarray(field, dtype=float)
    if grid_y[0] > grid_y[-1]:
        grid_y = grid_y[::-1]
        values = values[::-1, :]
    interp = RegularGridInterpolator((grid_y, grid_x), values, bounds_error=False, fill_value=np.nan)
    line = np.asarray(line_xy, dtype=float)
    return interp(np.column_stack([line[:, 1], line[:, 0]]))


