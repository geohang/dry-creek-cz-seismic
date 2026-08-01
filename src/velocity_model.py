"""Convert critical-zone layers to P-wave velocity grids."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, brentq

from .dem_tools import line_distance


UNIT_NAMES = {
    1: "regolith",
    2: "fractured_bedrock",
    3: "fresh_bedrock",
}


def _patch_pyhydrogeophysx_root(vm: Any) -> None:
    """Patch PyHydroGeophysX DEM root solving away from SciPy LM on Windows."""

    def scalar_root(fun: Any, x0: Any, args: tuple = (), method: str | None = None, **kwargs: Any) -> OptimizeResult:
        x0f = float(np.atleast_1d(x0)[0])

        def f(x: float) -> float:
            return float(np.atleast_1d(fun(float(x)))[0])

        grid = np.geomspace(max(x0f * 1e-6, 1e-9), max(x0f * 10.0, 1.0), 256)
        previous_x = float(grid[0])
        previous_f = f(previous_x)
        for x in grid[1:]:
            x = float(x)
            fx = f(x)
            if np.isfinite(previous_f) and np.isfinite(fx) and previous_f * fx <= 0:
                root_x = brentq(f, previous_x, x, maxiter=200)
                return OptimizeResult(success=True, x=np.array([root_x]), message="brentq converged", fun=np.array([f(root_x)]))
            previous_x, previous_f = x, fx
        return OptimizeResult(success=False, x=np.array([x0f]), message="no sign-changing bracket", fun=np.array([f(x0f)]))

    vm.root = scalar_root


def extract_velocity_section_for_line(model3d: dict[str, Any], line_xy: np.ndarray) -> dict[str, np.ndarray]:
    """Extract a vertical Vp section along a map-view line."""
    from scipy.interpolate import RegularGridInterpolator

    line_xy = np.asarray(line_xy, dtype=float)
    if line_xy.shape[1] >= 3:
        distance = line_xy[:, 2].astype(float)
    else:
        distance = line_distance(line_xy[:, 0], line_xy[:, 1])

    x_grid = np.asarray(model3d["x"], dtype=float)
    y_grid = np.asarray(model3d["y"], dtype=float)
    z = np.asarray(model3d["z"], dtype=float)
    vp_grid = np.asarray(model3d["vp"], dtype=float)
    if y_grid[0] > y_grid[-1]:
        y_grid = y_grid[::-1]
        vp_grid = vp_grid[:, ::-1, :]

    vp_section = np.full((len(z), len(distance)), np.nan, dtype=float)
    for iz in range(len(z)):
        interp = RegularGridInterpolator((y_grid, x_grid), vp_grid[iz], bounds_error=False, fill_value=np.nan)
        vp_section[iz] = interp(np.column_stack([line_xy[:, 1], line_xy[:, 0]]))
    return {"distance": distance, "depth": z, "vp": vp_section, "line_xy": line_xy[:, :2]}


