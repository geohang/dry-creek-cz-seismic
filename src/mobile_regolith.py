"""Landlab mobile-regolith evolution used by the Dry Creek JGR workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from landlab import NodeStatus, RasterModelGrid
from landlab.components import DepthDependentDiffuser, ExponentialWeatherer

from .dem_tools import DemGrid, fill_nan_nearest


@dataclass(frozen=True)
class MobileRegolithSolution:
    """Fields and diagnostics returned by a Landlab regolith evolution."""

    soil_depth: np.ndarray
    soil_production_rate_m_per_year: np.ndarray
    evolved_surface_elevation: np.ndarray
    evolved_bedrock_elevation: np.ndarray
    evolution_time_years: float
    time_steps: int
    maximum_last_step_soil_change_m: float


def make_landlab_grid(
    shape: tuple[int, int],
    *,
    dx: float,
    dy: float,
) -> RasterModelGrid:
    """Create the raster grid used by the mobile-regolith components."""
    if len(shape) != 2 or min(shape) < 3:
        raise ValueError("Landlab grids must have at least 3 rows and 3 columns.")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive.")
    return RasterModelGrid(shape, xy_spacing=(dx, dy))


def evolve_mobile_regolith(
    grid: RasterModelGrid,
    *,
    channel_mask: np.ndarray,
    initial_soil_depth_m: float,
    minimum_soil_depth_m: float,
    soil_production_maximum_rate_m_per_year: float,
    soil_production_decay_depth_m: float,
    linear_diffusivity_m2_per_year: float,
    soil_transport_decay_depth_m: float,
    evolution_time_years: float,
    time_step_years: float,
) -> MobileRegolithSolution:
    """Evolve mobile regolith with exponential production and depth-dependent transport.

    Production follows ``P0 exp(-Hm / Hs)`` and transport follows
    ``q = -D [1 - exp(-Hm / H0)] grad(eta)``, the formulation reported in the paper.
    """
    if "topographic__elevation" not in grid.at_node:
        raise KeyError("Landlab grid is missing 'topographic__elevation'.")
    channels = np.asarray(channel_mask, dtype=bool)
    if channels.shape != grid.shape:
        raise ValueError("channel_mask shape must match the Landlab grid.")
    if not np.any(channels):
        raise ValueError("channel_mask must contain at least one sink node.")
    if initial_soil_depth_m < minimum_soil_depth_m or minimum_soil_depth_m < 0:
        raise ValueError("initial soil depth must exceed the non-negative minimum.")
    if min(
        soil_production_maximum_rate_m_per_year,
        soil_production_decay_depth_m,
        linear_diffusivity_m2_per_year,
        soil_transport_decay_depth_m,
        evolution_time_years,
        time_step_years,
    ) <= 0:
        raise ValueError("regolith production, transport, and time controls must be positive.")

    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
    channel_nodes = np.flatnonzero(channels.ravel())
    grid.status_at_node[channel_nodes] = NodeStatus.FIXED_VALUE

    surface = np.asarray(grid.at_node["topographic__elevation"], dtype=float)
    initial_surface = surface.copy()
    soil = grid.add_field(
        "soil__depth",
        np.full(grid.number_of_nodes, float(initial_soil_depth_m)),
        at="node",
        clobber=True,
    )
    soil[channel_nodes] = float(minimum_soil_depth_m)
    grid.add_field(
        "bedrock__elevation",
        surface - soil,
        at="node",
        clobber=True,
    )

    weatherer = ExponentialWeatherer(
        grid,
        soil_production_maximum_rate=float(soil_production_maximum_rate_m_per_year),
        soil_production_decay_depth=float(soil_production_decay_depth_m),
    )
    # Landlab takes the diffusivity already divided by the transport decay depth,
    # so the D reported in the paper is D / H0 here.
    diffuser = DepthDependentDiffuser(
        grid,
        linear_diffusivity=(
            float(linear_diffusivity_m2_per_year) / float(soil_transport_decay_depth_m)
        ),
        soil_transport_decay_depth=float(soil_transport_decay_depth_m),
    )

    elapsed = 0.0
    steps = 0
    maximum_last_change = np.inf
    while elapsed < evolution_time_years:
        dt = min(float(time_step_years), float(evolution_time_years) - elapsed)
        previous_soil = soil.copy()
        weatherer.calc_soil_prod_rate()
        diffuser.run_one_step(dt)
        soil[channel_nodes] = float(minimum_soil_depth_m)
        surface[channel_nodes] = initial_surface[channel_nodes]
        grid.at_node["bedrock__elevation"][channel_nodes] = (
            surface[channel_nodes] - soil[channel_nodes]
        )
        maximum_last_change = float(np.max(np.abs(soil - previous_soil)))
        elapsed += dt
        steps += 1

    weatherer.calc_soil_prod_rate()
    shape = grid.shape
    return MobileRegolithSolution(
        soil_depth=soil.reshape(shape).copy(),
        soil_production_rate_m_per_year=grid.at_node["soil_production__rate"]
        .reshape(shape)
        .copy(),
        evolved_surface_elevation=surface.reshape(shape).copy(),
        evolved_bedrock_elevation=grid.at_node["bedrock__elevation"]
        .reshape(shape)
        .copy(),
        evolution_time_years=float(evolution_time_years),
        time_steps=steps,
        maximum_last_step_soil_change_m=maximum_last_change,
    )


def _dem_array(
    dem: DemGrid | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if isinstance(dem, DemGrid):
        return (
            fill_nan_nearest(dem.data),
            np.asarray(dem.x),
            np.asarray(dem.y),
            float(dem.dx),
            float(dem.dy),
        )
    array = fill_nan_nearest(np.asarray(dem, dtype=float))
    return (
        array,
        np.arange(array.shape[1], dtype=float),
        np.arange(array.shape[0], dtype=float),
        1.0,
        1.0,
    )


def generate_mobile_regolith(
    dem: DemGrid | np.ndarray,
    params: dict[str, Any],
    geomorphic_metrics: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Generate the mobile-regolith field used in the RD/RPV calibration."""
    elevation, _, _, dx, dy = _dem_array(dem)
    flip_y = bool(
        isinstance(dem, DemGrid) and len(dem.y) > 1 and dem.y[0] > dem.y[-1]
    )
    elevation_landlab = elevation[::-1].copy() if flip_y else elevation.copy()
    channels = np.asarray(geomorphic_metrics["channel_mask"], dtype=bool)
    channels_landlab = channels[::-1].copy() if flip_y else channels.copy()

    if "regolith_soil_production_decay_depth_m" in params:
        production_decay_depth = float(
            params["regolith_soil_production_decay_depth_m"]
        )
    else:
        decay_coefficient = float(
            params["regolith_soil_production_decay_coefficient_per_m"]
        )
        if decay_coefficient <= 0:
            raise ValueError("soil-production decay coefficient must be positive.")
        production_decay_depth = 1.0 / decay_coefficient
    if production_decay_depth <= 0:
        raise ValueError("soil-production decay depth must be positive.")

    if "regolith_hillslope_diffusivity_m2_per_year" in params:
        hillslope_diffusivity = float(
            params["regolith_hillslope_diffusivity_m2_per_year"]
        )
    else:
        hillslope_diffusivity = float(
            params["regolith_linear_diffusivity_m2_per_year"]
        )
    if hillslope_diffusivity <= 0:
        raise ValueError("hillslope diffusivity must be positive.")

    grid = make_landlab_grid(elevation.shape, dx=dx, dy=dy)
    grid.add_field("topographic__elevation", elevation_landlab.ravel().copy(), at="node")
    solution = evolve_mobile_regolith(
        grid,
        channel_mask=channels_landlab,
        initial_soil_depth_m=float(params["regolith_initial_soil_depth_m"]),
        minimum_soil_depth_m=float(params["regolith_minimum_soil_depth_m"]),
        soil_production_maximum_rate_m_per_year=float(
            params["regolith_soil_production_maximum_rate_m_per_year"]
        ),
        soil_production_decay_depth_m=production_decay_depth,
        linear_diffusivity_m2_per_year=hillslope_diffusivity,
        soil_transport_decay_depth_m=float(params["regolith_transport_decay_depth_m"]),
        evolution_time_years=float(params["regolith_evolution_time_years"]),
        time_step_years=float(params["regolith_time_step_years"]),
    )

    def restore(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        return array[::-1].copy() if flip_y else array.copy()

    return {
        "regolith_thickness": restore(solution.soil_depth),
        "soil_production_rate_m_per_year": restore(
            solution.soil_production_rate_m_per_year
        ),
        "evolved_surface_elevation": restore(solution.evolved_surface_elevation),
        "evolved_bedrock_elevation": restore(solution.evolved_bedrock_elevation),
        "metadata": {
            "pde": "dH/dt = P0*exp(-H/Hs) - div(qs)",
            "landlab_components": "ExponentialWeatherer + DepthDependentDiffuser",
            "boundary_conditions": (
                "Mapped channels are fixed sinks at the minimum soil depth; "
                "outer raster edges have zero normal sediment flux."
            ),
            "evolution_time_years": solution.evolution_time_years,
            "time_steps": solution.time_steps,
            "maximum_last_step_soil_change_m": (
                solution.maximum_last_step_soil_change_m
            ),
        },
    }
