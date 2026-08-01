"""Seismic line discovery and pyGIMLi input preparation.

Each line is described by two files under ``data/seismic/picks``:

``TL<n>.txt``
    First-arrival travel times in the pyGIMLi unified format. A sensor count, a
    ``# x z`` block of along-line distance and relative elevation in metres, a
    pick count, then ``# s g t`` triples of shot index, geophone index, and
    travel time in seconds.

``TL<n>_topo.txt``
    Geophone easting, northing, and cumulative along-line distance in metres,
    NAD83 / UTM Zone 11N (EPSG:26911).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data_io import discover_files
from .seismic_forward import (
    create_mesh_for_data,
    load_line_topography,
    load_pygimli_traveltime_data,
    straighten_line_topography,
)


def _line_lookup(config: dict[str, Any], root: str | Path) -> dict[str, dict[str, Any]]:
    discovered = discover_files(config, root)
    lookup: dict[str, dict[str, Any]] = {}
    for line in discovered["lines"]:
        if line.get("seismic") is None or line.get("topography") is None:
            continue
        seismic = Path(line["seismic"])
        topography = Path(line["topography"])
        if seismic.exists() and topography.exists():
            lookup[str(line["name"])] = dict(line)
    return lookup


def prepare_seismic_lines(
    config: dict[str, Any],
    root: str | Path,
    line_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Load travel-time data, topography, line geometry, and pyGIMLi meshes."""
    root = Path(root)
    lookup = _line_lookup(config, root)
    missing = [name for name in line_names if name not in lookup]
    if missing:
        raise KeyError(
            f"Requested seismic lines were not found: {missing}. "
            f"Discovered lines: {sorted(lookup)}"
        )

    auto_scale = bool(
        config.get("forward", {}).get(
            "time_unit_auto_scale",
            config.get("forward_modeling", {}).get("time_unit_auto_scale", True),
        )
    )
    bottom_depth = float(
        config.get("forward", {}).get(
            "bottom_depth",
            config.get("forward_modeling", {}).get("bottom_depth", 60.0),
        )
    )
    cell_size = float(
        config.get("forward_modeling", {}).get(
            "cell_size",
            config.get("forward", {}).get("cell_size", 8.0),
        )
    )
    assume_straight = bool(
        config.get("forward", {}).get(
            "assume_straight_lines",
            config.get("forward_modeling", {}).get("assume_straight_lines", True),
        )
    )
    prepared: dict[str, dict[str, Any]] = {}
    for name in line_names:
        line = dict(lookup[name])
        tt_data = load_pygimli_traveltime_data(line["seismic"], auto_scale=auto_scale)
        topo = load_line_topography(line["topography"])
        sensors = tt_data["sensors"].sort_values("x")
        sensor_distance = sensors["x"].to_numpy(dtype=float)
        if assume_straight and {"x", "y"}.issubset(topo.columns):
            line_xy, geometry_diagnostics = straighten_line_topography(topo, sensor_distance)
        elif {"x", "y"}.issubset(topo.columns):
            line_xy = topo[["x", "y", "distance"]].to_numpy(dtype=float)
            geometry_diagnostics = {}
        else:
            line_xy = np.column_stack(
                [sensor_distance, np.zeros(len(sensor_distance)), sensor_distance]
            )
            geometry_diagnostics = {}
        mesh = create_mesh_for_data(tt_data["data"], bottom_depth, cell_size)
        line.update(
            {
                "tt_data": tt_data,
                "topography_df": topo,
                "line_xy": line_xy,
                "mesh": mesh,
                "geometry_diagnostics": geometry_diagnostics,
            }
        )
        prepared[name] = line
    return prepared
