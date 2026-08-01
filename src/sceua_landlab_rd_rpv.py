"""SCE-UA calibration of the critical-zone structure model against travel times.

One objective evaluation runs the full chain: evolve mobile regolith in Landlab,
place the fresh-bedrock interface from the relief ratio, convert the three-unit
structure to P-wave velocity, simulate first arrivals on each line with pyGIMLi,
and score the mean normalized RMSE over the five calibration lines. SPOTPY drives
the search; the best 5% of evaluations form the behavioral ensemble that carries
the reported uncertainty.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import inspect
import math
import os
import random

import numpy as np
import pandas as pd

from .seismic_lines import prepare_seismic_lines
from .mobile_regolith import generate_mobile_regolith
from .flow_routing import CHANNEL_AREA_THRESHOLD_M2, compute_arcgis_flow
from .rd_rpv_helpers import (
    DEFAULT_RPV_PARAMS,
    _dem_velocity,
    _hertz_mindlin_velocity,
    build_cz_unit_grid,
    load_dry_creek_dem,
    make_vp_model_dict,
    sample_2d_field_along_line,
)
from .seismic_forward import (
    compute_travel_time_residuals,
    map_velocity_section_to_mesh,
    simulate_travel_times,
)
from .velocity_model import extract_velocity_section_for_line


SCEUA_PARAMETER_NAMES = (
    "P0",
    "Hs",
    "D",
    "r",
    "phi_soil_top",
    "phi_weathered_top",
    "phi_fresh",
)
LARGE_OBJECTIVE_VALUE = 1.0e12


def _write_path(path: str | Path) -> str:
    """Return a filesystem path string that is safe for long Windows paths."""
    path = Path(path)
    if os.name != "nt":
        return str(path)
    text = str(path.resolve())
    if len(text) < 240 or text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


DEFAULT_SCEUA_CONFIG: dict[str, Any] = {
    "output_dir": "outputs/dry_creek_landlab_strict_rd_relief_ratio_Hs_D_H0_sceua",
    "parameters": {
        "P0": {
            "name": "regolith_soil_production_maximum_rate_m_per_year",
            "min": 3.0e-5,
            "max": 2.0e-4,
        },
        "Hs": {
            "name": "regolith_soil_production_decay_depth_m",
            "min": 0.125,
            "max": 1.0,
        },
        "D": {
            "name": "regolith_hillslope_diffusivity_m2_per_year",
            "min": 1.0e-4,
            "max": 2.0e-3,
        },
        "r": {
            "name": "rd_relief_ratio",
            "min": 0.05,
            "max": 0.95,
        },
        "phi_soil_top": {
            "name": "soil_surface_porosity",
            "min": 0.50,
            "max": 0.75,
        },
        "phi_weathered_top": {
            "name": "soil_weathered_interface_porosity",
            "min": 0.25,
            "max": 0.50,
        },
        "phi_fresh": {
            "name": "fresh_bedrock_porosity",
            "min": 0.05,
            "max": 0.15,
        },
    },
    "landlab_soil": {
        "regolith_initial_soil_depth_m": 1.90,
        "regolith_minimum_soil_depth_m": 0.02,
        "regolith_transport_decay_depth_m": 0.5,
        "regolith_evolution_time_years": 6000.0,
        "regolith_time_step_years": 25.0,
    },
    "rd_geometry": {},
    "forward_model": {
        "bottom_m": 120.0,
        "deep_cell_size_m": 10.0,
    },
    "rpv_params": {
        "basis": "Holbrook_Flinchum_granite_DEM_interface_porosity",
        "soil": {
            "Sw_top": 0.20,
            "Sw_bottom": 0.35,
        },
        "interface_porosity": {
            "phi_fresh": 0.05,
            "eps_m": 1.0e-6,
        },
        "weathered_bedrock": {
            "phi_top": 0.30,
            "phi_bottom": 0.05,
            "Sw_top": 0.40,
            "Sw_bottom": 1.00,
            "alpha_top": 0.016,
            "alpha_bottom": 0.025,
            "Km": 51.69,
            "Gm": 30.38,
            "rho": 2631.0,
        },
        "fresh_bedrock": {
            "phi": 0.05,
            "Sw": 1.00,
            "alpha": 0.025,
            "Km": 51.69,
            "Gm": 30.38,
            "rho": 2631.0,
        },
        "vp_min": 300.0,
        "vp_max": 4500.0,
        "weathered_lookup_samples": 512,
    },
    "training_validation": {
        "training_lines": [],
        "validation_lines": [],
    },
    "objective": {
        "type": "seismic_only",
        "metric": "normalized_rmse",
        "sigma_t_seconds": 0.003,
        "sigma_model_seconds": 0.003,
    },
    "sceua": {
        "repetitions": {
            "smoke": 20,
            "balanced": 800,
            "production": 2500,
        },
        "behavioral_top_fraction": 0.05,
        "minimum_behavioral_sets": 20,
        "random_seed": 42,
    },
    "propagation": {
        "n_behavioral_sets": 200,
        "dz": 1.0,
        "selected_vp_depths_m": [2.0, 10.0, 25.0, 40.0, 80.0],
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_sceua_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the SCE-UA Landlab RD/RPV configuration with defaults filled in."""
    return _deep_update(DEFAULT_SCEUA_CONFIG, dict(config.get("sceua_landlab_rd_rpv", {})))


def rpv_params_from_config(sceua_config: dict[str, Any]) -> dict[str, Any]:
    """Return RD/RPV rock-physics parameters with SCE-UA overrides applied."""
    return _deep_update(
        DEFAULT_RPV_PARAMS,
        dict(sceua_config.get("rpv_params", {})),
    )


def rpv_params_for_candidate(
    candidate: dict[str, float] | pd.Series | np.ndarray | list[float],
    base_rpv_params: dict[str, Any],
) -> dict[str, Any]:
    """Return a candidate-specific deep copy of fixed granite RPV parameters."""
    params = candidate_to_params(candidate)
    rpv = deepcopy(base_rpv_params)
    phi_fresh = float(params["phi_fresh"])
    phi_soil = float(params["phi_soil_top"])
    phi_weathered = float(params["phi_weathered_top"])
    if not (0.0 < phi_fresh < phi_weathered < phi_soil < 1.0):
        raise ValueError(
            "Porosity ordering must satisfy "
            "0 < phi_fresh < phi_weathered_top < phi_soil_top < 1."
        )
    rpv.setdefault("interface_porosity", {})["phi_fresh"] = phi_fresh
    rpv["soil"]["phi"] = phi_soil
    rpv["weathered_bedrock"]["phi_top"] = phi_weathered
    rpv["weathered_bedrock"]["phi_bottom"] = phi_fresh
    rpv["fresh_bedrock"]["phi"] = phi_fresh
    return rpv


def build_interface_porosity_fields(
    cz_unit: np.ndarray,
    h_soil: np.ndarray,
    h_weathered: np.ndarray,
    d_fresh: np.ndarray,
    depth: np.ndarray,
    candidate: dict[str, float] | pd.Series | np.ndarray | list[float],
    rpv_params: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build interface-anchored porosity, saturation, and crack-aspect fields."""
    params = candidate_to_params(candidate)
    unit = np.asarray(cz_unit)
    soil_thickness = np.maximum(np.asarray(h_soil, dtype=float), 0.0)
    weathered_thickness = np.maximum(np.asarray(h_weathered, dtype=float), 0.0)
    fresh_depth = np.maximum(np.asarray(d_fresh, dtype=float), soil_thickness)
    if unit.shape != (len(depth), *soil_thickness.shape):
        raise ValueError("cz_unit must have shape (len(depth), *h_soil.shape).")
    if soil_thickness.shape != weathered_thickness.shape or soil_thickness.shape != fresh_depth.shape:
        raise ValueError("h_soil, h_weathered, and d_fresh must have the same shape.")

    phi_soil = float(params["phi_soil_top"])
    phi_weathered = float(params["phi_weathered_top"])
    phi_fresh = float(params["phi_fresh"])
    if not (0.0 < phi_fresh < phi_weathered < phi_soil < 1.0):
        raise ValueError(
            "Porosity ordering must satisfy "
            "0 < phi_fresh < phi_weathered_top < phi_soil_top < 1."
        )

    eps = float(rpv_params.get("interface_porosity", {}).get("eps_m", 1.0e-6))
    z = np.asarray(depth, dtype=float)[:, None, None]
    soil_3d = soil_thickness[None, :, :]
    weathered_3d = weathered_thickness[None, :, :]

    soil_fraction = np.divide(
        z,
        soil_3d,
        out=np.ones(unit.shape, dtype=float),
        where=soil_3d > eps,
    )
    soil_fraction = np.clip(soil_fraction, 0.0, 1.0)
    weathered_fraction = np.divide(
        z - soil_3d,
        weathered_3d,
        out=np.ones(unit.shape, dtype=float),
        where=weathered_3d > eps,
    )
    weathered_fraction = np.clip(weathered_fraction, 0.0, 1.0)

    phi = np.full(unit.shape, np.nan, dtype=float)
    sw = np.full(unit.shape, np.nan, dtype=float)
    alpha = np.full(unit.shape, np.nan, dtype=float)
    eta = np.full(unit.shape, np.nan, dtype=float)

    soil_mask = unit == 0
    weathered_mask = unit == 1
    fresh_mask = unit == 2

    soil_params = rpv_params["soil"]
    phi[soil_mask] = (
        phi_soil
        + (phi_weathered - phi_soil) * soil_fraction[soil_mask]
    )
    soil_sw_top = float(soil_params.get("Sw_top", soil_params["Sw"]))
    soil_sw_bottom = float(soil_params.get("Sw_bottom", soil_sw_top))
    sw[soil_mask] = soil_sw_top + (
        soil_sw_bottom - soil_sw_top
    ) * soil_fraction[soil_mask]
    alpha[soil_mask] = float(soil_params.get("alpha", 0.15))

    wb_params = rpv_params["weathered_bedrock"]
    eta[weathered_mask] = weathered_fraction[weathered_mask]
    phi[weathered_mask] = (
        phi_weathered
        + (phi_fresh - phi_weathered) * weathered_fraction[weathered_mask]
    )
    sw[weathered_mask] = float(wb_params["Sw_top"]) + (
        float(wb_params["Sw_bottom"]) - float(wb_params["Sw_top"])
    ) * weathered_fraction[weathered_mask]
    alpha[weathered_mask] = float(wb_params["alpha_top"]) + (
        float(wb_params["alpha_bottom"]) - float(wb_params["alpha_top"])
    ) * weathered_fraction[weathered_mask]

    fresh_params = rpv_params["fresh_bedrock"]
    phi[fresh_mask] = phi_fresh
    sw[fresh_mask] = float(fresh_params["Sw"])
    alpha[fresh_mask] = float(fresh_params["alpha"])
    return {
        "phi": phi,
        "Sw": sw,
        "alpha": alpha,
        "eta": eta,
        "W": np.where(np.isfinite(eta), 1.0 - eta, np.nan),
    }


def compute_vp_interface_porosity(
    cz_unit: np.ndarray,
    depth: np.ndarray,
    fields: dict[str, np.ndarray],
    params: dict[str, Any],
) -> dict[str, np.ndarray | str | dict[str, Any]]:
    """Compute Vp from interface-anchored porosity fields."""
    p = deepcopy(params)
    unit = np.asarray(cz_unit)
    vp = np.full(unit.shape, np.nan, dtype=float)

    soil = unit == 0
    if np.any(soil):
        vp[soil] = _hertz_mindlin_velocity(
            fields["phi"][soil],
            fields["Sw"][soil],
            p["soil"],
        )

    wb = unit == 1
    if np.any(wb):
        wb_params = p["weathered_bedrock"]
        eta_wb = np.asarray(fields["eta"][wb], dtype=float)
        lookup_samples = int(p.get("weathered_lookup_samples", 512))
        if lookup_samples > 1:
            eta_lookup = np.linspace(0.0, 1.0, lookup_samples)
            phi_lookup = float(wb_params["phi_top"]) + (
                float(wb_params["phi_bottom"]) - float(wb_params["phi_top"])
            ) * eta_lookup
            sw_lookup = float(wb_params["Sw_top"]) + (
                float(wb_params["Sw_bottom"]) - float(wb_params["Sw_top"])
            ) * eta_lookup
            alpha_lookup = float(wb_params["alpha_top"]) + (
                float(wb_params["alpha_bottom"]) - float(wb_params["alpha_top"])
            ) * eta_lookup
            vp_lookup = _dem_velocity(
                phi_lookup,
                sw_lookup,
                alpha_lookup,
                wb_params,
            )
            vp[wb] = np.interp(
                np.nan_to_num(eta_wb, nan=0.0),
                eta_lookup,
                vp_lookup,
            )
        else:
            vp[wb] = _dem_velocity(
                fields["phi"][wb],
                fields["Sw"][wb],
                fields["alpha"][wb],
                wb_params,
            )

    fresh = unit == 2
    if np.any(fresh):
        fresh_params = p["fresh_bedrock"]
        vp[fresh] = float(_dem_velocity(
            np.array([float(fresh_params["phi"])]),
            np.array([float(fresh_params["Sw"])]),
            np.array([float(fresh_params["alpha"])]),
            fresh_params,
        )[0])

    vp = np.clip(vp, float(p["vp_min"]), float(p["vp_max"]))
    return {
        "vp": vp,
        "phi": fields["phi"],
        "Sw": fields["Sw"],
        "alpha": fields["alpha"],
        "eta": fields["eta"],
        "W": fields["W"],
        "method": "interface-anchored porosity + vel_porous soil + granite velDEM bedrock",
        "params": p,
        "depth": np.asarray(depth, dtype=float),
    }


def parameter_bounds(sceua_config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Extract ordered SCE-UA parameter bounds."""
    params = sceua_config.get("parameters", {})
    bounds: dict[str, tuple[float, float]] = {}
    for name in SCEUA_PARAMETER_NAMES:
        spec = params.get(name, {})
        lower = float(spec.get("min"))
        upper = float(spec.get("max"))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid SCE-UA bounds for {name}: {spec}")
        bounds[name] = (lower, upper)
    return bounds


def fixed_parameter_values(sceua_config: dict[str, Any]) -> dict[str, float]:
    """Return validated SCE-UA parameters that should remain constant."""
    raw = dict(sceua_config.get("fixed_parameters", {}))
    unknown = sorted(set(raw) - set(SCEUA_PARAMETER_NAMES))
    if unknown:
        raise KeyError(f"Unknown fixed SCE-UA parameters: {unknown}")
    fixed = {name: float(value) for name, value in raw.items()}
    for name, value in fixed.items():
        if not np.isfinite(value):
            raise ValueError(f"Fixed SCE-UA parameter {name} must be finite.")
        if name.startswith("phi_") and not 0.0 < value < 1.0:
            raise ValueError(f"Fixed porosity parameter {name} must be between 0 and 1.")
    return fixed


def candidate_to_params(candidate: dict[str, float] | pd.Series | np.ndarray | list[float]) -> dict[str, float]:
    """Convert a candidate to the ordered seven-parameter dictionary.

    Accepts a mapping keyed by parameter name, the SPOTPY ``par<name>`` form, or
    a positional sequence in ``SCEUA_PARAMETER_NAMES`` order.
    """
    if isinstance(candidate, pd.Series):
        candidate = candidate.to_dict()
    if isinstance(candidate, dict):
        out: dict[str, float] = {}
        for name in SCEUA_PARAMETER_NAMES:
            if name in candidate:
                out[name] = float(candidate[name])
            elif f"par{name}" in candidate:
                out[name] = float(candidate[f"par{name}"])
            else:
                raise KeyError(f"Candidate is missing parameter {name}.")
        return out
    values = np.asarray(candidate, dtype=float)
    if values.size != len(SCEUA_PARAMETER_NAMES):
        raise ValueError(f"candidate must have {len(SCEUA_PARAMETER_NAMES)} values.")
    return {name: float(value) for name, value in zip(SCEUA_PARAMETER_NAMES, values)}


def normalize_sceua_parameter_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame unchanged once the seven parameter columns are present."""
    missing = [name for name in SCEUA_PARAMETER_NAMES if name not in frame.columns]
    if missing:
        raise KeyError(f"Parameter frame is missing columns: {missing}")
    return frame.copy()


def candidate_to_landlab_params(
    candidate: dict[str, float] | pd.Series | np.ndarray | list[float],
    sceua_config: dict[str, Any],
) -> dict[str, Any]:
    """Map calibrated parameters onto the Landlab soil-production controls."""
    params = candidate_to_params(candidate)
    landlab_params = dict(sceua_config.get("landlab_soil", {}))
    for name in ("P0", "Hs", "D"):
        config_name = str(sceua_config["parameters"][name]["name"])
        landlab_params[config_name] = params[name]
    return landlab_params


def get_line_split(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return the calibration and held-out validation lines from the config."""
    sceua = load_sceua_config(config)
    line_cfg = sceua.get("training_validation", {})
    training = [str(name) for name in line_cfg.get("training_lines", [])]
    validation = [str(name) for name in line_cfg.get("validation_lines", [])]
    if not training or not validation:
        raise ValueError(
            "sceua_landlab_rd_rpv.training_validation must list 5 training_lines "
            "and 2 validation_lines."
        )
    overlap = sorted(set(training) & set(validation))
    if overlap:
        raise ValueError(f"Validation lines also appear in training: {overlap}")
    if len(training) != 5 or len(validation) != 2:
        raise ValueError(
            "SCE-UA RD/RPV workflow requires exactly 5 training lines and "
            "2 held-out validation lines."
        )
    return training, validation


def validate_forward_depth(sceua_config: dict[str, Any]) -> None:
    """Raise if the forward domain depth setting is invalid."""
    bottom = float(sceua_config["forward_model"]["bottom_m"])
    if bottom <= 0.0:
        raise ValueError("forward_model.bottom_m must be positive.")


def make_forward_config(config: dict[str, Any], sceua_config: dict[str, Any]) -> dict[str, Any]:
    """Return a runtime config with the SCE-UA forward-model depth controls."""
    runtime = deepcopy(config)
    forward = runtime.setdefault("forward", {})
    forward_modeling = runtime.setdefault("forward_modeling", {})
    bottom = float(sceua_config["forward_model"]["bottom_m"])
    deep_cell_size = float(sceua_config["forward_model"]["deep_cell_size_m"])
    runtime["model_bottom_depth"] = bottom
    forward["bottom_depth"] = bottom
    forward["cell_size"] = deep_cell_size
    forward_modeling["bottom_depth"] = bottom
    forward_modeling["cell_size"] = deep_cell_size
    return runtime


def load_sceua_base_inputs(
    config: dict[str, Any],
    root: str | Path,
    *,
    line_names: list[str] | None = None,
    target_resolution_m: float | None = None,
) -> dict[str, Any]:
    """Load DEM, flow/Zs, Landlab controls, and reusable pyGIMLi line meshes."""
    root = Path(root)
    sceua = load_sceua_config(config)
    validate_forward_depth(sceua)
    forward_config = make_forward_config(config, sceua)
    dem = load_dry_creek_dem(
        config,
        root,
        target_resolution_m=target_resolution_m,
    )
    landlab_config = config.get("landlab_evolution", {})
    if "flow_source" not in landlab_config:
        raise ValueError(
            "landlab_evolution.flow_source must be set explicitly. "
            "Use 'arcgis' for the ArcGIS flow-raster workflow."
        )
    flow_source = str(landlab_config["flow_source"]).lower()
    if flow_source == "arcgis":
        required = {
            "flow_direction_file": landlab_config.get("flow_direction_file"),
            "flow_accumulation_file": landlab_config.get("flow_accumulation_file"),
            "channel_raster_file": landlab_config.get("channel_raster_file"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(
                "ArcGIS flow_source requires " + ", ".join(missing) + "."
            )
        flow = compute_arcgis_flow(
            dem,
            flow_dem_path=root / str(
                landlab_config.get("flow_dem_file", config.get("dem_file", ""))
            ),
            flow_direction_path=root / str(required["flow_direction_file"]),
            flow_accumulation_path=root / str(required["flow_accumulation_file"]),
            channel_raster_path=root / str(required["channel_raster_file"]),
            channel_threshold_m2=float(
                landlab_config.get(
                    "channel_drainage_area_threshold",
                    CHANNEL_AREA_THRESHOLD_M2,
                )
            ),
            include_outflow_nodes_as_channels=bool(
                landlab_config.get("include_arcgis_outflow_nodes_as_channels", True)
            ),
            smooth_hand_sigma_cells=float(
                landlab_config.get("smooth_hand_sigma_cells", 0.0)
            ),
        )
    else:
        raise ValueError(
            f"Unsupported flow_source: {flow_source!r}. This workflow routes flow "
            "from the ArcGIS D8 products, so flow_source must be 'arcgis'."
        )
    dz = float(sceua.get("propagation", {}).get("dz", config.get("dz", 1.0)))
    max_depth = float(sceua["forward_model"]["bottom_m"])
    depth = np.arange(0.0, max_depth + dz, dz, dtype=float)
    if line_names is None:
        training, validation = get_line_split(config)
        line_names = training + validation
    else:
        training, validation = get_line_split(config)
    lines = prepare_seismic_lines(forward_config, root, list(line_names))
    return {
        "root": root,
        "dem": dem,
        "flow": flow,
        "Zs": np.asarray(flow.height_above_channel_m, dtype=float),
        "depth": depth,
        "lines": lines,
        "training_lines": training,
        "validation_lines": validation,
        "rpv_params": rpv_params_from_config(sceua),
        "sceua_config": sceua,
        "forward_config": forward_config,
        "mesh_note": (
            "Small lines use terrain-following 0-5, 5-20, and 20-120 m regions "
            "with target triangle sizes of 1, 2, and 10 m, respectively."
        ),
    }


def run_landlab_soil(
    candidate: dict[str, float] | pd.Series | np.ndarray | list[float],
    base_inputs: dict[str, Any],
    sceua_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the Landlab soil-production/diffusion model for one candidate."""
    if sceua_config is None:
        sceua_config = base_inputs.get("sceua_config", load_sceua_config({}))
    landlab_params = candidate_to_landlab_params(candidate, sceua_config)
    regolith = generate_mobile_regolith(
        base_inputs["dem"],
        landlab_params,
        {"channel_mask": base_inputs["flow"].channel_mask},
    )
    h_soil = np.asarray(regolith["regolith_thickness"], dtype=float)
    minimum = float(landlab_params["regolith_minimum_soil_depth_m"])
    h_soil = np.maximum(h_soil, minimum)
    regolith["regolith_thickness"] = h_soil
    regolith["landlab_params"] = landlab_params
    return regolith


def _cached_landlab_soil(
    params: dict[str, float],
    base_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a compatible fixed-parameter Landlab cache when available."""
    cache = base_inputs.get("landlab_soil_cache")
    if not isinstance(cache, dict) or "regolith" not in cache:
        return None
    cached_values = cache.get("parameter_values", {})
    if all(
        np.isclose(float(params[name]), float(cached_values.get(name, np.nan)))
        for name in ("P0", "Hs", "D")
    ):
        return cache["regolith"]
    return None


def rd_geometry_from_candidate(
    candidate: dict[str, float] | pd.Series | np.ndarray | list[float],
    h_soil: np.ndarray,
    base_inputs: dict[str, Any],
    sceua_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build strict RD relief-ratio geometry and physical ordering diagnostics."""
    if sceua_config is None:
        sceua_config = base_inputs.get("sceua_config", load_sceua_config({}))
    params = candidate_to_params(candidate)
    relief = np.maximum(np.asarray(base_inputs["Zs"], dtype=float), 0.0)
    soil = np.asarray(h_soil, dtype=float)
    z_b_relief = params["r"] * relief
    d_fresh_raw = np.maximum(relief - z_b_relief, 0.0)
    d_fresh_geometric = d_fresh_raw
    soil_exceeds_rd_depth = soil > d_fresh_geometric
    d_fresh = np.maximum(d_fresh_geometric, soil)
    h_weathered_raw = d_fresh_raw - soil
    h_weathered = np.maximum(d_fresh - soil, 0.0)
    return {
        "Zs": relief,
        "Zb": z_b_relief,
        "Zb_relief": z_b_relief,
        "D_fresh_raw": d_fresh_raw,
        "D_fresh_geometric": d_fresh_geometric,
        "H_soil": soil,
        "H_weathered_raw": h_weathered_raw,
        "H_weathered": h_weathered,
        "D_fresh": d_fresh,
        "soil_exceeds_rd_depth_fraction": float(np.mean(soil_exceeds_rd_depth)),
        "r": float(params["r"]),
    }


def build_model_from_candidate(
    candidate: dict[str, float] | pd.Series | np.ndarray | list[float],
    base_inputs: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full 3-D Landlab RD/RPV model for one candidate."""
    sceua = base_inputs.get("sceua_config", load_sceua_config(config or {}))
    params = candidate_to_params(candidate)
    if not 0.0 <= params["r"] <= 1.0:
        raise ValueError("r must be between 0 and 1.")
    rpv_params = rpv_params_for_candidate(
        params,
        base_inputs.get("rpv_params", DEFAULT_RPV_PARAMS),
    )
    regolith = _cached_landlab_soil(params, base_inputs)
    if regolith is None:
        regolith = run_landlab_soil(params, base_inputs, sceua)
    h_soil = np.asarray(regolith["regolith_thickness"], dtype=float)
    geometry = rd_geometry_from_candidate(params, h_soil, base_inputs, sceua)
    depth = np.asarray(base_inputs["depth"], dtype=float)
    unit = build_cz_unit_grid(h_soil, geometry["D_fresh"], depth)
    fields = build_interface_porosity_fields(
        unit,
        h_soil,
        geometry["H_weathered"],
        geometry["D_fresh"],
        depth,
        params,
        rpv_params,
    )
    rpv = compute_vp_interface_porosity(
        unit,
        depth,
        fields,
        rpv_params,
    )
    model = make_vp_model_dict(
        base_inputs["dem"],
        depth,
        rpv["vp"],
        unit,
        phi=rpv["phi"],
        Sw=rpv["Sw"],
        alpha=rpv["alpha"],
        Zs=geometry["Zs"],
        Zb_relief=geometry["Zb_relief"],
        D_fresh_raw=geometry["D_fresh_raw"],
        D_fresh_geometric=geometry["D_fresh_geometric"],
        H_soil=h_soil,
        H_weathered=geometry["H_weathered"],
        H_weathered_raw=geometry["H_weathered_raw"],
        D_fresh=geometry["D_fresh"],
        eta=rpv["eta"],
        W=rpv["W"],
    )
    model["candidate_params"] = params
    model["rpv_params"] = rpv["params"]
    model["landlab_params"] = regolith["landlab_params"]
    model["landlab_metadata"] = regolith.get("metadata", {})
    model["soil_exceeds_rd_depth_fraction"] = geometry["soil_exceeds_rd_depth_fraction"]
    return model


def _fill_line_values(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.all(np.isfinite(arr)):
        return arr
    replacement = float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else fallback
    return np.nan_to_num(arr, nan=replacement)


def _extract_3d_field_section(
    model: dict[str, Any],
    field_name: str,
    line_xy: np.ndarray,
) -> np.ndarray:
    """Extract one continuous 3-D model field along a map-view line."""
    from scipy.interpolate import RegularGridInterpolator

    x_grid = np.asarray(model["x"], dtype=float)
    y_grid = np.asarray(model["y"], dtype=float)
    z = np.asarray(model["z"], dtype=float)
    field = np.asarray(model[field_name], dtype=float)
    if y_grid[0] > y_grid[-1]:
        y_grid = y_grid[::-1]
        field = field[:, ::-1, :]

    section = np.full((len(z), len(line_xy)), np.nan, dtype=float)
    sample_points = np.column_stack([line_xy[:, 1], line_xy[:, 0]])
    for iz in range(len(z)):
        interp = RegularGridInterpolator(
            (y_grid, x_grid),
            field[iz],
            bounds_error=False,
            fill_value=np.nan,
        )
        section[iz] = interp(sample_points)
    return section


def build_line_section_from_model(
    model: dict[str, Any],
    line: dict[str, Any],
    base_inputs: dict[str, Any],
) -> dict[str, np.ndarray | dict[str, float]]:
    """Extract a terrain-following Vp section from a full 3-D candidate model."""
    line_xy = np.asarray(line["line_xy"], dtype=float)
    section = extract_velocity_section_for_line(model, line_xy)
    dem = base_inputs["dem"]
    for key, fallback in (
        ("Zs", 0.0),
        ("Zb_relief", 0.0),
        ("H_soil", 0.6),
        ("H_weathered", 0.0),
        ("D_fresh_raw", 1.0),
        ("D_fresh_geometric", 1.0),
        ("D_fresh", 1.0),
    ):
        if key in model:
            section[key] = _fill_line_values(
                sample_2d_field_along_line(model[key], dem.x, dem.y, line_xy),
                fallback=fallback,
            )
    for key in ("phi", "Sw", "alpha", "eta", "W"):
        if key in model:
            section[key] = _extract_3d_field_section(model, key, line_xy)
    sensors = line["tt_data"]["sensors"].sort_values("x")
    section["surface_elevation"] = np.interp(
        line_xy[:, 2],
        sensors["x"].to_numpy(dtype=float),
        sensors["elevation"].to_numpy(dtype=float),
    )
    section["theta"] = dict(model["candidate_params"])
    return section


def _sigma_eff(
    observations: pd.DataFrame,
    sceua_config: dict[str, Any],
) -> np.ndarray:
    objective = sceua_config.get("objective", {})
    sigma = np.full(len(observations), float(objective.get("sigma_t_seconds", 0.003)))
    if "err" in observations:
        err = observations["err"].to_numpy(dtype=float)
        sigma = np.where(np.isfinite(err) & (err > 0.0), err, sigma)
    sigma_model = float(objective.get("sigma_model_seconds", 0.003))
    return np.sqrt(sigma**2 + sigma_model**2)


def forward_predict_line(
    model: dict[str, Any],
    line: dict[str, Any],
    base_inputs: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Forward predict first-arrival times for one prepared seismic line."""
    runtime_config = base_inputs.get("forward_config", config or {})
    sceua = base_inputs.get("sceua_config", load_sceua_config(config or {}))
    section = build_line_section_from_model(model, line, base_inputs)
    cell_velocity = map_velocity_section_to_mesh(line["mesh"], section)
    sec_nodes = int(
        runtime_config.get("forward_modeling", {}).get(
            "sec_nodes",
            runtime_config.get("forward", {}).get("sec_nodes", 3),
        )
    )
    simulated = simulate_travel_times(
        line["mesh"],
        line["tt_data"]["data"],
        cell_velocity,
        sec_nodes=sec_nodes,
    )
    observations = line["tt_data"]["observations"]
    observed = observations["t"].to_numpy(dtype=float)
    predicted = np.asarray(simulated["predicted"], dtype=float)
    sigma = _sigma_eff(observations, sceua)
    n = min(observed.size, predicted.size, sigma.size)
    residuals = compute_travel_time_residuals(observed[:n], predicted[:n], err=sigma[:n])
    normalized = (predicted[:n] - observed[:n]) / sigma[:n]
    return {
        "line": line.get("name", "unknown"),
        "section": section,
        "observed": observed[:n],
        "predicted": predicted[:n],
        "sigma_eff": sigma[:n],
        "offset": observations["offset"].to_numpy(dtype=float)[:n]
        if "offset" in observations
        else np.arange(n, dtype=float),
        "residual": predicted[:n] - observed[:n],
        "rmse_s": float(residuals["rmse"]),
        "normalized_rmse": float(np.sqrt(np.nanmean(normalized**2))),
        "n_picks": int(n),
        "status": "ok",
    }


def seismic_objective_from_predictions(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Compute a seismic-only objective as an unweighted mean across lines."""
    if not predictions:
        return {
            "objective_value": LARGE_OBJECTIVE_VALUE,
            "mean_training_rmse_s": math.inf,
            "mean_training_normalized_rmse": math.inf,
            "n_picks": 0,
        }
    line_rmse = []
    line_normalized_rmse = []
    n_picks = 0
    line_metrics: dict[str, float] = {}
    for item in predictions:
        residual = np.asarray(item["residual"], dtype=float)
        sigma = np.asarray(item["sigma_eff"], dtype=float)
        n = min(residual.size, sigma.size)
        ok = np.isfinite(residual[:n]) & np.isfinite(sigma[:n]) & (sigma[:n] > 0.0)
        if np.any(ok):
            residual_ok = residual[:n][ok]
            normalized_ok = residual_ok / sigma[:n][ok]
            rmse = float(np.sqrt(np.nanmean(residual_ok**2)))
            normalized_rmse = float(np.sqrt(np.nanmean(normalized_ok**2)))
            line_rmse.append(rmse)
            line_normalized_rmse.append(normalized_rmse)
            n_line = int(np.sum(ok))
            n_picks += n_line
            line_name = _safe_column_name(str(item.get("line", f"line_{len(line_rmse)}")))
            line_metrics[f"{line_name}_rmse_s"] = rmse
            line_metrics[f"{line_name}_normalized_rmse"] = normalized_rmse
            line_metrics[f"{line_name}_n_picks"] = float(n_line)
    if not line_normalized_rmse:
        return {
            "objective_value": LARGE_OBJECTIVE_VALUE,
            "mean_training_rmse_s": math.inf,
            "mean_training_normalized_rmse": math.inf,
            "n_picks": 0,
        }
    objective = float(np.nanmean(line_normalized_rmse))
    return {
        "objective_value": objective,
        "mean_training_rmse_s": float(np.nanmean(line_rmse)),
        "mean_training_normalized_rmse": objective,
        "n_picks": int(n_picks),
        **line_metrics,
    }


def _safe_column_name(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)


def _model_diagnostics(model: dict[str, Any]) -> dict[str, float]:
    diagnostics = {
        "soil_exceeds_rd_depth_fraction": float(model.get("soil_exceeds_rd_depth_fraction", 0.0)),
        "mean_H_soil_m": float(np.nanmean(model["H_soil"])),
        "mean_H_weathered_m": float(np.nanmean(model["H_weathered"])),
        "mean_D_fresh_m": float(np.nanmean(model["D_fresh"])),
        "max_H_soil_m": float(np.nanmax(model["H_soil"])),
        "max_H_weathered_m": float(np.nanmax(model["H_weathered"])),
        "max_D_fresh_m": float(np.nanmax(model["D_fresh"])),
        "mean_H_m_m": float(np.nanmean(model["H_soil"])),
        "max_H_m_m": float(np.nanmax(model["H_soil"])),
        "mean_H_w_m": float(np.nanmean(model["H_weathered"])),
        "max_H_w_m": float(np.nanmax(model["H_weathered"])),
        "mean_D_f_m": float(np.nanmean(model["D_fresh"])),
        "max_D_f_m": float(np.nanmax(model["D_fresh"])),
        "mean_h_weathered": float(np.nanmean(model["H_weathered"])),
        "max_h_weathered": float(np.nanmax(model["H_weathered"])),
        "mean_d_fresh": float(np.nanmean(model["D_fresh"])),
        "max_d_fresh": float(np.nanmax(model["D_fresh"])),
    }
    if "Zs" in model:
        diagnostics["mean_z_s_relief"] = float(np.nanmean(model["Zs"]))
        diagnostics["max_z_s_relief"] = float(np.nanmax(model["Zs"]))
        diagnostics["mean_Zs_m"] = float(np.nanmean(model["Zs"]))
        diagnostics["max_Zs_m"] = float(np.nanmax(model["Zs"]))
    if "Zb_relief" in model:
        diagnostics["mean_z_b_relief"] = float(np.nanmean(model["Zb_relief"]))
        diagnostics["max_z_b_relief"] = float(np.nanmax(model["Zb_relief"]))
        diagnostics["mean_Zb_m"] = float(np.nanmean(model["Zb_relief"]))
        diagnostics["max_Zb_m"] = float(np.nanmax(model["Zb_relief"]))
    if "D_fresh_raw" in model:
        diagnostics["mean_D_fresh_raw_m"] = float(np.nanmean(model["D_fresh_raw"]))
        diagnostics["max_D_fresh_raw_m"] = float(np.nanmax(model["D_fresh_raw"]))
    if "D_fresh_geometric" in model:
        diagnostics["mean_D_fresh_geometric_m"] = float(np.nanmean(model["D_fresh_geometric"]))
        diagnostics["max_D_fresh_geometric_m"] = float(np.nanmax(model["D_fresh_geometric"]))
    unit = np.asarray(model["unit"])
    labels = {
        0: "soil",
        1: "weathered",
        2: "fresh",
    }
    for unit_id, label in labels.items():
        mask = unit == unit_id
        for field_name, output_name in (
            ("vp", "Vp_m_per_s"),
            ("phi", "phi"),
        ):
            values = np.asarray(model[field_name], dtype=float)[mask]
            finite = values[np.isfinite(values)]
            if finite.size:
                diagnostics[f"min_{output_name}_{label}"] = float(np.nanmin(finite))
                diagnostics[f"mean_{output_name}_{label}"] = float(np.nanmean(finite))
                diagnostics[f"max_{output_name}_{label}"] = float(np.nanmax(finite))
            else:
                diagnostics[f"min_{output_name}_{label}"] = math.nan
                diagnostics[f"mean_{output_name}_{label}"] = math.nan
                diagnostics[f"max_{output_name}_{label}"] = math.nan
    return diagnostics


def evaluate_candidate(
    candidate: dict[str, float] | pd.Series | np.ndarray | list[float],
    training_lines: list[str],
    base_inputs: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    candidate_id: int | None = None,
) -> dict[str, Any]:
    """Evaluate one SCE-UA candidate using the seismic-only training objective."""
    params = candidate_to_params(candidate)
    row: dict[str, Any] = {
        "candidate_id": -1 if candidate_id is None else int(candidate_id),
        **params,
        "objective_value": LARGE_OBJECTIVE_VALUE,
        "mean_training_rmse_s": math.inf,
        "mean_training_normalized_rmse": math.inf,
        "n_picks": 0,
        "n_failed_lines": len(training_lines),
        "status": "failed",
        "message": "",
    }
    try:
        model = build_model_from_candidate(params, base_inputs, config)
        row.update(_model_diagnostics(model))
        predictions = []
        failed_lines = []
        for line_id in training_lines:
            try:
                predictions.append(
                    forward_predict_line(
                        model,
                        base_inputs["lines"][line_id],
                        base_inputs,
                        config,
                    )
                )
            except Exception as exc:
                failed_lines.append(f"{line_id}: {exc}")
        row["n_failed_lines"] = len(failed_lines)
        row["message"] = "; ".join(failed_lines)
        if failed_lines:
            row["status"] = "failed"
            return row
        objective = seismic_objective_from_predictions(predictions)
        row.update(objective)
        row["status"] = "ok" if np.isfinite(row["objective_value"]) else "failed"
    except Exception as exc:
        row["message"] = str(exc)
    return row


def select_behavioral_sets(
    results: pd.DataFrame,
    *,
    top_fraction: float = 0.05,
    minimum_sets: int = 20,
) -> pd.DataFrame:
    """Select the top behavioral parameter sets by seismic objective."""
    if len(results) == 0:
        return results.copy()
    frame = normalize_sceua_parameter_frame(results)
    frame = frame[np.isfinite(frame["objective_value"].to_numpy(dtype=float))]
    frame = frame.sort_values("objective_value", ascending=True).reset_index(drop=True)
    if frame.empty:
        return frame
    n_top = int(math.ceil(float(top_fraction) * len(frame)))
    n_select = min(len(frame), max(int(minimum_sets), n_top, 1))
    selected = frame.iloc[:n_select].copy().reset_index(drop=True)
    selected.insert(0, "behavioral_id", np.arange(len(selected), dtype=int))
    return selected


def summarize_behavioral_ensemble(behavioral_sets: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibrated behavioral parameter ranges."""
    rows = []
    for name in SCEUA_PARAMETER_NAMES:
        values = behavioral_sets[name].to_numpy(dtype=float)
        rows.append(
            {
                "parameter": name,
                "p05": float(np.nanpercentile(values, 5)),
                "median": float(np.nanpercentile(values, 50)),
                "p95": float(np.nanpercentile(values, 95)),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }
        )
    return pd.DataFrame(rows)


class SpotpySceuaSetup:
    """SPOTPY setup object wrapping the local seismic forward model."""

    def __init__(
        self,
        training_lines: list[str],
        base_inputs: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        self.training_lines = list(training_lines)
        self.base_inputs = base_inputs
        self.config = config
        self.sceua_config = base_inputs.get("sceua_config", load_sceua_config(config))
        self.bounds = parameter_bounds(self.sceua_config)
        self.fixed_parameters = fixed_parameter_values(self.sceua_config)
        self.records: list[dict[str, Any]] = []

    def parameters(self) -> np.ndarray:
        """Return SPOTPY's structured parameter array."""
        import spotpy

        parameters = []
        for name, limits in self.bounds.items():
            if name in self.fixed_parameters:
                parameters.append(
                    spotpy.parameter.Constant(name, self.fixed_parameters[name])
                )
            else:
                parameters.append(spotpy.parameter.Uniform(name, limits[0], limits[1]))
        return spotpy.parameter.generate(parameters)

    def simulation(self, vector: Any) -> list[float]:
        """Evaluate one SPOTPY candidate and return a minimized scalar."""
        candidate = candidate_to_params(vector)
        row = evaluate_candidate(
            candidate,
            self.training_lines,
            self.base_inputs,
            self.config,
            candidate_id=len(self.records),
        )
        self.records.append(row)
        return [float(row["objective_value"])]

    def evaluation(self) -> list[float]:
        """SPOTPY target value for objective-function compatibility."""
        return [0.0]

    def objectivefunction(self, simulation: list[float], evaluation: list[float]) -> float:
        """Return the normalized RMSE for SCE-UA minimization."""
        return float(simulation[0])


def _sceua_repetitions(sceua_config: dict[str, Any], preset: str | int) -> int:
    if isinstance(preset, int):
        return int(preset)
    repetitions = sceua_config.get("sceua", {}).get("repetitions", {})
    if str(preset) not in repetitions:
        raise KeyError(f"Unknown SCE-UA repetition preset: {preset}")
    return int(repetitions[str(preset)])


def run_sceua_calibration(
    training_lines: list[str],
    base_inputs: dict[str, Any],
    config: dict[str, Any],
    *,
    preset: str | int = "smoke",
    output_dir: str | Path | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Run SPOTPY SCE-UA and retain the calibrated behavioral ensemble."""
    try:
        import spotpy
    except ImportError as exc:
        raise ImportError(
            "SPOTPY is required for SCE-UA calibration. Create the environment "
            "from environment.yml, or install it with "
            "`conda install -c conda-forge spotpy`."
        ) from exc

    sceua = base_inputs.get("sceua_config", load_sceua_config(config))
    validate_forward_depth(sceua)
    seed = int(sceua.get("sceua", {}).get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    repetitions = _sceua_repetitions(sceua, preset)
    out = Path(output_dir or sceua["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    setup = SpotpySceuaSetup(training_lines, base_inputs, config)
    init_kwargs: dict[str, Any] = {
        "dbname": str(out / "spotpy_sceua"),
        "dbformat": "csv" if save_outputs else "noData",
    }
    sceua_signature = inspect.signature(spotpy.algorithms.sceua)
    if "save_sim" in sceua_signature.parameters:
        init_kwargs["save_sim"] = False
    if "optimization_direction" in sceua_signature.parameters:
        init_kwargs["optimization_direction"] = "minimize"
    sampler = spotpy.algorithms.sceua(setup, **init_kwargs)

    sample_kwargs: dict[str, Any] = {}
    sample_cfg = sceua.get("sceua", {})
    sample_signature = inspect.signature(sampler.sample)
    for key in ("ngs", "kstop", "pcento", "peps"):
        if key in sample_cfg and key in sample_signature.parameters:
            sample_kwargs[key] = sample_cfg[key]
    sampler.sample(repetitions, **sample_kwargs)

    results = pd.DataFrame(setup.records)
    if results.empty:
        raise RuntimeError("SCE-UA finished without evaluating any candidates.")
    results = results.sort_values("objective_value", ascending=True).reset_index(drop=True)
    behavioral = select_behavioral_sets(
        results,
        top_fraction=float(sceua["sceua"].get("behavioral_top_fraction", 0.05)),
        minimum_sets=int(sceua["sceua"].get("minimum_behavioral_sets", 20)),
    )
    best = results.iloc[[0]].copy().reset_index(drop=True)
    output = {
        "method": "spotpy_sceua",
        "parameter_names": list(SCEUA_PARAMETER_NAMES),
        "training_lines": list(training_lines),
        "validation_lines": list(base_inputs.get("validation_lines", [])),
        "sceua_config": sceua,
        "repetitions": repetitions,
        "records": results,
        "behavioral_sets": behavioral,
        "behavioral_summary": summarize_behavioral_ensemble(behavioral),
        "best_parameters": best,
        "output_dir": out,
    }
    if save_outputs:
        save_sceua_outputs(output, out)
    return output


def _behavioral_frame(
    behavioral_sets: pd.DataFrame | np.ndarray | list[dict[str, float]],
) -> pd.DataFrame:
    if isinstance(behavioral_sets, pd.DataFrame):
        return normalize_sceua_parameter_frame(behavioral_sets).reset_index(drop=True)
    if isinstance(behavioral_sets, list) and behavioral_sets and isinstance(behavioral_sets[0], dict):
        return normalize_sceua_parameter_frame(pd.DataFrame(behavioral_sets))
    arr = np.asarray(behavioral_sets, dtype=float)
    return pd.DataFrame(arr, columns=SCEUA_PARAMETER_NAMES)


def _select_behavioral_for_prediction(
    behavioral_sets: pd.DataFrame | np.ndarray,
    *,
    max_sets: int | None,
    seed: int,
) -> pd.DataFrame:
    frame = _behavioral_frame(behavioral_sets)
    if len(frame) == 0:
        raise ValueError("behavioral_sets cannot be empty.")
    if max_sets is not None and len(frame) > max_sets:
        rng = np.random.default_rng(seed)
        frame = frame.iloc[rng.choice(len(frame), size=int(max_sets), replace=False)].copy()
    return frame.reset_index(drop=True)


def predict_lines_with_behavioral_sets(
    behavioral_sets: pd.DataFrame | np.ndarray,
    line_ids: list[str],
    base_inputs: dict[str, Any],
    config: dict[str, Any],
    *,
    max_sets: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Predict travel times for lines using calibrated behavioral sets."""
    frame = _select_behavioral_for_prediction(behavioral_sets, max_sets=max_sets, seed=seed)
    output: dict[str, Any] = {"behavioral_sets": frame, "lines": {}}
    for line_id in line_ids:
        predictions = []
        records = []
        observed = None
        offset = None
        sigma = None
        for sample_id, row in frame.iterrows():
            model = build_model_from_candidate(row, base_inputs, config)
            result = forward_predict_line(
                model,
                base_inputs["lines"][line_id],
                base_inputs,
                config,
            )
            predictions.append(result["predicted"])
            observed = result["observed"]
            offset = result["offset"]
            sigma = result["sigma_eff"]
            records.append(
                {
                    "behavioral_id": int(sample_id),
                    "line": line_id,
                    "rmse_s": result["rmse_s"],
                    "normalized_rmse": result["normalized_rmse"],
                    **{name: float(row[name]) for name in SCEUA_PARAMETER_NAMES},
                }
            )
        pred_stack = np.stack(predictions)
        output["lines"][line_id] = {
            "observed": observed,
            "offset": offset,
            "sigma_eff": sigma,
            "predicted": pred_stack,
            "p05": np.nanpercentile(pred_stack, 5, axis=0),
            "p50": np.nanpercentile(pred_stack, 50, axis=0),
            "p95": np.nanpercentile(pred_stack, 95, axis=0),
            "summary": pd.DataFrame(records),
        }
    return output


def summarize_prediction_ensemble(prediction: dict[str, Any]) -> pd.DataFrame:
    """Summarize behavioral travel-time prediction envelopes by line."""
    rows = []
    for line_id, line_result in prediction["lines"].items():
        observed = np.asarray(line_result["observed"], dtype=float)
        p50 = np.asarray(line_result["p50"], dtype=float)
        p05 = np.asarray(line_result["p05"], dtype=float)
        p95 = np.asarray(line_result["p95"], dtype=float)
        n = min(observed.size, p50.size)
        residual = p50[:n] - observed[:n]
        sigma = np.asarray(line_result["sigma_eff"], dtype=float)[:n]
        rows.append(
            {
                "line": line_id,
                "rmse_s": float(np.sqrt(np.nanmean(residual**2))),
                "normalized_rmse": float(np.sqrt(np.nanmean((residual / sigma) ** 2))),
                "coverage_90": float(np.mean((observed[:n] >= p05[:n]) & (observed[:n] <= p95[:n]))),
                "n_picks": int(n),
            }
        )
    return pd.DataFrame(rows)


def propagate_behavioral_to_3d(
    behavioral_sets: pd.DataFrame | np.ndarray,
    base_inputs: dict[str, Any],
    config: dict[str, Any],
    *,
    n_sets: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Propagate calibrated behavioral sets to full 3-D CZ and Vp uncertainty."""
    sceua = base_inputs.get("sceua_config", load_sceua_config(config))
    if n_sets is None:
        n_sets = int(sceua.get("propagation", {}).get("n_behavioral_sets", 200))
    frame = _select_behavioral_for_prediction(behavioral_sets, max_sets=int(n_sets), seed=seed)
    dem = base_inputs["dem"]
    depth = np.asarray(base_inputs["depth"], dtype=float)
    selected_depths = np.asarray(
        sceua.get("propagation", {}).get("selected_vp_depths_m", [2.0, 10.0, 25.0, 40.0, 80.0]),
        dtype=float,
    )
    depth_indices = np.asarray([int(np.argmin(np.abs(depth - value))) for value in selected_depths], dtype=int)
    row_index = len(dem.y) // 2

    h_soil_stack = []
    h_weathered_stack = []
    d_fresh_stack = []
    vp_depth_stack = []
    vp_section_stack = []
    vp_profile_stack = []
    phi_depth_stack = []
    phi_section_stack = []
    phi_profile_stack = []
    unit_section_stack = []
    diagnostics = []
    for sample_id, row in frame.iterrows():
        model = build_model_from_candidate(row, base_inputs, config)
        h_soil_stack.append(model["H_soil"])
        h_weathered_stack.append(model["H_weathered"])
        d_fresh_stack.append(model["D_fresh"])
        vp = np.asarray(model["vp"], dtype=float)
        phi = np.asarray(model["phi"], dtype=float)
        vp_depth_stack.append(vp[depth_indices])
        vp_section_stack.append(vp[:, row_index, :])
        vp_profile_stack.append(np.nanmedian(vp, axis=(1, 2)))
        phi_depth_stack.append(phi[depth_indices])
        phi_section_stack.append(phi[:, row_index, :])
        phi_profile_stack.append(np.nanmedian(phi, axis=(1, 2)))
        unit_section_stack.append(np.asarray(model["unit"], dtype=float)[:, row_index, :])
        diagnostics.append({"behavioral_id": int(sample_id), **_model_diagnostics(model)})

    h_soil_arr = np.stack(h_soil_stack)
    h_weathered_arr = np.stack(h_weathered_stack)
    d_fresh_arr = np.stack(d_fresh_stack)
    vp_depth_arr = np.stack(vp_depth_stack)
    vp_section_arr = np.stack(vp_section_stack)
    vp_profile_arr = np.stack(vp_profile_stack)
    phi_depth_arr = np.stack(phi_depth_stack)
    phi_section_arr = np.stack(phi_section_stack)
    phi_profile_arr = np.stack(phi_profile_stack)
    unit_section_arr = np.stack(unit_section_stack)
    return {
        "behavioral_sets": frame,
        "diagnostics": pd.DataFrame(diagnostics),
        "x": dem.x,
        "y": dem.y,
        "depth": depth,
        "selected_vp_depths_m": depth[depth_indices],
        "H_soil_mean": np.nanmean(h_soil_arr, axis=0),
        "H_soil_std": np.nanstd(h_soil_arr, axis=0),
        "H_weathered_mean": np.nanmean(h_weathered_arr, axis=0),
        "H_weathered_std": np.nanstd(h_weathered_arr, axis=0),
        "D_fresh_mean": np.nanmean(d_fresh_arr, axis=0),
        "D_fresh_std": np.nanstd(d_fresh_arr, axis=0),
        "D_fresh_p05": np.nanpercentile(d_fresh_arr, 5, axis=0),
        "D_fresh_p50": np.nanpercentile(d_fresh_arr, 50, axis=0),
        "D_fresh_p95": np.nanpercentile(d_fresh_arr, 95, axis=0),
        "Vp_depth_mean": np.nanmean(vp_depth_arr, axis=0),
        "Vp_depth_std": np.nanstd(vp_depth_arr, axis=0),
        "Vp_depth_p05": np.nanpercentile(vp_depth_arr, 5, axis=0),
        "Vp_depth_p50": np.nanpercentile(vp_depth_arr, 50, axis=0),
        "Vp_depth_p95": np.nanpercentile(vp_depth_arr, 95, axis=0),
        "Vp_section_p50": np.nanpercentile(vp_section_arr, 50, axis=0),
        "Vp_section_std": np.nanstd(vp_section_arr, axis=0),
        "Vp_profile_p05": np.nanpercentile(vp_profile_arr, 5, axis=0),
        "Vp_profile_p50": np.nanpercentile(vp_profile_arr, 50, axis=0),
        "Vp_profile_p95": np.nanpercentile(vp_profile_arr, 95, axis=0),
        "Phi_depth_mean": np.nanmean(phi_depth_arr, axis=0),
        "Phi_depth_std": np.nanstd(phi_depth_arr, axis=0),
        "Phi_depth_p05": np.nanpercentile(phi_depth_arr, 5, axis=0),
        "Phi_depth_p50": np.nanpercentile(phi_depth_arr, 50, axis=0),
        "Phi_depth_p95": np.nanpercentile(phi_depth_arr, 95, axis=0),
        "Phi_section_p50": np.nanpercentile(phi_section_arr, 50, axis=0),
        "Phi_section_std": np.nanstd(phi_section_arr, axis=0),
        "Phi_profile_p05": np.nanpercentile(phi_profile_arr, 5, axis=0),
        "Phi_profile_p50": np.nanpercentile(phi_profile_arr, 50, axis=0),
        "Phi_profile_p95": np.nanpercentile(phi_profile_arr, 95, axis=0),
        "unit_section_p50": np.nanpercentile(unit_section_arr, 50, axis=0),
        "section_row_index": row_index,
    }


def save_prediction_outputs(
    prediction: dict[str, Any],
    output_dir: str | Path,
    *,
    prefix: str,
) -> pd.DataFrame:
    """Save behavioral travel-time prediction envelopes and summaries."""
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_prediction_ensemble(prediction)
    summary.to_csv(_write_path(data_dir / f"{prefix}_prediction_summary.csv"), index=False)
    for line_id, line_result in prediction["lines"].items():
        arrays = {
            key: value
            for key, value in line_result.items()
            if isinstance(value, np.ndarray)
        }
        np.savez_compressed(_write_path(data_dir / f"{prefix}_{line_id}_prediction.npz"), **arrays)
        if isinstance(line_result.get("summary"), pd.DataFrame):
            line_result["summary"].to_csv(
                _write_path(data_dir / f"{prefix}_{line_id}_sample_summary.csv"),
                index=False,
            )
    return summary


def save_3d_uncertainty_outputs(summary: dict[str, Any], output_dir: str | Path) -> None:
    """Save 3-D CZ and Vp behavioral uncertainty arrays."""
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in summary.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(
        _write_path(data_dir / "behavioral_3d_uncertainty.npz"),
        **arrays,
    )
    if isinstance(summary.get("diagnostics"), pd.DataFrame):
        summary["diagnostics"].to_csv(
            _write_path(data_dir / "behavioral_3d_diagnostics.csv"),
            index=False,
        )


def save_sceua_outputs(results: dict[str, Any], output_dir: str | Path) -> None:
    """Save standard SCE-UA calibration CSV products."""
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = results["records"]
    behavioral = results["behavioral_sets"]
    best = results["best_parameters"]
    summary = results.get("behavioral_summary")
    for directory in (output_dir, data_dir):
        records.to_csv(_write_path(directory / "sceua_results.csv"), index=False)
        behavioral.to_csv(
            _write_path(directory / "behavioral_parameter_sets.csv"),
            index=False,
        )
        best.to_csv(_write_path(directory / "best_fit_parameters.csv"), index=False)
        if isinstance(summary, pd.DataFrame):
            summary.to_csv(
                _write_path(directory / "behavioral_parameter_summary.csv"),
                index=False,
            )
