"""Flow routing and channel-relative relief from ArcGIS D8 products.

The workflow reads flow direction, flow accumulation, and the channel raster
produced in ArcGIS, reprojects them onto the model DEM, and returns the
height above the hydrologically connected downstream channel that the
relief-ratio model uses as its datum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dem_tools import DemGrid, compute_hand


CHANNEL_AREA_THRESHOLD_M2 = 1500.0
@dataclass(frozen=True)
class FlowResult:
    """Flow-routing fields in the source DEM row order."""

    drainage_area_m2: np.ndarray
    flow_receivers: np.ndarray
    channel_mask: np.ndarray
    height_above_channel_m: np.ndarray
    threshold_m2: float
    channel_source: str = "drainage_area_threshold_or_terminal"
    include_terminal_nodes_as_channels: bool = True


def _target_transform_for_dem(dem: DemGrid):
    from rasterio.transform import from_origin

    x = np.asarray(dem.x, dtype=float)
    y = np.asarray(dem.y, dtype=float)
    left = float(np.nanmin(x) - 0.5 * float(dem.dx))
    top = float(np.nanmax(y) + 0.5 * float(dem.dy))
    return from_origin(left, top, float(dem.dx), float(dem.dy))


def _reproject_to_dem(
    source: np.ndarray,
    *,
    src_transform,
    src_crs,
    dem: DemGrid,
    resampling,
    src_nodata: float | int | None = None,
    dst_nodata: float | int = 0,
    dtype: str = "float64",
) -> np.ndarray:
    from rasterio.warp import reproject

    y_is_ascending = bool(len(dem.y) > 1 and dem.y[0] < dem.y[-1])
    dst = np.full(dem.data.shape, dst_nodata, dtype=dtype)
    reproject(
        source=source,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=_target_transform_for_dem(dem),
        dst_crs=dem.crs or src_crs,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst[::-1].copy() if y_is_ascending else dst


def _arcgis_d8_receivers(flow_direction: np.ndarray) -> np.ndarray:
    """Convert ArcGIS D8 flow-direction codes to flat downstream receivers."""
    directions = np.asarray(flow_direction, dtype=np.int16)
    nrows, ncols = directions.shape
    receivers = np.full((nrows, ncols), -1, dtype=int)
    offsets = {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
    }
    for code, (dr, dc) in offsets.items():
        rows, cols = np.where(directions == code)
        rr = rows + dr
        cc = cols + dc
        inside = (rr >= 0) & (rr < nrows) & (cc >= 0) & (cc < ncols)
        receivers[rows[inside], cols[inside]] = rr[inside] * ncols + cc[inside]
    return receivers


def compute_arcgis_flow(
    dem: DemGrid,
    *,
    flow_dem_path: str | Path | None = None,
    flow_direction_path: str | Path,
    flow_accumulation_path: str | Path,
    channel_raster_path: str | Path,
    channel_threshold_m2: float = CHANNEL_AREA_THRESHOLD_M2,
    include_outflow_nodes_as_channels: bool = True,
    smooth_hand_sigma_cells: float = 0.0,
) -> FlowResult:
    """Build flow-routing fields from ArcGIS D8 rasters.

    The HAND calculation is performed on the native ArcGIS raster grid and then
    aggregated to the supplied DEM grid. No Landlab flow routing is used.
    """
    import rasterio
    from rasterio.warp import Resampling

    flow_direction_path = Path(flow_direction_path)
    flow_accumulation_path = Path(flow_accumulation_path)
    channel_raster_path = Path(channel_raster_path)
    for path in (flow_direction_path, flow_accumulation_path, channel_raster_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if flow_dem_path is None:
        if dem.path is None or not Path(dem.path).exists():
            raise ValueError(
                "ArcGIS flow requires either flow_dem_path or dem.path to point "
                "to the source filled DEM raster."
            )
        flow_dem_path = dem.path
    flow_dem_path = Path(flow_dem_path)
    if not flow_dem_path.exists():
        raise FileNotFoundError(flow_dem_path)

    with rasterio.open(flow_dem_path) as dem_src:
        z_native = dem_src.read(1).astype(float)
        dem_src_transform = dem_src.transform
        dem_src_crs = dem_src.crs
        dem_src_nodata = dem_src.nodata
        native_shape = (dem_src.height, dem_src.width)
    if dem_src_nodata is not None:
        z_native = np.where(np.isclose(z_native, dem_src_nodata), np.nan, z_native)
    z_native = np.where(np.isfinite(z_native) & (z_native < 1.0e20), z_native, np.nan)

    with rasterio.open(flow_direction_path) as direction_src:
        flow_direction = direction_src.read(1)
        direction_transform = direction_src.transform
        direction_crs = direction_src.crs
        direction_shape = (direction_src.height, direction_src.width)
    with rasterio.open(flow_accumulation_path) as accumulation_src:
        flow_accumulation = accumulation_src.read(1).astype(float)
        accumulation_transform = accumulation_src.transform
        accumulation_crs = accumulation_src.crs
        accumulation_shape = (accumulation_src.height, accumulation_src.width)
        accumulation_nodata = accumulation_src.nodata
    with rasterio.open(channel_raster_path) as channel_src:
        channel_values = channel_src.read(1)
        channel_transform = channel_src.transform
        channel_crs = channel_src.crs
        channel_shape = (channel_src.height, channel_src.width)
        channel_nodata = channel_src.nodata

    shapes = {native_shape, direction_shape, accumulation_shape, channel_shape}
    if len(shapes) != 1:
        raise ValueError(
            "ArcGIS DEM, flow direction, flow accumulation, and channel rasters must share a grid."
        )
    transforms = {
        tuple(np.round(tuple(dem_src_transform), 9)),
        tuple(np.round(tuple(direction_transform), 9)),
        tuple(np.round(tuple(accumulation_transform), 9)),
        tuple(np.round(tuple(channel_transform), 9)),
    }
    if len(transforms) != 1:
        raise ValueError(
            "ArcGIS DEM, flow direction, flow accumulation, and channel rasters must share a transform."
        )
    crs_values = {str(crs) for crs in (dem_src_crs, direction_crs, accumulation_crs, channel_crs) if crs is not None}
    if len(crs_values) > 1:
        raise ValueError(
            "ArcGIS DEM, flow direction, flow accumulation, and channel rasters must share a CRS."
        )
    if dem.crs is not None and dem_src_crs is not None and dem.crs != dem_src_crs:
        raise ValueError(f"Target DEM CRS {dem.crs} does not match ArcGIS raster CRS {dem_src_crs}.")

    if accumulation_nodata is not None:
        flow_accumulation = np.where(
            np.isclose(flow_accumulation, accumulation_nodata),
            np.nan,
            flow_accumulation,
        )
    cell_area_m2 = abs(float(dem_src_transform.a) * float(dem_src_transform.e))
    drainage_area_native = flow_accumulation * cell_area_m2

    if channel_nodata is not None:
        channel_values = np.where(np.isclose(channel_values, channel_nodata), 0, channel_values)
    channel_native = np.asarray(channel_values) > 0

    receivers_native = _arcgis_d8_receivers(flow_direction)
    outflow_native = receivers_native < 0
    active_channel_native = channel_native.copy()
    if include_outflow_nodes_as_channels:
        active_channel_native |= outflow_native
    flat_native = np.arange(receivers_native.size, dtype=int).reshape(receivers_native.shape)
    receivers_native[active_channel_native] = flat_native[active_channel_native]

    try:
        hand_native = compute_hand(
            z_native,
            drainage_area_native,
            float(channel_threshold_m2),
            receivers_native,
            channel_mask=active_channel_native,
            include_terminal_nodes=False,
            normalized=False,
        )
    except ValueError as exc:
        unresolved = int(np.sum(outflow_native & ~active_channel_native))
        raise ValueError(
            "ArcGIS HAND tracing failed before all cells reached the ArcGIS channel raster. "
            f"Outflow cells not treated as channel datum: {unresolved}. "
            "Use a larger ArcGIS raster/channel extent or set include_outflow_nodes_as_channels=True."
        ) from exc

    drainage_target = _reproject_to_dem(
        drainage_area_native,
        src_transform=dem_src_transform,
        src_crs=dem_src_crs,
        dem=dem,
        resampling=Resampling.max,
        src_nodata=np.nan,
        dst_nodata=0.0,
        dtype="float64",
    )
    hand_target = _reproject_to_dem(
        hand_native,
        src_transform=dem_src_transform,
        src_crs=dem_src_crs,
        dem=dem,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=0.0,
        dtype="float64",
    )
    channel_target = _reproject_to_dem(
        active_channel_native.astype("uint8"),
        src_transform=dem_src_transform,
        src_crs=dem_src_crs,
        dem=dem,
        resampling=Resampling.max,
        src_nodata=0,
        dst_nodata=0,
        dtype="uint8",
    ).astype(bool)
    if smooth_hand_sigma_cells > 0:
        from scipy.ndimage import gaussian_filter

        sigma = float(smooth_hand_sigma_cells)
        valid = np.isfinite(hand_target)
        numerator = gaussian_filter(
            np.where(valid, hand_target, 0.0),
            sigma=sigma,
            mode="nearest",
        )
        denominator = gaussian_filter(
            valid.astype(float),
            sigma=sigma,
            mode="nearest",
        )
        smoothed = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(hand_target, dtype=float),
            where=denominator > 0,
        )
        hand_target = np.where(valid, smoothed, hand_target)
    hand_target[channel_target] = 0.0
    direction_target = _reproject_to_dem(
        flow_direction.astype("int16"),
        src_transform=dem_src_transform,
        src_crs=dem_src_crs,
        dem=dem,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
        dtype="int16",
    )
    receivers_target = _arcgis_d8_receivers(direction_target)
    flat_target = np.arange(receivers_target.size, dtype=int).reshape(receivers_target.shape)
    receivers_target[channel_target | (receivers_target < 0)] = flat_target[
        channel_target | (receivers_target < 0)
    ]

    channel_source = (
        f"arcgis_dem={flow_dem_path.name}; "
        f"arcgis_flow_direction={flow_direction_path.name}; "
        f"arcgis_flow_accumulation={flow_accumulation_path.name}; "
        f"arcgis_channel={channel_raster_path.name}"
    )
    if include_outflow_nodes_as_channels:
        channel_source += "; arcgis_outflow_nodes_included"
    if smooth_hand_sigma_cells > 0:
        channel_source += f"; hand_smoothing_sigma_cells={float(smooth_hand_sigma_cells):g}"

    return FlowResult(
        drainage_area_m2=np.asarray(drainage_target, dtype=float),
        flow_receivers=np.asarray(receivers_target, dtype=int),
        channel_mask=np.asarray(channel_target, dtype=bool),
        height_above_channel_m=np.asarray(hand_target, dtype=float),
        threshold_m2=float(channel_threshold_m2),
        channel_source=channel_source,
        include_terminal_nodes_as_channels=False,
    )


