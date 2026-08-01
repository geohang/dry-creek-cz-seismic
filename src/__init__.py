"""Modules for process-informed critical-zone seismic modeling at Dry Creek.

Dependency order, shallowest first:

``data_io``
    YAML config reading and DEM / seismic file discovery.
``dem_tools``
    DEM loading, resampling, and sampling along a seismic line.
``mobile_regolith``
    Landlab mobile-regolith production and depth-dependent transport.
``flow_routing``
    Channel mask and channel-relative relief from the ArcGIS D8 products.
``velocity_model``
    Extraction of velocity sections from the three-dimensional model.
``rd_rpv_helpers``
    Rock physics and the three-unit critical-zone grid.
``seismic_forward``
    pyGIMLi mesh construction and first-arrival travel-time simulation.
``seismic_lines``
    Line discovery, calibration/validation split, pyGIMLi input preparation.
``sceua_landlab_rd_rpv``
    The SCE-UA calibration driver that ties the rest together.

Every function here is on the path that produces a result reported in the paper.
"""

__all__ = [
    "data_io",
    "dem_tools",
    "flow_routing",
    "mobile_regolith",
    "rd_rpv_helpers",
    "sceua_landlab_rd_rpv",
    "seismic_forward",
    "seismic_lines",
    "velocity_model",
]
