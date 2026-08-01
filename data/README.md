# Data Dictionary

All coordinates are NAD83 / UTM Zone 11N (EPSG:26911). Lengths and depths are in
metres, travel times in seconds, velocities in metres per second, and porosities
are dimensionless fractions.

Line labels `TL1` to `TL7` correspond to manuscript labels TL-1 to TL-7. Field
dates come from the SEG-Y trace headers.

| Line |  Sensors | Length (m) | Shots | Picks | Role |
|---|---:|---:|---:|---:|---|
| `TL1` | 72 | 330 | 19 | 1,349 | calibration |
| `TL2`  | 63 | 278 | 16 | 992 | calibration |
| `TL3`  | 62 | 282 | 16 | 976 | calibration |
| `TL4`  | 62 | 281 | 16 | 976 | calibration |
| `TL5` | 62 | 288 | 16 | 976 | calibration |
| `TL6`  | 62 | 282 | 16 | 976 | validation |
| `TL7`  | 60 | 273 | 16 | 945 | validation |

Geophone counts are the sensor positions retained in the pick files. All
profiles were recorded on 72 channels at a nominal 5 m ground spacing; along-line
spacing is about 4.6 m because that spacing projects onto sloping terrain.

## Directory Summary

| Path | Contents | Files | Size |
|---|---|---:|---:|
| `seismic/picks/` | Modeling inputs: first-arrival picks and line topography | 14 | 0.2 MB |
| `seismic/raw/` | Merged SEG-Y line file per profile | 7 | 57 MB |
| `terrain/` | DEM and ArcGIS D8 flow products | 5 | 16 MB |
| `../outputs/dc_sceua_pc055_lb/` | Archived calibration results | 34 | 13 MB |

## `seismic/picks/`

Two files per profile. These are the inputs the model is calibrated against.

| File | Format |
|---|---|
| `TL<n>.txt` | pyGIMLi unified travel-time format. Sensor count, a `# x z` block of along-line distance and relative elevation, a pick count, then `# s g t` triples of one-based shot sensor index, one-based geophone sensor index, and travel time in seconds. |
| `TL<n>_topo.txt` | Whitespace-delimited, no header: easting, northing, cumulative along-line distance. Row order matches the sensor order in `TL<n>.txt`. |


## `seismic/raw/`

`TL<n>_line.sgy` is the merged SEG-Y line file for each profile: SEG-Y rev 1,
big-endian IBM floats, 1,152 to 1,368 traces, 1,500 or 3,000 samples per trace,
0.5 or 1.0 ms sample interval. Trace headers carry the acquisition year and
day-of-year used to date each profile.

No notebook reads these. They are included so the picks can be checked against
the records they were made from.

## `terrain/`

GeoTIFF, EPSG:26911, 1 m native resolution. The workflow resamples to the 5 m
model grid using `target_dem_resolution` in `config.yaml`.

| File | Type | Grid | Description |
|---|---|---|---|
| `Fill_cz_modeling_2.tif` | float32 | 435 x 447 | Depression-filled DEM, CZ model domain. Elevation in metres. |
| `Fill_cz_modeling_large.tif` | float32 | 1481 x 947 | Depression-filled DEM, flow-routing domain. |
| `Flow_Direction_large.tif` | uint8 | 1481 x 947 | ArcGIS D8 flow direction codes (1, 2, 4, 8, 16, 32, 64, 128). |
| `Flow_Accumulation_Flow_large.tif` | float32 | 1481 x 947 | Upslope contributing cell count. |
| `RasterC_1200_large.tif` | uint8 | 1481 x 947 | Channel mask from a 1,200-cell accumulation threshold. |

## `../outputs/dc_sceua_pc055_lb/`

Archived calibration results. Files at the top level are the raw SCE-UA output;
the `data/` subdirectory holds the derived products the result notebooks read.

### Parameters

The seven calibrated parameters, in the order used throughout:

| Name | Unit | Meaning |
|---|---|---|
| `P0` | m yr⁻¹ | Maximum mobile-regolith production rate |
| `Hs` | m | Production decay depth |
| `D` | m² yr⁻¹ | Hillslope diffusivity coefficient |
| `r` | – | Relief ratio, fresh-bedrock relief divided by channel-relative surface relief |
| `phi_soil_top` | – | Mobile-regolith porosity at the surface |
| `phi_weathered_top` | – | Porosity at the top of weathered bedrock |
| `phi_fresh` | – | Fresh-bedrock porosity |

