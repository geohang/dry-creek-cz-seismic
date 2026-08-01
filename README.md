# Dry Creek Critical-Zone Seismic Modeling

Code and data for calibrating a process-based model of critical-zone structure
directly against seismic-refraction travel times, at the Treeline site in the
Dry Creek Experimental Watershed, Idaho.

Mobile-regolith thickness evolves through production and hillslope transport in
Landlab, fresh-bedrock depth is set by relief above the channel network, the
resulting structure is converted to P-wave velocity through rock physics, and
first arrivals are simulated with pyGIMLi and compared with the measured picks.
Parameters are calibrated with SCE-UA.

> Chen, H. A Geophysics-Informed Framework for Modeling Critical-Zone Subsurface
> Structure.
```
src/          modeling and calibration code, nine modules
notebooks/    calibration, then travel-time and velocity results
data/         seismic records, first-arrival picks, DEM and flow rasters
outputs/      calibration results: full search history and behavioral ensemble
```

## Seismic data

Seven P-wave refraction profiles collected at Treeline between May and September
2016, labelled `TL1` to `TL7`. Each was recorded on 72 channels at a nominal 5 m
geophone spacing with roughly 20 m source spacing, using a Geode seismograph and
a sledgehammer source. `TL1` to `TL5` are used for calibration; `TL6` and `TL7`
are held out.

| Line |  Sensors | Length (m) | Shots | Picks | Role |
|---|---:|---:|---:|---:|---|
| `TL1` | 72 | 330 | 19 | 1,349 | calibration |
| `TL2`  | 63 | 278 | 16 | 992 | calibration |
| `TL3`  | 62 | 282 | 16 | 976 | calibration |
| `TL4`  | 62 | 281 | 16 | 976 | calibration |
| `TL5` | 62 | 288 | 16 | 976 | calibration |
| `TL6`  | 62 | 282 | 16 | 976 | validation |
| `TL7`  | 60 | 273 | 16 | 945 | validation |

7,190 first-arrival picks in total. Field dates come from the SEG-Y trace
headers. Sensor counts are the positions retained in the pick files.

**`data/seismic/raw/TL<n>_line.sgy`** are the raw records: one merged SEG-Y line
file per profile, SEG-Y rev 1 with big-endian IBM floats, 1,152 to 1,368 traces
at 1,500 or 3,000 samples per trace and a 0.5 or 1.0 ms sample interval.

**`data/seismic/picks/`** holds the processed records, two files per profile.
`TL<n>.txt` is the picked first-arrival travel-time file in the pyGIMLi unified
format: a sensor block giving along-line distance and elevation, then `s g t`
triples of shot index, geophone index, and travel time in seconds, with indices
one-based into the sensor block. `TL<n>_topo.txt` gives the geophone easting,
northing, and along-line distance in NAD83 / UTM Zone 11N (EPSG:26911). These
are the direct model inputs, so the results can be reproduced without repeating
the picking.

**`data/terrain/`** holds the five GeoTIFF rasters that define the model domain
and the channel network: the depression-filled DEM over the modeling and
flow-routing domains, and the ArcGIS D8 flow direction, flow accumulation, and
channel rasters. All are EPSG:26911 at 1 m, resampled to the 5 m model grid at
run time.

`data/README.md` documents every file individually with its units, grid size,
and array contents.

## Calibration outputs

`outputs/dc_sceua_pc055_lb/` contains the calibration in full rather than as a
summary:

- the complete SCE-UA search history, all 1,690 successful objective evaluations
  with their parameter values and per-line misfits
- the 85-member behavioral ensemble retained from the best five percent, and the
  best-fitting parameter set
- measured and predicted travel times for every pick on all seven profiles, with
  the ensemble prediction envelope
- gridded mobile-regolith thickness, weathered-bedrock thickness, and depth to
  fresh bedrock, with their ensemble spread

## Setup

```bash
conda env create -f environment.yml
```

```bash
conda activate dry-creek-cz
```

The environment pins Landlab 2.10.1, pyGIMLi 1.5.5, and PyHydroGeophysX at a
fixed commit.

## Running

Run the notebooks in order from the repository root:

1. `01_calibration_and_validation.ipynb` builds the model and the behavioral
   ensemble. By default it loads the archived calibration in `outputs/` rather
   than repeating the search, and recovers the reported values in a few minutes.
   Set `RUN_SCEUA = True` to recalibrate; the full search is 2,500 forward
   evaluations and takes many hours.
2. `02_traveltime_and_parameter_figures.ipynb` produces the travel-time fit and
   the calibrated parameter distributions.
3. `03_velocity_section_figures.ipynb` produces the velocity sections along each
   profile.

Loading the archived calibration reproduces the reported per-line RMSE for all
seven profiles, a mean of 0.0051 s on the calibration lines and 0.0052 s on the
held-out lines, and the 85 retained parameter sets. Figures are written to
`_figures_local/` and are not tracked here.


## License

Code is Apache 2.0 (`LICENSE`). Data are CC BY 4.0 (`LICENSE-DATA`).

## Contact

Hang Chen, School of Earth, Environment, and Sustainability, University of Iowa.
