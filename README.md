# Solar Equity Suitability as a Service

CMSC 668 (Service-Oriented Computing), UMBC. A set of FastAPI microservices that score
solar suitability for Maryland: rooftop suitability per census tract and utility-scale
terrain suitability per grid cell, plus an equity gap index that compares solar potential
against current adoption and household energy burden. Everything is built from open data:
Census ACS, NREL LEAD, DeepSolar, NASA POWER, Copernicus DEM, ESA WorldCover,
OpenStreetMap, and the U.S. Large-Scale Solar Photovoltaic Database (USPVDB).

## Team

- **Krish Mehta**: data pipeline and preprocessing
- **Rahul Badwaik**: ML models and SOA services

## Setup

1. Install [Miniforge](https://github.com/conda-forge/miniforge) and
   [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Create the conda environment:
   ```
   mamba env create -f environment.yml
   conda activate solar
   ```
3. Copy the env template and fill in your keys:
   ```
   cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
   ```
   - Census API key: https://api.census.gov/data/key_signup.html
   - NREL API key: https://developer.nrel.gov/signup/
4. Start PostGIS:
   ```
   docker compose up -d
   ```
5. Verify everything:
   ```
   python scripts/check_setup.py
   ```

## Data pipeline

```
python scripts/run_pipeline.py            # every step, in order
python scripts/run_pipeline.py osm grid   # selected steps
python scripts/run_pipeline.py --list
```

Downloads are cached in `data/raw/` and intermediate rasters in `data/processed/`, so
re-runs are fast. Each step writes to PostGIS (CRS EPSG:26985, meters) and later steps read
from there. Only `acs` needs an API key (`CENSUS_API_KEY`).

| Step | Source | Output table |
|---|---|---|
| boundaries | Census cartographic boundaries 2022 (shoreline-clipped) | `raw.state_boundary`, `raw.tracts` |
| acs | ACS 2018-2022 5-year (housing, income, poverty, race) | `raw.acs` |
| lead | NREL LEAD 2022: energy burden, low-income share | `raw.lead` |
| deepsolar | DeepSolar installs, moved from 2010 to 2020 tracts by land-area share | `raw.deepsolar` |
| irradiance | NASA POWER annual/monthly GHI climatology (1 degree grid) | `raw.power_ghi` |
| grid | 1 km cells over Maryland land | `features.grid` |
| terrain | Copernicus DEM GLO-90 (elevation, slope, aspect), ESA WorldCover 2021 | `features.grid_terrain` |
| osm | OSM transmission lines and substations (Overpass) | `raw.osm_power_lines`, `raw.osm_substations` |
| uspvdb | USPVDB large-scale PV facility polygons | `raw.uspvdb` |
| tract_features | joins tracts + ACS + LEAD + DeepSolar + GHI | `features.tract_features` |
| grid_features | joins grid + terrain + GHI + grid distances + PV labels | `features.grid_features` |

The model-ready tables are `features.tract_features` (rooftop model and equity index) and
`features.grid_features` (terrain model). The grid label is `has_pv` (USPVDB covers at least
1% of the cell); `pv_frac` holds the raw coverage. WorldCover 2021 can show existing arrays as
built or bare land, so watch for label leakage through the `lc_*` features.

## Layout

```
data/raw/          source downloads (gitignored)
data/processed/    cleaned outputs (gitignored)
notebooks/         exploration
pipeline/          data ingestion steps (one module per source); db.py returns a PostGIS engine
services/          FastAPI microservices
models/            trained model artifacts (gitignored)
db/init/           SQL run on first database start (PostGIS + schemas raw, features, scores)
scripts/           utilities, e.g. check_setup.py
```

## Roadmap

- [x] 1. Project scaffold
- [ ] 2. Census tracts + ACS (tracts done; ACS step written, needs a Census API key to run)
- [x] 3. LEAD + DeepSolar
- [x] 4. NASA POWER irradiance
- [x] 5. Terrain features (DEM, WorldCover, OSM grid distance)
- [ ] 6. USPVDB labels + terrain model (labels done)
- [ ] 7. Rooftop model + equity gap index
- [ ] 8. FastAPI services + map
