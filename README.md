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

## Supabase (shared hosted database)

The local Docker database is fine for one person. To give the team one shared database
(e.g. so the services and models read the same tables), use a Supabase project instead.
The pipeline code doesn't change: `pipeline/db.py` uses `DATABASE_URL` when it is set.

1. **Create the project.** At https://supabase.com/dashboard, click *New project*. Pick a
   region close to Maryland (e.g. *East US (North Virginia)*), set a strong database
   password, and save it in a password manager. It can't be viewed again, only reset.
2. **Create the schemas.** Open *SQL Editor*, paste the contents of
   `db/init/01_extensions.sql`, and click *Run*. This enables PostGIS and creates the
   `raw`, `features`, and `scores` schemas. Check it worked:
   ```sql
   SELECT postgis_version();
   SELECT schema_name FROM information_schema.schemata
   WHERE schema_name IN ('raw', 'features', 'scores');
   ```
3. **Copy the connection string.** Click *Connect* (top bar), then under *Session pooler*
   copy the URI. Use the session pooler, not *Direct connection*: the direct host is IPv6
   only and won't work on most home or campus networks. It looks like
   ```
   postgresql://postgres.<project-ref>:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:5432/postgres
   ```
4. **Link it to the project.** In `.env`, add the URI with your password filled in and
   `?sslmode=require` added at the end:
   ```
   DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require
   ```
   If the password contains `@ : / ? # %`, percent-encode those characters (`@` becomes
   `%40`), or reset it to one without them. Delete or comment out the line to go back
   to local Docker.
5. **Verify.** `python scripts/check_setup.py` should show the pooler host in the
   PostGIS line.
6. **Load the data.** Re-run the pipeline. Downloads are cached in `data/raw/`, so this
   mostly just writes tables:
   ```
   python scripts/run_pipeline.py
   ```
   To copy an existing local database instead (Docker running, password in single quotes):
   ```
   docker exec solar-postgis sh -c "pg_dump -U solar -d solar -n raw -n features -n scores --no-owner --no-privileges | psql 'postgresql://postgres.<project-ref>:<password>@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require'"
   ```
7. **Share access.** Invite teammates under *Project Settings → Team*. Each person puts the
   same `DATABASE_URL` in their own `.env`. Never commit it.

Notes: the free tier allows 500 MB, which these tables fit in easily. Free projects
pause after a week with no activity; click *Restore* in the dashboard to wake one. Supabase's
REST API only exposes the `public` schema, so these tables are reachable only over the
Postgres connection.

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
| irradiance | NASA POWER climatology: GHI, DNI, DHI, clear-sky GHI, clearness index, temperature | `raw.power_climatology` |
| tract_irradiance | POWER values interpolated at each tract's representative point | `features.tract_irradiance` |
| grid | 1 km cells over Maryland land | `features.grid` |
| terrain | Copernicus DEM GLO-90 (elevation, slope, aspect), ESA WorldCover 2021 | `features.grid_terrain` |
| tract_terrain | DEM zonal statistics per tract (elevation, slope, flat/south-facing share, aspect) | `features.tract_terrain` |
| osm | OSM transmission lines and substations (Overpass) | `raw.osm_power_lines`, `raw.osm_substations` |
| uspvdb | USPVDB large-scale PV facility polygons | `raw.uspvdb` |
| tract_features | joins tracts + ACS + LEAD + DeepSolar + irradiance + terrain | `features.tract_features` |
| grid_features | joins grid + terrain + GHI + grid distances + PV labels | `features.grid_features` |

The model-ready tables are `features.tract_features` (rooftop model and equity index) and
`features.grid_features` (terrain model). The grid label is `has_pv` (USPVDB covers at least
1% of the cell); `pv_frac` holds the raw coverage. WorldCover 2021 can show existing arrays as
built or bare land, so watch for label leakage through the `lc_*` features.

## Services

Each service is a FastAPI app that reads its table from the database in `.env`. Run from the
repo root with the `solar` env active; interactive docs are at `/docs` on each port.

```
uvicorn services.irradiance:app --port 8001
uvicorn services.terrain:app --port 8002
```

| Service | Endpoints | Table |
|---|---|---|
| IrradianceService | `GET /irradiance`, `GET /irradiance?county=24005`, `GET /irradiance/{geoid}` | `features.tract_irradiance` |
| TerrainService | `GET /terrain`, `GET /terrain?county=24005`, `GET /terrain/{geoid}` | `features.tract_terrain` |

Both also have `GET /health`. An unknown tract returns 404; a malformed GEOID returns 422.

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
- [x] 2. Census tracts + ACS
- [x] 3. LEAD + DeepSolar
- [x] 4. NASA POWER irradiance
- [x] 5. Terrain features (DEM, WorldCover, OSM grid distance)
- [ ] 6. USPVDB labels + terrain model (labels done)
- [ ] 7. Rooftop model + equity gap index
- [ ] 8. FastAPI services + map
