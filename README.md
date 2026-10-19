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

## Layout

```
data/raw/          source downloads (gitignored)
data/processed/    cleaned outputs (gitignored)
notebooks/         exploration
pipeline/          data ingestion scripts; db.py returns a PostGIS engine
services/          FastAPI microservices
models/            trained model artifacts (gitignored)
db/init/           SQL run on first database start (PostGIS + schemas raw, features, scores)
scripts/           utilities, e.g. check_setup.py
```

## Roadmap

- [x] 1. Project scaffold
- [ ] 2. Census tracts + ACS
- [ ] 3. LEAD + DeepSolar
- [ ] 4. NASA POWER irradiance
- [ ] 5. Terrain features (DEM, WorldCover, OSM grid distance)
- [ ] 6. USPVDB labels + terrain model
- [ ] 7. Rooftop model + equity gap index
- [ ] 8. FastAPI services + map
