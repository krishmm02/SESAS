"""TerrainService: DEM zonal statistics (elevation, slope, aspect) per census tract.

    uvicorn services.terrain:app --port 8002

Elevations in meters, slopes and aspect in degrees (aspect is the compass direction
the ground faces, 180 = south). Data comes from features.tract_terrain
(pipeline step "tract_terrain").
"""

from fastapi import FastAPI, Path, Query

from services.common import COUNTY, GEOID, ping, tract_row, tract_rows

app = FastAPI(title="TerrainService")


@app.get("/health")
def health() -> dict:
    return ping()


@app.get("/terrain")
def list_terrain(county: str | None = Query(None, pattern=COUNTY, description="e.g. 24005")) -> list[dict]:
    return tract_rows("tract_terrain", county)


@app.get("/terrain/{geoid}")
def get_terrain(geoid: str = Path(pattern=GEOID, description="e.g. 24005400100")) -> dict:
    return tract_row("tract_terrain", geoid)
