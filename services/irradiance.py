"""IrradianceService: NASA POWER long-term solar climatology per census tract.

    uvicorn services.irradiance:app --port 8001

Irradiance values are kWh/m2/day, clearness_index is unitless, temp_c is degrees C.
Data comes from features.tract_irradiance (pipeline step "tract_irradiance").
"""

from fastapi import FastAPI, Path, Query

from services.common import COUNTY, GEOID, ping, tract_row, tract_rows

app = FastAPI(title="IrradianceService")


@app.get("/health")
def health() -> dict:
    return ping()


@app.get("/irradiance")
def list_irradiance(county: str | None = Query(None, pattern=COUNTY, description="e.g. 24005")) -> list[dict]:
    return tract_rows("tract_irradiance", county)


@app.get("/irradiance/{geoid}")
def get_irradiance(geoid: str = Path(pattern=GEOID, description="e.g. 24005400100")) -> dict:
    return tract_row("tract_irradiance", geoid)
