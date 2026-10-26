"""AdoptionService: installed solar per census tract, from DeepSolar.

    uvicorn services.adoption:app --port 8004

Counts come from DeepSolar (Stanford), built on 2016 aerial imagery and reallocated
from 2010 to 2020 tracts by land-area share (pipeline step "deepsolar"). Treat these
as a 2016 baseline, not current adoption - Maryland's installed base has grown
substantially since. ds_res_systems_per_1000_hh is null where a tract has fewer than
50 households, because the rate is noise at that size.
"""

from fastapi import FastAPI, Path, Query

from services.common import COUNTY, GEOID, ping, tract_row, tract_rows

app = FastAPI(title="AdoptionService")

VINTAGE = {"source": "DeepSolar (Stanford)", "imagery_year": 2016, "tract_vintage": 2020}


@app.get("/health")
def health() -> dict:
    return ping()


@app.get("/adoption")
def list_adoption(county: str | None = Query(None, pattern=COUNTY, description="e.g. 24005")) -> list[dict]:
    return tract_rows("deepsolar", county, schema="raw")


@app.get("/adoption/{geoid}")
def get_adoption(geoid: str = Path(pattern=GEOID, description="e.g. 24005400100")) -> dict:
    return tract_row("deepsolar", geoid, schema="raw")


@app.get("/adoption/meta/vintage")
def get_vintage() -> dict:
    """Stated separately so consumers can surface the age of this data to end users."""
    return VINTAGE
