"""YieldService: annual rooftop PV output per census tract.

    uvicorn services.pv_yield:app --port 8010

Physics, not ML. specific_yield_kwh_kw is kWh per year from 1 kW(dc) on a south-facing,
20 deg roof, computed with pvlib from NASA POWER climatology (pipeline step "tract_yield").
Pass ?kw= to scale to a system size; output is linear in capacity.
"""

from fastapi import FastAPI, Path, Query

from pipeline import pv_yield
from services.common import COUNTY, GEOID, ping, tract_row, tract_rows

app = FastAPI(title="YieldService")

TABLE = "tract_yield"
KW = Query(1.0, gt=0, le=1000, description="system size in kW(dc)")


def with_size(row: dict, kw: float) -> dict:
    return {**row, "system_kw": kw, "annual_kwh": round(row["specific_yield_kwh_kw"] * kw, 1)}


@app.get("/health")
def health() -> dict:
    return ping()


@app.get("/yield/meta/assumptions")
def get_assumptions() -> dict:
    """The PVWatts-equivalent inputs, so a result can be reproduced at pvwatts.nrel.gov."""
    return {
        "tilt_deg": pv_yield.TILT, "azimuth_deg": pv_yield.AZIMUTH, "albedo": pv_yield.ALBEDO,
        "system_losses": pv_yield.LOSSES, "inverter_efficiency": pv_yield.INVERTER_EFF,
        "temp_coefficient_per_c": pv_yield.GAMMA_PDC, "source": "NASA POWER climatology + pvlib",
    }


@app.get("/yield")
def list_yield(county: str | None = Query(None, pattern=COUNTY, description="e.g. 24510"),
               kw: float = KW) -> list[dict]:
    return [with_size(r, kw) for r in tract_rows(TABLE, county)]


@app.get("/yield/{geoid}")
def get_yield(geoid: str = Path(pattern=GEOID, description="e.g. 24510070100"), kw: float = KW) -> dict:
    return with_size(tract_row(TABLE, geoid), kw)
