"""SocioEconService: ACS demographics and NREL LEAD energy burden per census tract.

    uvicorn services.socioecon:app --port 8003

Joins raw.acs (ACS 2018-2022 5-year) with raw.lead (NREL LEAD 2022). Both sit on 2020
tract boundaries, so the join on geoid is exact. Incomes and costs are dollars per year,
energy_burden is a fraction of household income, rates and shares are fractions.
"""

from fastapi import FastAPI, HTTPException, Path, Query

from services.common import COUNTY, GEOID, ping, query

app = FastAPI(title="SocioEconService")

# LEAD and ACS both carry a geoid; drop the duplicate from the right-hand side.
BASE_SQL = """
SELECT a.*, l.lead_households, l.avg_household_income, l.avg_energy_cost,
       l.avg_electricity_cost, l.energy_burden, l.low_income_share,
       l.low_income_energy_burden, l.renter_share
FROM raw.acs AS a
LEFT JOIN raw.lead AS l USING (geoid)
"""


@app.get("/health")
def health() -> dict:
    return ping()


@app.get("/socioecon")
def list_socioecon(county: str | None = Query(None, pattern=COUNTY, description="e.g. 24005")) -> list[dict]:
    if county:
        return query(BASE_SQL + " WHERE a.geoid LIKE :prefix ORDER BY a.geoid", {"prefix": county + "%"})
    return query(BASE_SQL + " ORDER BY a.geoid")


@app.get("/socioecon/{geoid}")
def get_socioecon(geoid: str = Path(pattern=GEOID, description="e.g. 24005400100")) -> dict:
    rows = query(BASE_SQL + " WHERE a.geoid = :g", {"g": geoid})
    if not rows:
        raise HTTPException(404, f"tract {geoid} not found")
    return rows[0]
