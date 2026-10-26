"""EquityGapService: how much rooftop solar a tract should have versus what it has.

    uvicorn services.equity:app --port 8005

Serves scores.tract_equity, written by `python scripts/train_equity.py`.

gap = expected_adoption - actual_adoption, in residential systems per 1,000 households.
A positive gap means the tract has the sunlight and housing stock to support more solar
than it has. Both components are returned alongside the gap on purpose: a consumer that
sees only the difference cannot tell strong untapped potential from a poor expectation.
"""

import json

from fastapi import FastAPI, HTTPException, Path, Query

from pipeline.config import ROOT
from services.common import COUNTY, GEOID, ping, query, tract_row, tract_rows

app = FastAPI(title="EquityGapService")

METRICS_PATH = ROOT / "models" / "equity_metrics.json"
TABLE = "tract_equity"


@app.get("/health")
def health() -> dict:
    return ping()


@app.get("/equity/model/metrics")
def get_metrics() -> dict:
    """Spatial-CV scores, the ridge baseline comparison, and mean |SHAP| per barrier."""
    if not METRICS_PATH.exists():
        raise HTTPException(503, "model not trained yet - run scripts/train_equity.py")
    return json.loads(METRICS_PATH.read_text())


@app.get("/equity/summary")
def get_summary(county: str | None = Query(None, pattern=COUNTY)) -> dict:
    """Tier counts and the worst tracts - what the dashboard header needs."""
    where = " WHERE geoid LIKE :prefix" if county else ""
    params = {"prefix": county + "%"} if county else {}
    tiers = query(
        f'SELECT priority_tier, COUNT(*) AS n FROM scores."{TABLE}"{where} GROUP BY priority_tier', params
    )
    worst = query(
        f'SELECT geoid, gap, expected_adoption, actual_adoption, primary_barrier '
        f'FROM scores."{TABLE}"{where} ORDER BY gap DESC NULLS LAST LIMIT 10',
        params,
    )
    return {
        "county": county,
        "tier_counts": {r["priority_tier"]: r["n"] for r in tiers},
        "highest_gap": worst,
    }


@app.get("/equity")
def list_equity(county: str | None = Query(None, pattern=COUNTY, description="e.g. 24510")) -> list[dict]:
    return tract_rows(TABLE, county, schema="scores")


@app.get("/equity/{geoid}")
def get_equity(geoid: str = Path(pattern=GEOID, description="e.g. 24510070100")) -> dict:
    return tract_row(TABLE, geoid, schema="scores")
