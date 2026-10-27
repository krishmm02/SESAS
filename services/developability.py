"""DevelopabilityService: utility-scale solar siting score per 1 km grid cell.

    uvicorn services.developability:app --port 8009

Serves scores.grid_developability, written by `python scripts/train_developability.py`.
score is a relative ranking (class-weighted, not a calibrated probability); read
score_percentile and tier. Tier EXCLUDED is developed land (>= 40% built-up), not scored.
driver_1..3 name the top positive contributions with their direction, e.g.
"low dist_substation_km". has_pv marks cells with an existing farm.
"""

import json

from fastapi import FastAPI, HTTPException, Path, Query

from pipeline.config import ROOT
from services.common import COUNTY, ping, query

app = FastAPI(title="DevelopabilityService")

METRICS_PATH = ROOT / "models" / "developability_metrics.json"
TABLE = 'scores."grid_developability"'


@app.get("/health")
def health() -> dict:
    return ping()


@app.get("/developability/model/metrics")
def get_metrics() -> dict:
    """Spatial-CV and temporal-holdout scores against the naive baselines."""
    if not METRICS_PATH.exists():
        raise HTTPException(503, "model not trained yet - run scripts/train_developability.py")
    return json.loads(METRICS_PATH.read_text())


@app.get("/developability")
def list_developability(
    county: str | None = Query(None, pattern=COUNTY, description="e.g. 24047"),
    tier: str | None = Query(None, pattern="^(HIGH|MEDIUM|LOW|EXCLUDED)$"),
    exclude_existing: bool = Query(False, description="drop cells that already have a farm"),
    limit: int = Query(100, ge=1, le=30000),
) -> list[dict]:
    """Cells ranked best first."""
    where, params = [], {"limit": limit}
    if county:
        where.append("county = :county")
        params["county"] = county
    if tier:
        where.append("tier = :tier")
        params["tier"] = tier
    if exclude_existing:
        where.append("has_pv = 0")
    sql = f"SELECT * FROM {TABLE}" + (" WHERE " + " AND ".join(where) if where else "")
    return query(sql + " ORDER BY score DESC NULLS LAST LIMIT :limit", params)


@app.get("/developability/{cell_id}")
def get_developability(cell_id: int = Path(ge=0)) -> dict:
    rows = query(f"SELECT * FROM {TABLE} WHERE cell_id = :c", {"c": cell_id})
    if not rows:
        raise HTTPException(404, f"cell {cell_id} not found")
    return rows[0]
