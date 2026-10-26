"""VisualizationService: map-ready GeoJSON for the dashboard.

    uvicorn services.visualization:app --port 8006

The dashboard is served from this same service at /app/, so the browser fetches
/geojson/equity same-origin and no CORS configuration is needed anywhere.

Unlike the other services, this one does not read scores from the database. It calls
EquityGapService over REST and joins the response to tract geometry from PostGIS. That
is deliberate: it is the composition step the architecture claims, and it means the map
renders exactly what the API serves, with no second code path that can drift from it.

Geometry is transformed to EPSG:4326 for Leaflet and simplified for payload size.
"""

import json
import os

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from pipeline.config import ROOT
from services.common import COUNTY, ping, query

app = FastAPI(title="VisualizationService")

EQUITY_URL = os.getenv("EQUITY_SERVICE_URL", "http://localhost:8005")
TIMEOUT = 20

# Tolerance in metres (geometry is stored in EPSG:26985). 25 m keeps tract shapes
# recognisable while cutting the payload substantially.
SIMPLIFY_M = 25

GEOM_SQL = f"""
SELECT geoid,
       ST_AsGeoJSON(ST_Transform(ST_SimplifyPreserveTopology(geometry, {SIMPLIFY_M}), 4326)) AS geom
FROM raw.tracts
"""


def fetch_equity(county: str | None) -> list[dict]:
    params = {"county": county} if county else {}
    try:
        r = requests.get(f"{EQUITY_URL}/equity", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(503, f"EquityGapService unavailable at {EQUITY_URL}: {e}") from e


@app.get("/health")
def health() -> dict:
    """Reports this service and the upstream it depends on, so a failure is attributable."""
    try:
        upstream = requests.get(f"{EQUITY_URL}/health", timeout=5).json().get("status", "unknown")
    except requests.RequestException as e:
        upstream = f"unreachable: {e.__class__.__name__}"
    return {**ping(), "upstream": {"equity": upstream, "url": EQUITY_URL}}


@app.get("/geojson/equity")
def equity_geojson(county: str | None = Query(None, pattern=COUNTY, description="e.g. 24510")) -> dict:
    scores = {row["geoid"]: row for row in fetch_equity(county)}
    if not scores:
        raise HTTPException(404, "no scored tracts for that county - has the model been trained?")

    sql = GEOM_SQL + (" WHERE geoid LIKE :prefix" if county else "")
    params = {"prefix": county + "%"} if county else {}

    features = []
    for row in query(sql, params):
        score = scores.get(row["geoid"])
        if score is None:
            continue  # tract has geometry but no score (e.g. too few households)
        features.append({
            "type": "Feature",
            "id": row["geoid"],
            "geometry": json.loads(row["geom"]),
            "properties": score,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {"county": county, "count": len(features), "source": f"{EQUITY_URL}/equity"},
    }


# Mounted last so it cannot shadow the API routes above.
app.mount("/app", StaticFiles(directory=ROOT / "web", html=True), name="dashboard")
