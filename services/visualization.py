"""VisualizationService: map-ready GeoJSON for the dashboard.

    uvicorn services.visualization:app --port 8006

The dashboard is served from this same service at /app/, so the browser fetches
/geojson/equity same-origin and no CORS configuration is needed anywhere.

Unlike the other services, this one does not read scores from the database. It calls
EquityGapService, DevelopabilityService and GeocodingService over REST and joins the
responses to geometry from PostGIS. That is deliberate: it is the composition step the
architecture claims, and it means the map renders exactly what the API serves, with no
second code path that can drift from it.

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

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and stalls ~2 s per call.
EQUITY_URL = os.getenv("EQUITY_SERVICE_URL", "http://127.0.0.1:8005")
GEOCODE_URL = os.getenv("GEOCODE_SERVICE_URL", "http://127.0.0.1:8007")
DEVELOPABILITY_URL = os.getenv("DEVELOPABILITY_SERVICE_URL", "http://127.0.0.1:8009")
TIMEOUT = 25
UPSTREAMS = {"equity": EQUITY_URL, "geocoding": GEOCODE_URL, "developability": DEVELOPABILITY_URL}

# Tolerance in metres (geometry is stored in EPSG:26985). 25 m keeps tract shapes
# recognisable while cutting the payload substantially.
SIMPLIFY_M = 25

GEOM_SQL = f"""
SELECT geoid,
       ST_AsGeoJSON(ST_Transform(ST_SimplifyPreserveTopology(geometry, {SIMPLIFY_M}), 4326)) AS geom
FROM raw.tracts
"""


GRID_SQL = """
SELECT cell_id, ST_AsGeoJSON(ST_Transform(geometry, 4326), 5) AS geom
FROM features.grid
WHERE cell_id = ANY(:ids)
"""


def fetch(name: str, url: str, params: dict) -> list[dict]:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(503, f"{name} unavailable at {url}: {e}") from e


def fetch_equity(county: str | None) -> list[dict]:
    return fetch("EquityGapService", f"{EQUITY_URL}/equity", {"county": county} if county else {})


@app.get("/health")
def health() -> dict:
    """Reports this service and each upstream it depends on, so a failure is attributable."""
    up = {}
    for name, url in UPSTREAMS.items():
        try:
            up[name] = requests.get(f"{url}/health", timeout=5).json().get("status", "unknown")
        except requests.RequestException as e:
            up[name] = f"unreachable: {e.__class__.__name__}"
    return {**ping(), "upstream": {**up, "urls": UPSTREAMS}}


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


@app.get("/geojson/developability")
def developability_geojson(
    county: str | None = Query(None, pattern=COUNTY, description="e.g. 24047"),
    tier: str | None = Query(None, pattern="^(HIGH|MEDIUM|LOW|EXCLUDED)$"),
) -> dict:
    """1 km cells with their siting score. Statewide with no tier would be ~27,000 squares,
    so the statewide default is the HIGH tier only; a county returns every cell."""
    if not county and not tier:
        tier = "HIGH"
    params = {"limit": 30000, **({"county": county} if county else {}), **({"tier": tier} if tier else {})}
    scores = {r["cell_id"]: r for r in fetch("DevelopabilityService", f"{DEVELOPABILITY_URL}/developability", params)}
    if not scores:
        raise HTTPException(404, "no scored cells - has the developability model been trained?")

    features = [
        {"type": "Feature", "id": row["cell_id"], "geometry": json.loads(row["geom"]),
         "properties": scores[row["cell_id"]]}
        for row in query(GRID_SQL, {"ids": list(scores)})
    ]
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {"county": county, "tier": tier, "count": len(features),
                     "source": f"{DEVELOPABILITY_URL}/developability"},
    }


@app.get("/lookup")
def lookup(address: str = Query(min_length=5, max_length=200)) -> dict:
    """Address -> tract -> equity score, in one call.

    Two upstreams chained: GeocodingService resolves the address to a GEOID, then
    EquityGapService scores it. Neither knows about the other - this is the only place
    the two are joined, which is what keeps each of them replaceable.
    """
    try:
        g = requests.get(f"{GEOCODE_URL}/geocode", params={"address": address}, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise HTTPException(503, f"GeocodingService unavailable at {GEOCODE_URL}: {e}") from e
    if g.status_code != 200:
        # Pass the upstream's own explanation through rather than flattening it to a 500;
        # "outside Maryland" and "no match" need different fixes from the user.
        raise HTTPException(g.status_code, g.json().get("detail", g.text))
    located = g.json()

    try:
        e = requests.get(f"{EQUITY_URL}/equity/{located['geoid']}", timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise HTTPException(503, f"EquityGapService unavailable at {EQUITY_URL}: {exc}") from exc
    if e.status_code == 404:
        raise HTTPException(404, f"tract {located['geoid']} has no score - too few "
                                 "households for a reliable adoption rate")
    if e.status_code != 200:
        raise HTTPException(e.status_code, e.json().get("detail", e.text))

    return {"location": located, "equity": e.json()}


# Mounted last so it cannot shadow the API routes above.
app.mount("/app", StaticFiles(directory=ROOT / "web", html=True), name="dashboard")
