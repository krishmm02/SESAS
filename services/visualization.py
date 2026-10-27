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

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and stalls ~2 s per
# call, which made every address search take ~4.5 s instead of ~0.5 s.
EQUITY_URL = os.getenv("EQUITY_SERVICE_URL", "http://127.0.0.1:8005")
GEOCODE_URL = os.getenv("GEOCODE_SERVICE_URL", "http://127.0.0.1:8007")
TIMEOUT = 25

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
    up = {}
    for name, url in (("equity", EQUITY_URL), ("geocoding", GEOCODE_URL)):
        try:
            up[name] = requests.get(f"{url}/health", timeout=5).json().get("status", "unknown")
        except requests.RequestException as e:
            up[name] = f"unreachable: {e.__class__.__name__}"
    return {**ping(), "upstream": {**up, "urls": {"equity": EQUITY_URL, "geocoding": GEOCODE_URL}}}


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


def geocoder(path: str, params: dict) -> dict | list:
    try:
        g = requests.get(f"{GEOCODE_URL}{path}", params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise HTTPException(503, f"GeocodingService unavailable at {GEOCODE_URL}: {e}") from e
    if g.status_code != 200:
        # Pass the upstream's own explanation through rather than flattening it to a 500;
        # "outside Maryland" and "no match" need different fixes from the user.
        raise HTTPException(g.status_code, g.json().get("detail", g.text))
    return g.json()


@app.get("/suggest")
def suggest(q: str = Query(min_length=3, max_length=200)) -> list[dict]:
    """Address suggestions for the search box, relayed so the browser stays same-origin."""
    return geocoder("/geocode/suggest", {"q": q})


@app.get("/lookup")
def lookup(
    address: str | None = Query(None, min_length=5, max_length=200),
    lat: float | None = Query(None), lon: float | None = Query(None),
    label: str | None = Query(None, max_length=200, description="shown as the matched address"),
    osm_type: str | None = Query(None, pattern="^[WRN]$"), osm_id: int | None = Query(None, gt=0),
) -> dict:
    """Address or picked suggestion -> tract -> equity score, in one call.

    Two upstreams chained: GeocodingService resolves the location to a GEOID, then
    EquityGapService scores it. Neither knows about the other - this is the only place
    the two are joined, which is what keeps each of them replaceable.

    Pass either ?address= (Census exact match) or the lat/lon of a picked suggestion. With
    osm_type/osm_id the response also carries the building outline for the map; a missing
    outline is not an error, the pin alone still locates the address.
    """
    if lat is not None and lon is not None:
        located = geocoder("/geocode/reverse", {"lat": lat, "lon": lon})
        located["matched_address"] = label or f"{lat:.5f}, {lon:.5f}"
    elif address:
        located = geocoder("/geocode", {"address": address})
    else:
        raise HTTPException(422, "pass ?address= or both ?lat= and ?lon=")

    try:
        e = requests.get(f"{EQUITY_URL}/equity/{located['geoid']}", timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise HTTPException(503, f"EquityGapService unavailable at {EQUITY_URL}: {exc}") from exc
    if e.status_code == 404:
        raise HTTPException(404, f"tract {located['geoid']} has no score - too few "
                                 "households for a reliable adoption rate")
    if e.status_code != 200:
        raise HTTPException(e.status_code, e.json().get("detail", e.text))

    building = None
    if osm_type and osm_id:
        try:
            building = geocoder("/geocode/building", {"osm_type": osm_type, "osm_id": osm_id})
        except HTTPException:
            pass  # no outline mapped, or OSM unreachable: the pin is enough

    return {"location": located, "equity": e.json(), "building": building}


# Mounted last so it cannot shadow the API routes above.
app.mount("/app", StaticFiles(directory=ROOT / "web", html=True), name="dashboard")
