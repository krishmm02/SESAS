"""GeocodingService: street address or coordinate -> 2020 census tract GEOID.

    uvicorn services.geocoding:app --port 8007

Wraps the US Census Geocoder, which is free and needs no API key. The vintage is pinned to
Census2020_Current because every table in this project sits on 2020 tract boundaries; a
2010-vintage lookup would return GEOIDs that silently fail to join.

This service resolves location only. It deliberately knows nothing about solar - composing
a lookup with a score is VisualizationService's job, so that each service stays replaceable
on its own.
"""

import os

import requests
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="GeocodingService")

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/geographies"
BENCHMARK = "Public_AR_Current"
VINTAGE = "Census2020_Current"          # must match the tract vintage used project-wide
LAYER = "Census Tracts"
TIMEOUT = int(os.getenv("GEOCODER_TIMEOUT", "20"))

STATE_FIPS = "24"
STATE_NAME = "Maryland"


def _call(endpoint: str, params: dict) -> dict:
    try:
        r = requests.get(f"{CENSUS_URL}/{endpoint}", timeout=TIMEOUT,
                         params={**params, "benchmark": BENCHMARK, "vintage": VINTAGE,
                                 "layers": LAYER, "format": "json"})
        r.raise_for_status()
        return r.json()["result"]
    except requests.RequestException as e:
        raise HTTPException(503, f"Census Geocoder unavailable: {e}") from e
    except (KeyError, ValueError) as e:
        raise HTTPException(502, f"Unexpected Census Geocoder response: {e}") from e


def _tract(geographies: dict) -> dict:
    tracts = geographies.get(LAYER) or []
    if not tracts:
        raise HTTPException(404, "no census tract found for that location")
    return tracts[0]


def _in_maryland(geoid: str, where: str) -> None:
    """Every downstream service validates GEOIDs as ^24\\d{9}$, so an out-of-state hit
    would otherwise surface as a confusing 422 two services later."""
    if not geoid.startswith(STATE_FIPS):
        raise HTTPException(
            422, f"{where} resolves to tract {geoid}, which is outside {STATE_NAME}. "
                 f"This project only covers {STATE_NAME}.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "vintage": VINTAGE, "coverage": STATE_NAME}


@app.get("/geocode")
def geocode(address: str = Query(min_length=5, max_length=200,
                                 description="e.g. 100 N Charles St, Baltimore, MD")) -> dict:
    result = _call("onelineaddress", {"address": address})
    matches = result.get("addressMatches") or []
    if not matches:
        # The Census address file has real gaps - "9000 Rockville Pike, Bethesda, MD"
        # fails even with its ZIP. Say so, rather than implying the input was malformed.
        raise HTTPException(
            404, f"The Census geocoder has no record of {address!r}. Include city and "
                 "state, and note its address file is incomplete - a nearby street "
                 "number on a major road usually resolves.")

    match = matches[0]
    tract = _tract(match["geographies"])
    _in_maryland(tract["GEOID"], f"{address!r}")
    return {
        "query": address,
        "matched_address": match["matchedAddress"],
        "lat": match["coordinates"]["y"],
        "lon": match["coordinates"]["x"],
        "geoid": tract["GEOID"],
        "tract_name": tract.get("NAME"),
        "alternatives": len(matches) - 1,
    }


@app.get("/geocode/reverse")
def reverse(lat: float = Query(ge=37.0, le=40.0, description="Maryland latitude"),
            lon: float = Query(ge=-80.0, le=-74.5, description="Maryland longitude")) -> dict:
    result = _call("coordinates", {"x": lon, "y": lat})
    tract = _tract(result.get("geographies") or {})
    _in_maryland(tract["GEOID"], f"({lat}, {lon})")
    return {"lat": lat, "lon": lon, "geoid": tract["GEOID"], "tract_name": tract.get("NAME")}
