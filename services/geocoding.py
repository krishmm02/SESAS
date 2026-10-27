"""GeocodingService: street address or coordinate -> 2020 census tract GEOID.

    uvicorn services.geocoding:app --port 8007

Wraps the US Census Geocoder, which is free and needs no API key. The vintage is pinned to
Census2020_Current because every table in this project sits on 2020 tract boundaries; a
2010-vintage lookup would return GEOIDs that silently fail to join.

This service resolves location only. It deliberately knows nothing about solar - composing
a lookup with a score is VisualizationService's job, so that each service stays replaceable
on its own.

The Census geocoder needs a near-exact address, so search-as-you-type uses Photon
(komoot's OpenStreetMap geocoder: free, no key, tolerant of typos and partial input).
A picked suggestion carries coordinates, which /geocode/reverse turns into the 2020 tract
through Census as before - Photon picks the point, Census still owns the tract. Building
outlines come from the OpenStreetMap API by the id Photon returns. Both are public,
fair-use services, so responses are cached and the dashboard debounces keystrokes.
"""

import os
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="GeocodingService")

PHOTON_URL = "https://photon.komoot.io/api/"
OSM_API_URL = "https://api.openstreetmap.org/api/0.6"
USER_AGENT = "solar-equity-course-project/0.1 (UMBC CMSC 668)"
MD_BBOX = "-79.49,37.88,-75.04,39.73"   # lon/lat, for Photon's bbox filter
MD_CENTER = {"lat": 39.05, "lon": -76.75}  # biases ranking toward the state's population
OSM_TYPES = {"W": "way", "R": "relation", "N": "node"}

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


def _label(p: dict) -> str:
    """'5200 Park Heights Avenue, Baltimore, MD 21215', or a place name in front of it."""
    street = " ".join(x for x in (p.get("housenumber"), p.get("street")) if x)
    name = p.get("name") if p.get("name") != street else None
    city = p.get("city") or p.get("town") or p.get("village") or p.get("county")
    tail = " ".join(x for x in ("MD", p.get("postcode")) if x)
    return ", ".join(x for x in (name, street, city, tail) if x)


# Photon's fuzzy matching handles misspellings well but not abbreviations: "1 e prat st"
# misses One East Pratt Street while "1 east pratt street" finds it first.
STREET_TYPES = {"st": "street", "ave": "avenue", "av": "avenue", "rd": "road", "dr": "drive",
                "blvd": "boulevard", "pkwy": "parkway", "ln": "lane", "ct": "court",
                "cir": "circle", "hwy": "highway", "pl": "place", "ter": "terrace", "sq": "square"}
DIRECTIONS = {"n": "north", "s": "south", "e": "east", "w": "west",
              "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"}


def _expand(q: str) -> str:
    words = q.replace(",", " ").replace(".", " ").split()
    out = []
    for i, w in enumerate(words):
        if w in DIRECTIONS and i > 0 and words[i - 1].isdigit():  # "1 e pratt", not "e street"
            w = DIRECTIONS[w]
        elif w in STREET_TYPES and i > 0:                         # "pratt st", not "st marys"
            w = STREET_TYPES[w]
        out.append(w)
    return " ".join(out)


@lru_cache(maxsize=2048)
def _photon(q: str, limit: int) -> tuple:
    try:
        r = requests.get(PHOTON_URL, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT},
                         params={"q": _expand(q), "limit": 15, "bbox": MD_BBOX, "lang": "en", **MD_CENTER})
        r.raise_for_status()
        features = r.json()["features"]
    except requests.RequestException as e:
        raise HTTPException(503, f"Photon autocomplete unavailable: {e}") from e

    typed_number = q.split()[0] if q.split() and q.split()[0].isdigit() else None
    best = {}  # one suggestion per street address; shops in one building collapse to it
    for rank, f in enumerate(features):
        p = f["properties"]
        if p.get("state") != STATE_NAME:  # the bbox is a rectangle; it clips DC, VA, PA, DE
            continue
        lon, lat = f["geometry"]["coordinates"]
        s = {"label": _label(p), "lat": lat, "lon": lon,
             "osm_type": p.get("osm_type"), "osm_id": p.get("osm_id"),
             "is_address": bool(p.get("housenumber")), "kind": p.get("osm_value"),
             "_key": (not (typed_number and p.get("housenumber") == typed_number),
                      not p.get("housenumber"), rank)}
        addr = (p.get("housenumber"), p.get("street"), p.get("postcode")) if p.get("housenumber") else s["label"]
        # A shop is a point (N) inside its building (W): prefer the building, it has the
        # rooftop outline. A campus (R) is not inside one building, so it is never replaced.
        if addr not in best or (s["osm_type"] == "W" and best[addr]["osm_type"] == "N"):
            if addr in best:
                s["_key"] = min(s["_key"], best[addr]["_key"])
            best[addr] = s
    # Exact house number first, then any street address, then Photon's own order - a bus
    # stop named after a school is a worse answer than the house number being typed.
    out = sorted(best.values(), key=lambda s: s.pop("_key"))
    return tuple(out[:limit])


@app.get("/geocode/suggest")
def suggest(q: str = Query(min_length=3, max_length=200, description="partial address, typos ok"),
            limit: int = Query(6, ge=1, le=10)) -> list[dict]:
    """Search-as-you-type. Each suggestion has coordinates and, where OpenStreetMap maps the
    building, an osm_type/osm_id for /geocode/building."""
    return list(_photon(q.strip().lower(), limit))


@lru_cache(maxsize=1024)
def _outline(osm_type: str, osm_id: int) -> dict | None:
    """GeoJSON polygon of an OSM building way, or None if it is not one."""
    if osm_type != "W":
        return None  # address nodes have no outline; multipolygon relations are rare for houses
    try:
        r = requests.get(f"{OSM_API_URL}/way/{osm_id}/full.json", timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        if r.status_code in (404, 410):
            return None
        r.raise_for_status()
        elements = r.json()["elements"]
    except requests.RequestException as e:
        raise HTTPException(503, f"OpenStreetMap API unavailable: {e}") from e
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
    way = next(e for e in elements if e["type"] == "way")
    if "building" not in way.get("tags", {}):
        return None
    ring = [nodes[n] for n in way["nodes"] if n in nodes]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"osm_id": osm_id, "building": way["tags"]["building"],
                           "levels": way["tags"].get("building:levels")}}


@app.get("/geocode/building")
def building(osm_type: str = Query(pattern="^[WRN]$"), osm_id: int = Query(gt=0)) -> dict:
    """Rooftop outline for a suggestion, from OpenStreetMap."""
    outline = _outline(osm_type, osm_id)
    if outline is None:
        raise HTTPException(404, f"OpenStreetMap has no building outline for {OSM_TYPES[osm_type]} {osm_id}")
    return outline


@app.get("/geocode/reverse")
def reverse(lat: float = Query(ge=37.0, le=40.0, description="Maryland latitude"),
            lon: float = Query(ge=-80.0, le=-74.5, description="Maryland longitude")) -> dict:
    result = _call("coordinates", {"x": lon, "y": lat})
    tract = _tract(result.get("geographies") or {})
    _in_maryland(tract["GEOID"], f"({lat}, {lon})")
    return {"lat": lat, "lon": lon, "geoid": tract["GEOID"], "tract_name": tract.get("NAME")}
