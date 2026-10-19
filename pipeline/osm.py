"""Step 5c: OpenStreetMap transmission lines and substations (Overpass API).

Tables: raw.osm_power_lines, raw.osm_substations
"""

import json
import re
import time

import geopandas as gpd
import requests
from shapely.geometry import LineString, Point

from pipeline.config import BBOX, CRS, RAW_DIR
from pipeline.utils import USER_AGENT, save

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
BBOX_STR = f'{BBOX["south"]},{BBOX["west"]},{BBOX["north"]},{BBOX["east"]}'
QUERIES = {
    "power_lines": f'[out:json][timeout:600];way["power"="line"]({BBOX_STR});out geom;',
    "substations": (
        f'[out:json][timeout:600];'
        f'(node["power"="substation"]({BBOX_STR});way["power"="substation"]({BBOX_STR}););out center;'
    ),
}


def fetch(name: str) -> list[dict]:
    path = RAW_DIR / "osm" / f"{name}.json"
    if not path.exists():
        # Overpass servers are busy and sometimes drop connections; retry across mirrors
        for attempt, url in enumerate(OVERPASS_URLS * 2, start=1):
            try:
                r = requests.post(url, data={"data": QUERIES[name]}, timeout=900,
                                  headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  overpass attempt {attempt} failed ({url}): {e}")
                time.sleep(10 * attempt)
        else:
            raise RuntimeError(f"Overpass query '{name}' failed on all attempts")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))["elements"]


def max_voltage_kv(tag: str | None) -> float | None:
    """'230000;115000' -> 230.0"""
    volts = [int(v) for v in re.findall(r"\d+", tag or "")]
    return max(volts) / 1000 if volts else None


def build_lines() -> gpd.GeoDataFrame:
    rows = [
        {
            "osm_id": el["id"],
            "voltage_kv": max_voltage_kv(el.get("tags", {}).get("voltage")),
            "operator": el.get("tags", {}).get("operator"),
            "geometry": LineString([(p["lon"], p["lat"]) for p in el["geometry"]]),
        }
        for el in fetch("power_lines")
        if len(el.get("geometry", [])) >= 2
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(CRS)


def build_substations() -> gpd.GeoDataFrame:
    rows = []
    for el in fetch("substations"):
        loc = el if el["type"] == "node" else el.get("center")
        if loc:
            tags = el.get("tags", {})
            rows.append({
                "osm_id": el["id"],
                "name": tags.get("name"),
                "voltage_kv": max_voltage_kv(tags.get("voltage")),
                "geometry": Point(loc["lon"], loc["lat"]),
            })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(CRS)


def run() -> None:
    save(build_lines(), "osm_power_lines")
    save(build_substations(), "osm_substations")
