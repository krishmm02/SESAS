"""SolarSiteService: solar potential and a panel-layout estimate for any point in Maryland.

    uvicorn services.site:app --port 8011

Physics only - no households, income or equity - so it answers the same way for a rooftop
and for open land. Sunlight is NASA POWER climatology at the exact point (bilinear between
grid points), turned into kWh with pvlib (pipeline/pv_yield.py, PVWatts-equivalent losses).

/site/panel is the single source of truth for module size and spacing: the dashboard lays
panels out with it, so the count it suggests and the kW this service estimates always agree.

Energy only: what a layout generates, not what it is worth. Prices, savings and payback
are the endterm cost layer.

CO2 factor: 0.2708 kg/kWh, EPA eGRID 2023, RFC East subregion (596.9 lb/MWh).
"""

from functools import lru_cache

import geopandas as gpd
import numpy as np
import pvlib
from fastapi import FastAPI, HTTPException, Query
from shapely.geometry import Point

from pipeline import pv_yield
from pipeline.irradiance import sample
from pipeline.utils import load

app = FastAPI(title="SolarSiteService")

# A common ~400 W residential module (108 half-cut cells): 1.72 m x 1.13 m.
PANEL = {"watts": 400, "length_m": 1.72, "width_m": 1.13, "gap_m": 0.02}
# Roof: flush on a pitched roof, facing wherever the roof faces, kept 0.9 m (3 ft) from every
#   edge for fire access.
# Flat: a commercial flat roof, low ballasted racks at 10 deg facing south, rows at a ground
#   coverage ratio of 0.7, 1.2 m (4 ft) from the edges.
# Ground: fixed racks at 30 deg facing south, rows spaced so the next row is not shaded -
#   a ground coverage ratio of 0.4 is typical for fixed-tilt arrays at Maryland's latitude.
OPEN_RACK = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
MOUNTS = {
    "roof": {"tilt": pv_yield.TILT, "setback_m": 0.9, "gcr": None,
             "temp": pv_yield.TEMP_PARAMS},
    "flat": {"tilt": 10, "setback_m": 1.2, "gcr": 0.7, "temp": OPEN_RACK},
    "ground": {"tilt": 30, "setback_m": 2.0, "gcr": 0.4, "temp": OPEN_RACK},
}
MONTHS = pv_yield.MONTHS
CO2_KG_PER_KWH = 596.904 * 0.45359237 / 1000  # eGRID 2023 RFCE, lb/MWh -> kg/kWh
MD = {"lat": (37.88, 39.73), "lon": (-79.49, -75.04)}

LAT = Query(ge=MD["lat"][0], le=MD["lat"][1], description="Maryland latitude")
LON = Query(ge=MD["lon"][0], le=MD["lon"][1], description="Maryland longitude")


@lru_cache(maxsize=1)
def _power():
    return load("power_climatology")


def _climate(lat: float, lon: float) -> dict:
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    power = _power()
    per_month = lambda param: np.array([sample(pt, power, param, m)[0] for m in MONTHS])
    return {"ghi": per_month("ALLSKY_SFC_SW_DWN"), "dhi": per_month("ALLSKY_SFC_SW_DIFF"),
            "temp": per_month("T2M")}


@lru_cache(maxsize=4096)
def _yield(lat: float, lon: float, tilt: float, azimuth: float, mount: str) -> tuple:
    """Monthly kWh per kW(dc). Cached on a ~100 m rounding of the point: climatology does
    not change over that distance, and dragging panels re-asks constantly."""
    c = _climate(lat, lon)
    return tuple(pv_yield.tract_yield(lat, lon, c["ghi"], c["dhi"], c["temp"],
                                      tilt=tilt, azimuth=azimuth, temp_params=MOUNTS[mount]["temp"]))


def _monthly(lat, lon, tilt, azimuth, mount) -> np.ndarray:
    return np.array(_yield(round(lat, 3), round(lon, 3), round(tilt, 1), round(azimuth), mount))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "coverage": "Maryland", "uses": "NASA POWER climatology + pvlib"}


@app.get("/site/panel")
def panel_spec() -> dict:
    """Module and spacing rules the dashboard's layout uses."""
    return {"panel": PANEL,
            "mounts": {k: {kk: vv for kk, vv in v.items() if kk != "temp"} for k, v in MOUNTS.items()}}


@app.get("/site/potential")
def potential(lat: float = LAT, lon: float = LON) -> dict:
    """How sunny this exact spot is, and what 1 kW of panels makes here on a roof and on a rack."""
    c = _climate(lat, lon)
    days = np.array(pv_yield.DAYS)
    per_kw = {m: round(float(_monthly(lat, lon, MOUNTS[m]["tilt"], 180, m).sum())) for m in MOUNTS}
    return {
        "lat": lat, "lon": lon,
        "ghi_kwh_m2_day": round(float((c["ghi"] * days).sum() / days.sum()), 2),
        "ghi_kwh_m2_year": round(float((c["ghi"] * days).sum())),
        "ghi_monthly": dict(zip(MONTHS, c["ghi"].round(2).tolist())),
        "specific_yield_kwh_kw": per_kw,   # facing south; a pitched roof facing elsewhere makes less
        "source": "NASA POWER climatology at this point, pvlib, PVWatts-equivalent losses",
    }


@app.get("/site/estimate")
def estimate(
    lat: float = LAT, lon: float = LON,
    panels: int = Query(ge=1, le=100000, description="number of modules"),
    mount: str = Query("roof", pattern="^(roof|flat|ground)$"),
    tilt: float | None = Query(None, ge=0, le=60, description="default: the mount's tilt"),
    azimuth: float = Query(180, ge=0, lt=360, description="direction the panels face, 180 = south"),
    panel_w: float = Query(PANEL["watts"], gt=0, le=1000),
) -> dict:
    """Energy a given number of panels generates at this point, by month and per year."""
    tilt = MOUNTS[mount]["tilt"] if tilt is None else tilt
    monthly_per_kw = _monthly(lat, lon, tilt, azimuth, mount)
    kw = panels * panel_w / 1000
    monthly = monthly_per_kw * kw
    kwh = float(monthly.sum())
    if not kwh:
        raise HTTPException(422, "this layout produces no energy - check tilt and azimuth")
    return {
        "panels": panels, "system_kw": round(kw, 2), "mount": mount,
        "tilt": tilt, "azimuth": azimuth,
        "annual_kwh": round(kwh), "monthly_kwh": dict(zip(MONTHS, monthly.round().astype(int).tolist())),
        "specific_yield_kwh_kw": round(float(monthly_per_kw.sum())),
        "capacity_factor": round(kwh / (kw * 8760), 4),
        "co2_avoided_kg_year": round(kwh * CO2_KG_PER_KWH),
        "assumptions": {"panel_w": panel_w, "co2_kg_per_kwh": round(CO2_KG_PER_KWH, 4)},
    }
