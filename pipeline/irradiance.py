"""Step 4: NASA POWER long-term climatology (IrradianceService data).

POWER's regional endpoint returns a regular grid of points for a bounding box, one
parameter per request (solar parameters on a 1-degree grid, T2M on 0.5 x 0.625).
sample() bilinearly interpolates a parameter at any set of points.

Tables: raw.power_climatology  one row per (parameter, POWER grid point)
        features.tract_irradiance  one row per census tract
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RegularGridInterpolator

from pipeline.config import BBOX, CRS, RAW_DIR
from pipeline.utils import load, save

POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/regional"
# POWER parameter -> output column prefix. Irradiance in kWh/m2/day, KT unitless, T2M in C.
PARAMETERS = {
    "ALLSKY_SFC_SW_DWN": "ghi",            # global horizontal irradiance
    "ALLSKY_SFC_SW_DNI": "dni",            # direct normal irradiance
    "ALLSKY_SFC_SW_DIFF": "dhi",           # diffuse horizontal irradiance
    "CLRSKY_SFC_SW_DWN": "clearsky_ghi",   # GHI if there were no clouds
    "ALLSKY_KT": "clearness_index",        # share of top-of-atmosphere sunlight reaching ground
    "T2M": "temp_c",                       # air temperature at 2 m
}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "ANN"]
MISSING = -999


def fetch(parameter: str) -> dict:
    path = RAW_DIR / "power" / f"{parameter.lower()}_climatology.json"
    if not path.exists():
        r = requests.get(
            POWER_URL,
            params={
                "parameters": parameter,
                "community": "RE",
                # pad by one grid step so every Maryland point lies between grid points
                "longitude-min": BBOX["west"] - 1,
                "longitude-max": BBOX["east"] + 1,
                "latitude-min": BBOX["south"] - 1,
                "latitude-max": BBOX["north"] + 1,
                "format": "JSON",
            },
            timeout=300,
        )
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> gpd.GeoDataFrame:
    records = []
    for parameter in PARAMETERS:
        for feat in fetch(parameter)["features"]:
            lon, lat = feat["geometry"]["coordinates"][:2]
            values = feat["properties"]["parameter"][parameter]
            records.append({"parameter": parameter, "lon": lon, "lat": lat,
                            **{m.lower(): values[m] for m in MONTHS}})
    df = pd.DataFrame(records).replace(MISSING, np.nan)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")


def sample(points: gpd.GeoSeries, power: gpd.GeoDataFrame, parameter: str = "ALLSKY_SFC_SW_DWN",
           month: str = "ann") -> np.ndarray:
    """Value of one parameter/month at each point, bilinear between POWER grid points."""
    rows = power[power["parameter"] == parameter]
    grid = rows.pivot(index="lat", columns="lon", values=month).sort_index()
    interp = RegularGridInterpolator(
        (grid.index.values, grid.columns.values), grid.values, bounds_error=False, fill_value=None
    )
    pts = points.to_crs("EPSG:4326")
    return interp(np.column_stack([pts.y, pts.x])).round(3)


def build_tracts(tracts: gpd.GeoDataFrame, power: gpd.GeoDataFrame) -> pd.DataFrame:
    """Annual value of every parameter plus monthly GHI, at each tract's representative
    point (a point guaranteed inside the polygon, unlike the centroid)."""
    pts = tracts.representative_point()
    out = pd.DataFrame({"geoid": tracts["geoid"].to_numpy()})
    for parameter, name in PARAMETERS.items():
        out[name] = sample(pts, power, parameter)
    for m in MONTHS[:-1]:
        out[f"ghi_{m.lower()}"] = sample(pts, power, "ALLSKY_SFC_SW_DWN", m.lower())
    return out


def run() -> None:
    save(build().to_crs(CRS), "power_climatology")


def run_tracts() -> None:
    save(build_tracts(load("tracts"), load("power_climatology")), "tract_irradiance", schema="features")
