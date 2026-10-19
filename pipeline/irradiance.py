"""Step 4: NASA POWER long-term solar irradiance (GHI) climatology.

POWER returns a regular 1-degree grid of points for a bounding box. sample_ghi()
bilinearly interpolates annual GHI (kWh/m2/day) at any set of points.

Table: raw.power_ghi
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RegularGridInterpolator

from pipeline.config import BBOX, CRS, RAW_DIR
from pipeline.utils import save

POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/regional"
PARAMETER = "ALLSKY_SFC_SW_DWN"  # all-sky surface shortwave downward irradiance (GHI)
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "ANN"]


def fetch() -> dict:
    path = RAW_DIR / "power" / "ghi_climatology.json"
    if not path.exists():
        r = requests.get(
            POWER_URL,
            params={
                "parameters": PARAMETER,
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
    for feat in fetch()["features"]:
        lon, lat = feat["geometry"]["coordinates"][:2]
        values = feat["properties"]["parameter"][PARAMETER]
        records.append({"lon": lon, "lat": lat, **{f"ghi_{m.lower()}": values[m] for m in MONTHS}})
    df = pd.DataFrame(records)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")


def sample_ghi(points: gpd.GeoSeries, power: gpd.GeoDataFrame) -> np.ndarray:
    """Annual GHI at each point, bilinear between POWER grid points."""
    grid = power.pivot(index="lat", columns="lon", values="ghi_ann").sort_index()
    interp = RegularGridInterpolator(
        (grid.index.values, grid.columns.values), grid.values, bounds_error=False, fill_value=None
    )
    pts = points.to_crs("EPSG:4326")
    return interp(np.column_stack([pts.y, pts.x])).round(3)


def run() -> None:
    save(build().to_crs(CRS), "power_ghi")
