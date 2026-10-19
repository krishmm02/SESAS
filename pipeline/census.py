"""Step 2: Census boundaries (tracts, state) and ACS 5-year tract attributes.

Tables: raw.state_boundary, raw.tracts, raw.acs
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from pipeline.config import ACS_YEAR, CRS, RAW_DIR, STATE_FIPS
from pipeline.utils import download, save

# Cartographic boundary files are clipped to the shoreline, so the Chesapeake Bay
# is excluded from tract and state shapes.
CB_URL = "https://www2.census.gov/geo/tiger/GENZ{y}/shp/cb_{y}_{name}_500k.zip"

# ACS variable -> column name
ACS_VARS = {
    "B01003_001E": "population",
    "B11001_001E": "households",
    "B19013_001E": "median_household_income",
    "B25077_001E": "median_home_value",
    "B25035_001E": "median_year_built",
    "B25003_001E": "occupied_units",
    "B25003_002E": "owner_occupied_units",
    "B25024_001E": "housing_units",
    "B25024_002E": "units_1_detached",
    "B25024_003E": "units_1_attached",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "below_poverty",
    "B03002_001E": "race_universe",
    "B03002_003E": "nh_white",
    "B03002_004E": "nh_black",
    "B03002_012E": "hispanic",
}


def load_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    out = RAW_DIR / "census"
    state_zip = download(CB_URL.format(y=ACS_YEAR, name="us_state"), out / f"cb_{ACS_YEAR}_us_state_500k.zip")
    tract_zip = download(CB_URL.format(y=ACS_YEAR, name=f"{STATE_FIPS}_tract"), out / f"cb_{ACS_YEAR}_{STATE_FIPS}_tract_500k.zip")

    state = gpd.read_file(state_zip)
    state = state.loc[state["STATEFP"] == STATE_FIPS, ["GEOID", "NAME", "geometry"]].to_crs(CRS)
    state.columns = ["geoid", "name", "geometry"]

    tracts = gpd.read_file(tract_zip)[["GEOID", "NAMELSADCO", "ALAND", "AWATER", "geometry"]]
    tracts.columns = ["geoid", "county", "aland_m2", "awater_m2", "geometry"]
    return state, tracts.to_crs(CRS)


def fetch_acs() -> pd.DataFrame:
    key = os.getenv("CENSUS_API_KEY", "")
    if not key or key.startswith("your_"):
        raise SystemExit("CENSUS_API_KEY is not set in .env (the Census API rejects keyless requests).")

    r = requests.get(
        f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5",
        params={
            "get": ",".join(ACS_VARS),
            "for": "tract:*",
            "in": f"state:{STATE_FIPS}",
            "key": key,
        },
        timeout=120,
    )
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    df = df.rename(columns=ACS_VARS)[["geoid", *ACS_VARS.values()]]

    values = df[list(ACS_VARS.values())].apply(pd.to_numeric)
    values = values.where(values >= 0)  # ACS uses large negative codes for "not available"
    df[values.columns] = values

    def rate(num, den):
        return (num / df[den].replace(0, np.nan)).round(4)

    df["owner_rate"] = rate(df["owner_occupied_units"], "occupied_units")
    df["single_family_detached_rate"] = rate(df["units_1_detached"], "housing_units")
    df["single_family_rate"] = rate(df["units_1_detached"] + df["units_1_attached"], "housing_units")
    df["poverty_rate"] = rate(df["below_poverty"], "poverty_universe")
    df["pct_nh_white"] = rate(df["nh_white"], "race_universe")
    df["pct_nh_black"] = rate(df["nh_black"], "race_universe")
    df["pct_hispanic"] = rate(df["hispanic"], "race_universe")
    df["pct_minority"] = (1 - df["pct_nh_white"]).round(4)
    return df


def run_boundaries() -> None:
    state, tracts = load_boundaries()
    save(state, "state_boundary")
    save(tracts, "tracts")


def run_acs() -> None:
    save(fetch_acs(), "acs")
