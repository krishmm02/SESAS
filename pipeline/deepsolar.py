"""Step 3b: DeepSolar installed solar, moved from 2010 to 2020 census tracts.

DeepSolar (Stanford, 2016 imagery) reports on 2010 tracts. Counts are split across
2020 tracts using the Census 2020-2010 tract relationship file, weighted by the
share of each 2010 tract's land area that falls in each 2020 tract.

Table: raw.deepsolar
"""

import numpy as np
import pandas as pd

from pipeline.config import RAW_DIR, STATE_ABBR, STATE_FIPS
from pipeline.utils import download, save

DEEPSOLAR_URL = "https://web.stanford.edu/group/deepsolar/deepsolar_tract.csv"
RELATIONSHIP_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/"
    f"tab20_tract20_tract10_st{STATE_FIPS}.txt"
)

MIN_HOUSEHOLDS = 50  # below this, per-household adoption rates are noise

COUNT_COLUMNS = [
    "solar_system_count",
    "solar_system_count_residential",
    "solar_system_count_nonresidential",
    "total_panel_area",
    "total_panel_area_residential",
    "household_count",
]


def build() -> pd.DataFrame:
    ds_path = download(DEEPSOLAR_URL, RAW_DIR / "deepsolar" / "deepsolar_tract.csv")
    rel_path = download(RELATIONSHIP_URL, RAW_DIR / "deepsolar" / RELATIONSHIP_URL.rsplit("/", 1)[1])

    ds = pd.read_csv(ds_path, usecols=["fips", "state", *COUNT_COLUMNS], encoding="latin-1")
    ds = ds[ds["state"] == STATE_ABBR.lower()].copy()
    ds["geoid10"] = ds["fips"].astype("int64").astype(str).str.zfill(11)

    rel = pd.read_csv(rel_path, sep="|", dtype=str, encoding="utf-8-sig")
    rel = rel[["GEOID_TRACT_20", "GEOID_TRACT_10", "AREALAND_TRACT_10", "AREALAND_PART"]]
    rel.columns = ["geoid", "geoid10", "aland10", "aland_part"]
    rel[["aland10", "aland_part"]] = rel[["aland10", "aland_part"]].astype(float)
    rel["weight"] = rel["aland_part"] / rel["aland10"].replace(0, np.nan)

    merged = rel.merge(ds, on="geoid10", how="inner")
    merged[COUNT_COLUMNS] = merged[COUNT_COLUMNS].mul(merged["weight"], axis=0)
    out = merged.groupby("geoid")[COUNT_COLUMNS].sum().round(2)
    out.columns = [f"ds_{c}" for c in COUNT_COLUMNS]

    hh = out["ds_household_count"].where(out["ds_household_count"] >= MIN_HOUSEHOLDS)
    out["ds_res_systems_per_1000_hh"] = (1000 * out["ds_solar_system_count_residential"] / hh).round(2)
    return out.reset_index()


def run() -> None:
    save(build(), "deepsolar")
