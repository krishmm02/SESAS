"""Step 3a: NREL LEAD 2022 (energy burden) aggregated to census tracts.

The LEAD tract file has one row per (tract, income bin, tenure, building type,
heating fuel) cohort, with UNITS = households and *P*UNITS = dollar totals.
Energy burden = total energy spending / total household income.

Table: raw.lead
"""

import zipfile

import numpy as np
import pandas as pd

from pipeline.config import RAW_DIR, STATE_ABBR
from pipeline.utils import download, save

LEAD_URL = f"https://data.openei.org/files/6219/{STATE_ABBR}-2022-LEAD-data.zip"
LOW_INCOME_BINS = {"0-30%", "30-60%", "60-80%"}  # below 80% of area median income

COLUMNS = {
    "FIP": "geoid",
    "AMI150": "ami_bin",
    "TEN": "tenure",
    "UNITS": "units",
    "HINCP*UNITS": "income",
    "ELEP*UNITS": "elec_cost",
    "GASP*UNITS": "gas_cost",
    "FULP*UNITS": "other_fuel_cost",
}


def build() -> pd.DataFrame:
    path = download(LEAD_URL, RAW_DIR / "lead" / f"{STATE_ABBR}-2022-LEAD-data.zip")
    with zipfile.ZipFile(path) as z, z.open(f"{STATE_ABBR} AMI Census Tracts 2022.csv") as f:
        df = pd.read_csv(f, usecols=list(COLUMNS), dtype={"FIP": str})
    df = df.rename(columns=COLUMNS)

    df["energy_cost"] = df["elec_cost"] + df["gas_cost"] + df["other_fuel_cost"]
    df["low_income"] = df["ami_bin"].isin(LOW_INCOME_BINS)
    df["renter"] = df["tenure"] == "REN"
    for col in ("units", "income", "energy_cost"):
        df[f"li_{col}"] = df[col].where(df["low_income"], 0.0)
    df["renter_units"] = df["units"].where(df["renter"], 0.0)

    t = df.groupby("geoid")[
        ["units", "income", "energy_cost", "elec_cost", "li_units", "li_income", "li_energy_cost", "renter_units"]
    ].sum()

    def ratio(num, den):
        return (t[num] / t[den].replace(0, np.nan)).round(4)

    out = pd.DataFrame(index=t.index)
    out["lead_households"] = t["units"].round(1)
    out["avg_household_income"] = ratio("income", "units").round(0)
    out["avg_energy_cost"] = ratio("energy_cost", "units").round(0)
    out["avg_electricity_cost"] = ratio("elec_cost", "units").round(0)
    out["energy_burden"] = ratio("energy_cost", "income")
    out["low_income_share"] = ratio("li_units", "units")
    out["low_income_energy_burden"] = ratio("li_energy_cost", "li_income")
    out["renter_share"] = ratio("renter_units", "units")
    return out.reset_index()


def run() -> None:
    save(build(), "lead")
