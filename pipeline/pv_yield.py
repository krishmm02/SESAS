"""Step: rooftop PV specific yield per census tract (YieldService data).

Physics, not ML: kWh a 1 kW(dc) rooftop system produces per year, via pvlib.

POWER gives monthly-mean daily totals, not hours, so each month is one representative
day (the 15th): a clear-sky hourly shape scaled to POWER's measured GHI, split into
direct and diffuse with POWER's measured diffuse fraction for that month, transposed to
the plane of the roof, then cell temperature and PVWatts losses. Summed over the year.

Assumptions match NREL PVWatts residential defaults so results can be checked there:
south-facing, 20 deg tilt, 14.08% system losses, 96% inverter, -0.37 %/C.

Table: features.tract_yield  one row per census tract
"""

import calendar

import numpy as np
import pandas as pd
import pvlib

from pipeline.irradiance import sample
from pipeline.utils import load, save

TILT = 20
AZIMUTH = 180
ALBEDO = 0.2
LOSSES = 0.1408
INVERTER_EFF = 0.96
GAMMA_PDC = -0.0037
WIND_MS = 1.0
TEMP_PARAMS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
MONTHS = [m.lower() for m in calendar.month_abbr[1:]]
DAYS = [calendar.monthrange(2023, i)[1] for i in range(1, 13)]
# one representative day per month, hourly, local standard time
TIMES = pd.DatetimeIndex(
    [ts for i in range(1, 13) for ts in pd.date_range(f"2023-{i:02d}-15", periods=24, freq="h")],
    tz="Etc/GMT+5",
)


def tract_yield(lat: float, lon: float, ghi: np.ndarray, dhi: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Monthly kWh per kW(dc) for one site. ghi/dhi in kWh/m2/day, temp in C, one value per month."""
    sun = pvlib.solarposition.get_solarposition(TIMES, lat, lon)
    cos_z = np.cos(np.radians(sun["apparent_zenith"].to_numpy()))
    shape = pvlib.clearsky.haurwitz(sun["apparent_zenith"])["ghi"].to_numpy().reshape(12, 24)

    # scale each day's clear-sky shape so its total matches the measured monthly GHI (W/m2 per hour)
    ghi_h = shape * (ghi * 1000 / shape.sum(axis=1))[:, None]
    dhi_h = ghi_h * (dhi / ghi)[:, None]
    ghi_h, dhi_h = ghi_h.ravel(), dhi_h.ravel()
    dni_h = np.where(cos_z > 0.087, (ghi_h - dhi_h) / np.maximum(cos_z, 0.087), 0)  # sun below 5 deg

    poa = pvlib.irradiance.get_total_irradiance(
        TILT, AZIMUTH, sun["apparent_zenith"], sun["azimuth"], dni_h, ghi_h, dhi_h,
        albedo=ALBEDO, model="isotropic",
    )["poa_global"].fillna(0).to_numpy()
    t_cell = pvlib.temperature.sapm_cell(poa, np.repeat(temp, 24), WIND_MS, **TEMP_PARAMS)
    dc = pvlib.pvsystem.pvwatts_dc(poa, t_cell, pdc0=1000, gamma_pdc=GAMMA_PDC)  # W per kW(dc)
    ac = dc * (1 - LOSSES) * INVERTER_EFF
    return ac.reshape(12, 24).sum(axis=1) / 1000 * np.array(DAYS)  # kWh per month


def build_tracts(tracts, power) -> pd.DataFrame:
    pts = tracts.representative_point()
    ll = pts.to_crs("EPSG:4326")
    ghi = np.column_stack([sample(pts, power, "ALLSKY_SFC_SW_DWN", m) for m in MONTHS])
    dhi = np.column_stack([sample(pts, power, "ALLSKY_SFC_SW_DIFF", m) for m in MONTHS])
    temp = np.column_stack([sample(pts, power, "T2M", m) for m in MONTHS])

    monthly = np.array([tract_yield(ll.y.iloc[i], ll.x.iloc[i], ghi[i], dhi[i], temp[i])
                        for i in range(len(tracts))])
    out = pd.DataFrame({"geoid": tracts["geoid"].to_numpy(),
                        "specific_yield_kwh_kw": monthly.sum(axis=1).round(1)})
    out["capacity_factor"] = (out["specific_yield_kwh_kw"] / 8760).round(4)
    for j, m in enumerate(MONTHS):
        out[f"yield_{m}"] = monthly[:, j].round(1)
    return out


def run_tracts() -> None:
    save(build_tracts(load("tracts"), load("power_climatology")), "tract_yield", schema="features")
