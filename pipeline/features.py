"""Join pipeline outputs into the two model-ready tables.

features.tract_features  one row per 2020 census tract (rooftop model + equity index)
features.grid_features   one row per 1 km cell (utility-scale terrain model)

Grid label: pv_frac is the share of the cell covered by USPVDB facility polygons;
has_pv marks cells with at least PV_MIN_FRAC coverage. Note that WorldCover (2021)
may already show existing arrays as built/bare land, so models that use land cover
features should keep that leakage in mind.
"""

import geopandas as gpd
import numpy as np

from pipeline.config import GRID_SIZE
from pipeline.irradiance import sample
from pipeline.utils import load, save

PV_MIN_FRAC = 0.01  # 1 ha of a 1 km2 cell


def build_tract_features() -> gpd.GeoDataFrame:
    tracts = load("tracts")
    df = (
        tracts.merge(load("acs"), on="geoid", how="left")
        .merge(load("lead"), on="geoid", how="left")
        .merge(load("deepsolar"), on="geoid", how="left")
        .merge(load("tract_irradiance", schema="features"), on="geoid", how="left")
        .merge(load("tract_terrain", schema="features"), on="geoid", how="left")
    )
    land_km2 = df["aland_m2"].replace(0, np.nan) / 1e6
    df["pop_density_km2"] = (df["population"] / land_km2).round(1)
    return df


def build_grid_features() -> gpd.GeoDataFrame:
    grid = load("grid", schema="features")
    df = grid.merge(load("grid_terrain", schema="features"), on="cell_id", how="left")
    centroids = df.centroid

    df["ghi_kwh_m2_day"] = sample(centroids, load("power_climatology"))

    pts = gpd.GeoDataFrame(geometry=centroids, crs=df.crs)
    for name, table in (("line", "osm_power_lines"), ("substation", "osm_substations")):
        near = gpd.sjoin_nearest(pts, load(table)[["geometry"]], distance_col="d")
        df[f"dist_{name}_km"] = (near.groupby(level=0)["d"].min() / 1000).round(3)

    pv = load("uspvdb")
    overlap = gpd.overlay(df[["cell_id", "geometry"]], pv[["p_year", "geometry"]], how="intersection")
    overlap["area"] = overlap.area
    per_cell = overlap.groupby("cell_id").agg(pv_area=("area", "sum"), pv_first_year=("p_year", "min"))
    df = df.merge(per_cell, on="cell_id", how="left")
    df["pv_frac"] = (df["pv_area"].fillna(0) / GRID_SIZE**2).clip(upper=1).round(4)
    df["has_pv"] = (df["pv_frac"] >= PV_MIN_FRAC).astype(int)
    return df.drop(columns="pv_area")


def run_tracts() -> None:
    save(build_tract_features(), "tract_features", schema="features")


def run_grid() -> None:
    save(build_grid_features(), "grid_features", schema="features")
