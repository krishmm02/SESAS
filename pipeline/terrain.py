"""Step 5b: terrain and land cover features per 1 km grid cell.

- Copernicus DEM GLO-90 tiles -> elevation, slope, south-facing share
- ESA WorldCover 2021 (read remotely from its ~40 m overview) -> land cover shares

Both are resampled to FINE_RES on the grid's origin, then summarized per cell.
Intermediate 100 m rasters are cached in data/processed/.

Table: features.grid_terrain
"""

import warnings

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from pipeline.config import CRS, FINE_RES, GRID_SIZE, PROCESSED_DIR, RAW_DIR
from pipeline.grid import grid_spec
from pipeline.utils import download, load, save

DEM_URL = (
    "https://copernicus-dem-90m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N{lat:02d}W{lon:03d}_Map.tif"
)
WORLDCOVER_CLASSES = {
    10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
    60: "bare", 80: "water", 90: "wetland",
}
FLAT_SLOPE_DEG = 2.86  # 5% grade, a common upper limit for utility-scale PV siting
BLOCK = GRID_SIZE // FINE_RES


def _target(boundary):
    minx, maxy, nrows, ncols = grid_spec(boundary)
    transform = from_origin(minx, maxy, FINE_RES, FINE_RES)
    return transform, (nrows * BLOCK, ncols * BLOCK)


def _write_cache(path, array, transform, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                   dtype=array.dtype, crs=CRS, transform=transform, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def _read_cache(path):
    with rasterio.open(path) as src:
        return src.read(1)


def dem_100m(boundary) -> np.ndarray:
    cache = PROCESSED_DIR / "dem_100m.tif"
    if cache.exists():
        return _read_cache(cache)

    west, south, east, north = boundary.to_crs("EPSG:4326").total_bounds
    tiles = []
    for lat in range(int(np.floor(south)), int(np.floor(north)) + 1):
        for lon in range(int(np.floor(west)), int(np.floor(east)) + 1):
            url = DEM_URL.format(ns="N", lat=lat, ew="W", lon=-lon)
            try:
                tiles.append(download(url, RAW_DIR / "dem" / url.rsplit("/", 1)[1]))
            except requests.HTTPError:
                pass  # tiles that are entirely ocean do not exist

    mosaic, src_transform = merge([str(t) for t in tiles])
    transform, shape = _target(boundary)
    dem = np.full(shape, np.nan, dtype="float32")
    reproject(mosaic[0].astype("float32"), dem, src_transform=src_transform, src_crs="EPSG:4326",
              dst_transform=transform, dst_crs=CRS, resampling=Resampling.bilinear, dst_nodata=np.nan)
    _write_cache(cache, dem, transform, np.nan)
    return dem


def landcover_100m(boundary) -> np.ndarray:
    cache = PROCESSED_DIR / "worldcover_100m.tif"
    if cache.exists():
        return _read_cache(cache)

    transform, shape = _target(boundary)
    lc = np.zeros(shape, dtype="uint8")
    west, south, east, north = boundary.to_crs("EPSG:4326").total_bounds
    # WorldCover tiles are 3x3 degrees, named by their south-west corner
    for lat in range(int(np.floor(south / 3) * 3), int(np.floor(north / 3) * 3) + 1, 3):
        for lon in range(int(np.floor(west / 3) * 3), int(np.floor(east / 3) * 3) + 1, 3):
            url = "/vsicurl/" + WORLDCOVER_URL.format(lat=lat, lon=-lon)
            print(f"  reading {url}")
            with rasterio.open(url, overview_level=1) as src:
                part = np.zeros(shape, dtype="uint8")
                reproject(rasterio.band(src, 1), part, dst_transform=transform, dst_crs=CRS,
                          resampling=Resampling.mode, src_nodata=0, dst_nodata=0)
            lc = np.where(lc == 0, part, lc)
    _write_cache(cache, lc, transform, 0)
    return lc


def slope_aspect(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d_row, d_col = np.gradient(dem, FINE_RES)
    dz_east, dz_north = d_col, -d_row
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_north)))
    aspect = (np.degrees(np.arctan2(-dz_east, -dz_north)) + 360) % 360  # compass direction faced
    return slope, aspect


def blocks(a: np.ndarray) -> np.ndarray:
    """Reshape a fine raster to (nrows, ncols, BLOCK*BLOCK): one row of pixels per grid cell."""
    nr, nc = a.shape[0] // BLOCK, a.shape[1] // BLOCK
    return a.reshape(nr, BLOCK, nc, BLOCK).swapaxes(1, 2).reshape(nr, nc, BLOCK * BLOCK)


def build(boundary, grid) -> pd.DataFrame:
    dem = dem_100m(boundary)
    lc = landcover_100m(boundary)
    slope, aspect = slope_aspect(dem)
    south = ((aspect >= 135) & (aspect <= 225) & (slope >= 1)).astype("float32")
    south[np.isnan(slope)] = np.nan
    flat = (slope < FLAT_SLOPE_DEG).astype("float32")
    flat[np.isnan(slope)] = np.nan

    r, c = grid["row"].to_numpy(), grid["col"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells over open water
        out = pd.DataFrame({
            "cell_id": grid["cell_id"].to_numpy(),
            "elev_mean_m": np.nanmean(blocks(dem), axis=2)[r, c],
            "elev_range_m": (np.nanmax(blocks(dem), axis=2) - np.nanmin(blocks(dem), axis=2))[r, c],
            "slope_mean_deg": np.nanmean(blocks(slope), axis=2)[r, c],
            "slope_p90_deg": np.nanpercentile(blocks(slope), 90, axis=2)[r, c],
            "flat_frac": np.nanmean(blocks(flat), axis=2)[r, c],
            "south_facing_frac": np.nanmean(blocks(south), axis=2)[r, c],
        })

    lc_blocks = blocks(lc)[r, c]
    for code, name in WORLDCOVER_CLASSES.items():
        out[f"lc_{name}_frac"] = (lc_blocks == code).mean(axis=1)
    return out.round(4)


def run() -> None:
    save(build(load("state_boundary"), load("grid", schema="features")), "grid_terrain", schema="features")
