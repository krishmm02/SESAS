"""Step 5a: 1 km analysis grid over Maryland land, for utility-scale suitability.

Cells are aligned to whole GRID_SIZE multiples in the projected CRS. Rasters are
resampled onto the same origin at FINE_RES, so each cell is exactly a
(GRID_SIZE / FINE_RES)^2 block of fine pixels at (row, col).

Table: features.grid
"""

import math

import geopandas as gpd
import numpy as np
import shapely

from pipeline.config import CRS, GRID_SIZE
from pipeline.utils import load, save


def grid_spec(boundary: gpd.GeoDataFrame) -> tuple[float, float, int, int]:
    """(minx, maxy, nrows, ncols) of the aligned grid covering the boundary."""
    minx, miny, maxx, maxy = boundary.total_bounds
    minx = math.floor(minx / GRID_SIZE) * GRID_SIZE
    miny = math.floor(miny / GRID_SIZE) * GRID_SIZE
    maxx = math.ceil(maxx / GRID_SIZE) * GRID_SIZE
    maxy = math.ceil(maxy / GRID_SIZE) * GRID_SIZE
    return minx, maxy, int((maxy - miny) / GRID_SIZE), int((maxx - minx) / GRID_SIZE)


def build(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    minx, maxy, nrows, ncols = grid_spec(boundary)
    row, col = np.divmod(np.arange(nrows * ncols), ncols)
    x0 = minx + col * GRID_SIZE
    y1 = maxy - row * GRID_SIZE
    cells = gpd.GeoDataFrame(
        {"cell_id": row * ncols + col, "row": row, "col": col},
        geometry=shapely.box(x0, y1 - GRID_SIZE, x0 + GRID_SIZE, y1),
        crs=CRS,
    )

    land = boundary.union_all()
    cells = cells[cells.intersects(land)].copy()
    cells["land_frac"] = (cells.intersection(land).area / GRID_SIZE**2).round(4)
    return cells[cells["land_frac"] > 0]


def run() -> None:
    save(build(load("state_boundary")), "grid", schema="features")
