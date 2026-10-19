"""Shared constants for the data pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

STATE_FIPS = "24"
STATE_ABBR = "MD"

# ACS 2018-2022 5-year: same vintage as NREL LEAD 2022, both on 2020 tract boundaries.
ACS_YEAR = 2022

# NAD83 / Maryland state plane, meters. Every table is stored in this CRS.
CRS = "EPSG:26985"

GRID_SIZE = 1000  # utility-scale grid cell edge, meters
FINE_RES = 100    # intermediate raster resolution, meters (GRID_SIZE must be a multiple)

# lon/lat box around Maryland with margin, for sources queried by bounding box
BBOX = {"west": -80.0, "south": 37.5, "east": -74.5, "north": 40.0}
