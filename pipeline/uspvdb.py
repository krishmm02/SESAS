"""Step 6 (data side): USGS/LBNL U.S. Large-Scale Solar PV Database facility polygons.

Keeps every facility inside the Maryland bounding box (including neighbors near the
border), since grid labels are made by spatial overlap.

Table: raw.uspvdb
"""

import zipfile

import geopandas as gpd

from pipeline.config import BBOX, CRS, RAW_DIR
from pipeline.utils import download, save

USPVDB_URL = "https://eerscmap.usgs.gov/uspvdb/assets/data/uspvdbGeoJSON.zip"
COLUMNS = ["case_id", "eia_id", "p_name", "p_state", "p_county", "p_year",
           "p_cap_ac", "p_cap_dc", "p_area", "p_type", "p_axis", "p_agrivolt", "geometry"]


def build() -> gpd.GeoDataFrame:
    path = download(USPVDB_URL, RAW_DIR / "uspvdb" / "uspvdbGeoJSON.zip")
    with zipfile.ZipFile(path) as z:
        geojson = next(n for n in z.namelist() if n.endswith(".geojson"))
    pv = gpd.read_file(f"zip://{path}!{geojson}",
                       bbox=(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]))
    return pv[COLUMNS].to_crs(CRS)


def run() -> None:
    save(build(), "uspvdb")
