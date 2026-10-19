"""Download and database I/O helpers shared by pipeline steps."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from sqlalchemy import text

from pipeline.db import get_engine

USER_AGENT = "solar-equity-course-project/0.1 (UMBC CMSC 668)"


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Stream url to dest unless it already exists."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def save(df: pd.DataFrame, table: str, schema: str = "raw") -> None:
    """Replace schema.table with df (GeoDataFrames keep their geometry)."""
    engine = get_engine()
    if isinstance(df, gpd.GeoDataFrame):
        df.to_postgis(table, engine, schema=schema, if_exists="replace", index=False)
    else:
        df.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    print(f"  wrote {schema}.{table}: {len(df):,} rows")


def load(table: str, schema: str = "raw") -> pd.DataFrame:
    """Read schema.table; returns a GeoDataFrame when it has a geometry column."""
    engine = get_engine()
    with engine.connect() as conn:
        has_geom = conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = :s AND table_name = :t AND column_name = 'geometry'"
            ),
            {"s": schema, "t": table},
        ).first()
    sql = f'SELECT * FROM "{schema}"."{table}"'
    if has_geom:
        return gpd.read_postgis(sql, engine, geom_col="geometry")
    return pd.read_sql(sql, engine)
