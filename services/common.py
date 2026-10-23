"""Shared database access for the per-tract services."""

from fastapi import HTTPException
from sqlalchemy import text

from pipeline.db import get_engine

engine = get_engine()

GEOID = r"^24\d{9}$"   # 11-digit Maryland tract GEOID
COUNTY = r"^24\d{3}$"  # 5-digit Maryland county FIPS


def tract_rows(table: str, county: str | None = None) -> list[dict]:
    """All rows of features.<table>, optionally only one county's tracts."""
    sql = f'SELECT * FROM features."{table}"'
    params = {}
    if county:
        sql += " WHERE geoid LIKE :prefix"
        params["prefix"] = county + "%"
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql + " ORDER BY geoid"), params).mappings()]


def tract_row(table: str, geoid: str) -> dict:
    """One tract's row of features.<table>, or 404."""
    with engine.connect() as conn:
        row = conn.execute(text(f'SELECT * FROM features."{table}" WHERE geoid = :g'),
                           {"g": geoid}).mappings().first()
    if row is None:
        raise HTTPException(404, f"tract {geoid} not found")
    return dict(row)


def ping() -> dict:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}
