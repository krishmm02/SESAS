"""Database helper: load .env and return a SQLAlchemy engine for PostGIS.

DATABASE_URL (e.g. a Supabase connection string) takes precedence over the
POSTGRES_* variables used by the local Docker database.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine, make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_engine(echo: bool = False) -> Engine:
    if os.getenv("DATABASE_URL"):
        # Supabase gives "postgresql://"; pin the driver we install (psycopg2).
        url = make_url(os.environ["DATABASE_URL"]).set(drivername="postgresql+psycopg2")
        # pre_ping replaces connections the pooler has closed while a service sat idle
        return create_engine(url, echo=echo, pool_pre_ping=True)
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB"),
    )
    return create_engine(url, echo=echo)
