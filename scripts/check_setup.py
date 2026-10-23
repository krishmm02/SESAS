"""Verify the local environment: libraries, PostGIS connection, API keys.

Run from the project root:  python scripts/check_setup.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

results = []


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def warn(name, detail):
    print(f"[WARN] {name} - {detail}")


# --- libraries ---------------------------------------------------------------
def check_import(name, version_of=lambda m: getattr(m, "__version__", "?")):
    try:
        module = __import__(name, fromlist=["_"])
        report(f"import {name}", True, version_of(module))
    except Exception as e:
        report(f"import {name}", False, str(e))


check_import("geopandas")
check_import("rasterio")
check_import("osgeo.gdal", version_of=lambda m: m.__version__)
check_import("xgboost")
check_import("shap")

# --- PostGIS -----------------------------------------------------------------
try:
    from sqlalchemy import text

    from pipeline.db import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT postgis_version()")).scalar()
    report("PostGIS connection", True, f"{engine.url.host}: postgis_version() = {version}")
except Exception as e:
    report("PostGIS connection", False, str(e).splitlines()[0])

# --- API keys (warn only) ----------------------------------------------------
for key in ("CENSUS_API_KEY", "NREL_API_KEY"):
    value = os.getenv(key, "")
    if not value:
        warn(key, "not set in .env")
    elif value.startswith("your_"):
        warn(key, "still a placeholder; replace it in .env")
    else:
        print(f"[PASS] {key} - set")

# --- summary -----------------------------------------------------------------
print()
if all(results):
    print("All checks passed.")
else:
    print(f"{results.count(False)} check(s) failed.")
    sys.exit(1)
