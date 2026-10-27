"""Run data pipeline steps in order. Each step reads its inputs from PostGIS.

    python scripts/run_pipeline.py              # all steps
    python scripts/run_pipeline.py osm terrain  # just these
    python scripts/run_pipeline.py --list
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import census, deepsolar, features, grid, irradiance, lead, osm, pv_yield, terrain, uspvdb  # noqa: E402

STEPS = {
    "boundaries": census.run_boundaries,
    "acs": census.run_acs,
    "lead": lead.run,
    "deepsolar": deepsolar.run,
    "irradiance": irradiance.run,
    "tract_irradiance": irradiance.run_tracts,
    "grid": grid.run,
    "terrain": terrain.run,
    "tract_terrain": terrain.run_tracts,
    "tract_yield": pv_yield.run_tracts,
    "osm": osm.run,
    "uspvdb": uspvdb.run,
    "tract_features": features.run_tracts,
    "grid_features": features.run_grid,
}


def main(args: list[str]) -> None:
    if "--list" in args:
        print("\n".join(STEPS))
        return
    unknown = [a for a in args if a not in STEPS]
    if unknown:
        raise SystemExit(f"Unknown step(s): {unknown}. Use --list.")

    for name in args or STEPS:
        print(f"[{name}]")
        start = time.time()
        STEPS[name]()
        print(f"  done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
