"""Train the utility-scale developability model and write scores.grid_developability.

Requires the pipeline to have produced features.grid_features:

    python scripts/run_pipeline.py
    python scripts/train_developability.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import developability  # noqa: E402

if __name__ == "__main__":
    print("[developability]")
    start = time.time()
    developability.run()
    print(f"  done in {time.time() - start:.0f}s")
