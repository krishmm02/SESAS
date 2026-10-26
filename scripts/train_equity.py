"""Train the two-stage equity gap model and write scores.tract_equity.

Requires the pipeline to have produced features.tract_features:

    python scripts/run_pipeline.py
    python scripts/train_equity.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import equity_gap  # noqa: E402

if __name__ == "__main__":
    print("[equity_gap]")
    start = time.time()
    equity_gap.run()
    print(f"  done in {time.time() - start:.0f}s")
