"""Start every service plus the gateway, one uvicorn process each. Ctrl+C stops them all.

    python scripts/run_services.py          # all services
    python scripts/run_services.py roi yield  # just these

Then open http://127.0.0.1:8000/app/ for the dashboard, or any service on its own port.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVICES = {
    "irradiance": ("services.irradiance", 8001),
    "terrain": ("services.terrain", 8002),
    "socioecon": ("services.socioecon", 8003),
    "adoption": ("services.adoption", 8004),
    "equity": ("services.equity", 8005),
    "visualization": ("services.visualization", 8006),
    "yield": ("services.pv_yield", 8007),
    "roi": ("services.roi", 8008),
    "developability": ("services.developability", 8009),
    "gateway": ("services.gateway", 8000),
}


def main(names: list[str]) -> None:
    unknown = [n for n in names if n not in SERVICES]
    if unknown:
        raise SystemExit(f"Unknown service(s): {unknown}. Choose from {list(SERVICES)}")

    procs = []
    for name in names or SERVICES:
        module, port = SERVICES[name]
        print(f"  {name:15} http://127.0.0.1:{port}")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", f"{module}:app", "--port", str(port), "--log-level", "warning"],
            cwd=ROOT,
        ))
    print("Dashboard: http://127.0.0.1:8000/app/   (Ctrl+C to stop)")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("A service exited; stopping the rest.")
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main(sys.argv[1:])
