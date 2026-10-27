"""Start every SESAS service at once.

    python scripts/run_services.py              # all six
    python scripts/run_services.py equity visualization
    python scripts/run_services.py --list

Each service runs as its own uvicorn process, which is the point: they are independently
deployable and any one of them can be killed without taking the others down. Ctrl+C stops
them all.

The dashboard is at http://localhost:8006/app/ once visualization is up.
"""

import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SERVICES = {
    "irradiance":    8001,
    "terrain":       8002,
    "socioecon":     8003,
    "adoption":      8004,
    "equity":        8005,
    "visualization": 8006,
    "geocoding":     8007,
    "site":          8011,
}

# visualization calls equity, geocoding and site over REST, so they listen before it starts.
ORDER = ["irradiance", "terrain", "socioecon", "adoption",
         "equity", "geocoding", "site", "visualization"]

COLORS = ["\033[36m", "\033[32m", "\033[33m", "\033[35m", "\033[34m", "\033[91m"]
RESET = "\033[0m"


def port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def pump(stream, name: str, color: str) -> None:
    """Prefix every line with its service so six logs in one terminal stay readable."""
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"{color}[{name:<13}]{RESET} {line}")
        sys.stdout.flush()


def main(argv: list[str]) -> int:
    if "--list" in argv:
        print("\n".join(f"{n:<14} {p}" for n, p in SERVICES.items()))
        return 0

    wanted = [a for a in argv if not a.startswith("-")] or ORDER
    unknown = [w for w in wanted if w not in SERVICES]
    if unknown:
        print(f"Unknown service(s): {unknown}. Use --list.", file=sys.stderr)
        return 2
    wanted = [n for n in ORDER if n in wanted]

    busy = [(n, SERVICES[n]) for n in wanted if port_busy(SERVICES[n])]
    if busy:
        print("These ports are already in use:", file=sys.stderr)
        for n, p in busy:
            print(f"  {n} on {p}", file=sys.stderr)
        print("\nStop the existing process, or:  lsof -ti:PORT | xargs kill", file=sys.stderr)
        return 1

    procs = []
    for i, name in enumerate(wanted):
        port = SERVICES[name]
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", f"services.{name}:app", "--port", str(port)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        procs.append((name, proc))
        threading.Thread(target=pump, args=(proc.stdout, name, COLORS[i % len(COLORS)]),
                         daemon=True).start()
        time.sleep(0.4)

    print(f"\n  {len(procs)} services starting. Dashboard: http://localhost:8006/app/")
    print("  Ctrl+C to stop all.\n")

    stopping = threading.Event()

    def shutdown(*_):
        if stopping.is_set():
            return
        stopping.set()
        print("\n  stopping...")
        for _, p in procs:
            p.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while not stopping.is_set():
            for name, p in procs:
                if p.poll() is not None:
                    print(f"\n  {name} exited with code {p.returncode}", file=sys.stderr)
                    shutdown()
                    break
            time.sleep(0.5)
    finally:
        for _, p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
