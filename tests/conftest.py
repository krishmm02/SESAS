"""Shared fixtures and markers.

These are integration tests by design. The services are thin enough that mocking the
database would test SQLAlchemy and FastAPI rather than this project, and the bugs worth
guarding against here are semantic (a ranking that inverts, an expectation that leaks
location) rather than logical. So the tests run against the real scored table and real
running services.

    pytest -v                      # everything
    pytest -v -m "not live"        # model invariants only, no services needed
    pytest -v -m live              # API contracts, needs services on 8005/8006
"""

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EQUITY = "http://127.0.0.1:8005"
VIZ = "http://127.0.0.1:8006"


def _listening(url: str) -> bool:
    host, port = url.rsplit(":", 1)
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex((host.split("//")[1], int(port))) == 0


def pytest_configure(config):
    config.addinivalue_line("markers", "live: needs running services")


@pytest.fixture(scope="session")
def scores():
    """scores.tract_equity as a DataFrame. Skips if the model has not been trained."""
    from pipeline.utils import load
    try:
        df = load("tract_equity", schema="scores")
    except Exception as e:
        pytest.skip(f"scores.tract_equity unavailable ({e.__class__.__name__}) - "
                    "run scripts/train_equity.py")
    if df.empty:
        pytest.skip("scores.tract_equity is empty")
    return df


@pytest.fixture(scope="session")
def equity_url():
    if not _listening(EQUITY):
        pytest.skip(f"EquityGapService not listening on {EQUITY}")
    return EQUITY


@pytest.fixture(scope="session")
def viz_url():
    if not _listening(VIZ):
        pytest.skip(f"VisualizationService not listening on {VIZ}")
    return VIZ
