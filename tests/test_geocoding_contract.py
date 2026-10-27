"""GeocodingService and the address -> tract -> score chain.

Needs services on 8005, 8006 and 8007, and reaches the public Census Geocoder, so these
are the slowest tests in the suite and the ones that can fail for reasons outside this
repo. Run them before a demo; skip them in a hurry with -k "not geocod".

    pytest tests/test_geocoding_contract.py -v
"""

import pytest
import requests

from conftest import _listening

pytestmark = pytest.mark.live

GEOCODE = "http://127.0.0.1:8007"
TIMEOUT = 40

# Verified against the live Census geocoder. Its address file is incomplete, so a demo
# must use addresses known to resolve rather than whatever comes to mind on stage.
KNOWN_GOOD = {
    "100 N Charles St, Baltimore, MD": "24510040100",
    "1000 Hilltop Circle, Baltimore, MD 21250": "24005492500",   # UMBC
}


@pytest.fixture(scope="session")
def geocode_url():
    if not _listening(GEOCODE):
        pytest.skip(f"GeocodingService not listening on {GEOCODE}")
    return GEOCODE


def test_health_declares_vintage(geocode_url):
    """A 2010-vintage lookup returns GEOIDs that silently fail to join our 2020 tables,
    so the vintage is part of the contract."""
    body = requests.get(f"{geocode_url}/health", timeout=TIMEOUT).json()
    assert body["status"] == "ok"
    assert body["vintage"] == "Census2020_Current"


@pytest.mark.parametrize("address,geoid", KNOWN_GOOD.items(), ids=lambda v: v[:28])
def test_known_addresses_resolve(geocode_url, address, geoid):
    body = requests.get(f"{geocode_url}/geocode",
                        params={"address": address}, timeout=TIMEOUT).json()
    assert body["geoid"] == geoid
    assert 37 < body["lat"] < 40 and -80 < body["lon"] < -74


def test_out_of_state_is_rejected_with_a_reason(geocode_url):
    """Downstream services validate ^24\\d{9}$, so an out-of-state hit would otherwise
    surface as a confusing 422 two services later."""
    r = requests.get(f"{geocode_url}/geocode",
                     params={"address": "1600 Pennsylvania Ave NW, Washington DC"},
                     timeout=TIMEOUT)
    assert r.status_code == 422
    assert "Maryland" in r.json()["detail"]


def test_unmatched_address_is_404(geocode_url):
    r = requests.get(f"{geocode_url}/geocode",
                     params={"address": "asdfghjkl qwertyuiop 99999"}, timeout=TIMEOUT)
    assert r.status_code == 404


def test_reverse_geocode(geocode_url):
    body = requests.get(f"{geocode_url}/geocode/reverse",
                        params={"lat": 39.2915, "lon": -76.6153}, timeout=TIMEOUT).json()
    assert body["geoid"].startswith("24510")


def test_coordinates_outside_maryland_rejected_by_route(geocode_url):
    r = requests.get(f"{geocode_url}/geocode/reverse",
                     params={"lat": 51.5, "lon": -0.12}, timeout=TIMEOUT)
    assert r.status_code == 422


# --------------------------------------------------------------- the composed chain

def test_lookup_chains_geocoding_to_equity(viz_url):
    """One call takes a street address to a scored tract across two upstreams."""
    body = requests.get(f"{viz_url}/lookup",
                        params={"address": "100 N Charles St, Baltimore, MD"},
                        timeout=TIMEOUT).json()
    assert body["location"]["geoid"] == "24510040100"
    eq = body["equity"]
    assert eq["geoid"] == body["location"]["geoid"], "chain joined the wrong tract"
    assert abs(eq["expected_adoption"] - eq["actual_adoption"] - eq["gap"]) < 0.01


def test_lookup_passes_upstream_errors_through(viz_url):
    """'Outside Maryland' and 'no match' need different fixes from the user, so the
    orchestrator must not flatten them into one 500."""
    r = requests.get(f"{viz_url}/lookup",
                     params={"address": "1600 Pennsylvania Ave NW, Washington DC"},
                     timeout=TIMEOUT)
    assert r.status_code == 422
    assert "Maryland" in r.json()["detail"]


def test_health_reports_both_upstreams(viz_url):
    up = requests.get(f"{viz_url}/health", timeout=TIMEOUT).json()["upstream"]
    assert up["equity"] == "ok" and up["geocoding"] == "ok"
