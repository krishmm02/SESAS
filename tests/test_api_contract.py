"""API contract tests against running services.

Doubles as the pre-demo smoke test: if this passes, the dashboard will work.

    python scripts/run_services.py equity visualization
    pytest tests/test_api_contract.py -v
"""

import pytest
import requests

pytestmark = pytest.mark.live

TIMEOUT = 30
A_TRACT = "24510190200"      # Baltimore City
A_COUNTY = "24510"


# --------------------------------------------------------------------------- health

def test_equity_health(equity_url):
    assert requests.get(f"{equity_url}/health", timeout=TIMEOUT).json()["status"] == "ok"


def test_visualization_reports_its_upstream(viz_url):
    """VisualizationService is the one service that calls another over REST, so its
    health has to attribute a failure to the right side of that boundary."""
    body = requests.get(f"{viz_url}/health", timeout=TIMEOUT).json()
    assert body["status"] == "ok"
    assert body["upstream"]["equity"] == "ok", \
        f"upstream equity reported {body['upstream']['equity']}"


# --------------------------------------------------------------------------- equity

def test_tract_response_carries_both_components(equity_url):
    """The gap alone cannot distinguish real untapped potential from a poor expectation,
    so expected and actual must always ship alongside it."""
    body = requests.get(f"{equity_url}/equity/{A_TRACT}", timeout=TIMEOUT).json()
    for field in ("geoid", "expected_adoption", "actual_adoption", "gap",
                  "gap_percentile", "priority_tier", "barrier_1", "barrier_1_shap"):
        assert field in body, f"missing {field}"
    assert abs(body["expected_adoption"] - body["actual_adoption"] - body["gap"]) < 0.01


def test_unknown_tract_is_404(equity_url):
    r = requests.get(f"{equity_url}/equity/24999999999", timeout=TIMEOUT)
    assert r.status_code == 404


@pytest.mark.parametrize("bad", ["abc", "123", "9910070100", "24510"])
def test_malformed_geoid_is_422(equity_url, bad):
    r = requests.get(f"{equity_url}/equity/{bad}", timeout=TIMEOUT)
    assert r.status_code == 422, f"{bad!r} should fail route validation, got {r.status_code}"


def test_county_filter_narrows_results(equity_url):
    all_rows = requests.get(f"{equity_url}/equity", timeout=TIMEOUT).json()
    one = requests.get(f"{equity_url}/equity", params={"county": A_COUNTY}, timeout=TIMEOUT).json()
    assert 0 < len(one) < len(all_rows)
    assert all(r["geoid"].startswith(A_COUNTY) for r in one)


def test_summary_has_no_none_key(equity_url):
    """Null tiers used to stringify into a literal "None" key in tier_counts."""
    body = requests.get(f"{equity_url}/equity/summary",
                        params={"county": A_COUNTY}, timeout=TIMEOUT).json()
    assert "None" not in body["tier_counts"]
    assert set(body["tier_counts"]) <= {"HIGH", "MEDIUM", "LOW"}
    assert isinstance(body["unscored"], int)


def test_metrics_report_both_cv_schemes(equity_url):
    """Reporting only the random split would have shown R2 +0.30 where the honest
    county-grouped figure is negative."""
    m = requests.get(f"{equity_url}/equity/model/metrics", timeout=TIMEOUT).json()
    for scheme in ("county", "random"):
        assert scheme in m["stage1"]["ridge"], f"stage 1 missing {scheme} CV"
        assert scheme in m["stage2"]["xgboost"], f"stage 2 missing {scheme} CV"
    assert m["stage1"]["selected"].startswith("none"), \
        "stage 1 must not be scoring while it shows no skill"


# --------------------------------------------------------------------------- geojson

def test_geojson_is_renderable(viz_url):
    gj = requests.get(f"{viz_url}/geojson/equity",
                      params={"county": A_COUNTY}, timeout=TIMEOUT).json()
    assert gj["type"] == "FeatureCollection"
    assert gj["features"], "no features returned"

    f = gj["features"][0]
    assert f["geometry"]["type"] in ("Polygon", "MultiPolygon")
    assert "priority_tier" in f["properties"], "the map colours by priority_tier"

    # Coordinates must be WGS84 degrees. The table stores EPSG:26985 metres, so a missing
    # ST_Transform ships numbers in the hundreds of thousands and Leaflet renders nothing.
    lon, lat = f["geometry"]["coordinates"][0][0][:2]
    assert -80 < lon < -74, f"longitude {lon} is not in Maryland - is ST_Transform applied?"
    assert 37 < lat < 40, f"latitude {lat} is not in Maryland"


def test_geojson_matches_the_api_it_composes(viz_url, equity_url):
    """The map must render exactly what the API serves - no second code path."""
    gj = requests.get(f"{viz_url}/geojson/equity",
                      params={"county": A_COUNTY}, timeout=TIMEOUT).json()
    api = requests.get(f"{equity_url}/equity",
                       params={"county": A_COUNTY}, timeout=TIMEOUT).json()
    by_geoid = {r["geoid"]: r for r in api}
    for feat in gj["features"][:25]:
        src = by_geoid[feat["id"]]
        assert feat["properties"]["gap"] == src["gap"]
        assert feat["properties"]["priority_tier"] == src["priority_tier"]


def test_dashboard_is_served(viz_url):
    r = requests.get(f"{viz_url}/app/", timeout=TIMEOUT)
    assert r.status_code == 200
    assert "Solar Equity Gap" in r.text
