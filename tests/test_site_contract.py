"""SolarSiteService: the physics behind the panel planner.

Needs the service on 8011 (and 8006 for the relay test). Every check is a physical
sanity bound, so a units slip or a swapped tilt/azimuth fails loudly rather than
producing a plausible-looking number.

    pytest tests/test_site_contract.py -v
"""

import pytest
import requests

from conftest import _listening

pytestmark = pytest.mark.live

SITE = "http://127.0.0.1:8011"
TIMEOUT = 40
BALTIMORE = {"lat": 39.349, "lon": -76.676}


@pytest.fixture(scope="session")
def site_url():
    if not _listening(SITE):
        pytest.skip(f"SolarSiteService not listening on {SITE}")
    return SITE


def estimate(url, **params):
    r = requests.get(f"{url}/site/estimate", params={**BALTIMORE, **params}, timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    return r.json()


def test_rooftop_yield_is_physically_plausible(site_url):
    """NREL PVWatts gives roughly 1,200-1,350 kWh per kW per year for a south-facing 20 deg
    roof in Baltimore. Far outside that means a units or transposition bug."""
    y = requests.get(f"{site_url}/site/potential", params=BALTIMORE, timeout=TIMEOUT).json()
    assert 1150 < y["specific_yield_kwh_kw"]["roof"] < 1400
    assert 3.5 < y["ghi_kwh_m2_day"] < 4.6


def test_ground_racks_beat_flush_roof_and_north_loses(site_url):
    """A 30 deg rack runs cooler and nearer the optimal tilt than a 20 deg flush roof; a
    north-facing roof at 39 N loses roughly a quarter of its output."""
    roof = estimate(site_url, panels=10, mount="roof")["annual_kwh"]
    ground = estimate(site_url, panels=10, mount="ground")["annual_kwh"]
    north = estimate(site_url, panels=10, mount="roof", azimuth=0)["annual_kwh"]
    assert ground > roof
    assert north < 0.85 * roof


def test_output_scales_with_panel_count(site_url):
    one, two = estimate(site_url, panels=10), estimate(site_url, panels=20)
    assert abs(two["annual_kwh"] - 2 * one["annual_kwh"]) <= 2
    assert one["system_kw"] == 4.0, "10 x 400 W modules must be 4.0 kW"
    assert one["capacity_factor"] == two["capacity_factor"]
    assert 0.12 < one["capacity_factor"] < 0.18, "rooftop PV in Maryland runs ~14-15%"


def test_energy_only_no_money(site_url):
    """SolarSiteService reports energy; prices and payback live in ROIService (/roi/layout)."""
    body = estimate(site_url, panels=10)
    assert not {"annual_savings", "net_cost", "simple_payback_years"} & body.keys()


ROI = "http://127.0.0.1:8008"


def test_roi_prices_a_layout_from_site_energy(site_url):
    """/roi/layout must price exactly the kWh SolarSiteService reports, and payback must not
    depend on array size (cost and savings both scale linearly with panels)."""
    if not _listening(ROI):
        pytest.skip(f"ROIService not listening on {ROI}")
    params = {**BALTIMORE, "panels": 10, "price_per_kwh": 0.20, "cost_per_w": 2.5}
    small = requests.get(f"{ROI}/roi/layout", params=params, timeout=TIMEOUT).json()
    big = requests.get(f"{ROI}/roi/layout", params={**params, "panels": 40}, timeout=TIMEOUT).json()
    assert small["annual_kwh"] == estimate(site_url, panels=10)["annual_kwh"]
    assert small["annual_savings"] == round(small["annual_kwh"] * 0.20)
    assert small["net_cost"] == 10 * 400 * 2.5
    assert abs(small["simple_payback_years"] - big["simple_payback_years"]) <= 0.1


def test_point_outside_maryland_is_rejected(site_url):
    r = requests.get(f"{site_url}/site/estimate", params={"lat": 40.7, "lon": -74.0, "panels": 5},
                     timeout=TIMEOUT)
    assert r.status_code == 422


def test_dashboard_reaches_it_through_visualization(viz_url, site_url):
    """The planner page only talks to its own origin, so the relay is part of the contract."""
    body = requests.get(f"{viz_url}/site/estimate", params={**BALTIMORE, "panels": 4},
                        timeout=TIMEOUT).json()
    assert body["panels"] == 4 and body["annual_kwh"] > 0
