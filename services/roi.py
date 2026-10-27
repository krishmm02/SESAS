"""ROIService: cost, savings and payback - per census tract, and for a panel-planner layout.

    uvicorn services.roi:app --port 8008

Composes services over REST rather than reading PostGIS. Per tract: annual kWh from
YieldService (8010) and the tract's average household electricity bill from
SocioEconService (8003, NREL LEAD). Per layout (/roi/layout): the layout's kWh from
SolarSiteService (8011). Every default below can be overridden per request.

Without ?kw=, the system is sized to cover the tract's average household use
(bill / price), the way installers size residential systems. Savings are capped at that
use: output beyond it earns well below retail under Maryland net metering, so it is
counted as zero.

Defaults, checked 2026-10:
- price_per_kwh  0.2177  EIA Maryland residential average, May 2026
- cost_per_w     2.59    EnergySage Maryland installed price, Aug 2026, before incentives
- tax_credit     0.0     the 30% federal homeowner credit (IRC 25D) ended 2025-12-31
Conservative by design: no rate escalation, financing or Maryland SRECs. LEAD bills are
2022 dollars, so household use is understated now that prices have risen.
"""

import os

import requests
from fastapi import FastAPI, HTTPException, Path, Query

from services.common import COUNTY, GEOID

app = FastAPI(title="ROIService")

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and stalls ~2 s per call.
YIELD_URL = os.getenv("YIELD_SERVICE_URL", "http://127.0.0.1:8010")
SOCIOECON_URL = os.getenv("SOCIOECON_SERVICE_URL", "http://127.0.0.1:8003")
SITE_URL = os.getenv("SITE_SERVICE_URL", "http://127.0.0.1:8011")
TIMEOUT = 20
LIFETIME_YEARS = 25
DEGRADATION = 0.005  # output lost per year


def fetch(base: str, path: str, params: dict | None = None):
    try:
        r = requests.get(f"{base}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        raise HTTPException(e.response.status_code, e.response.json().get("detail", str(e))) from e
    except requests.RequestException as e:
        raise HTTPException(503, f"upstream unavailable at {base}: {e}") from e


def roi(y: dict, s: dict | None, kw: float | None, cost_per_w: float, price: float, credit: float) -> dict:
    """y is a 1 kW YieldService row, s a SocioEconService row (None if the tract has none)."""
    bill = (s or {}).get("avg_electricity_cost")
    income = (s or {}).get("median_household_income")
    use_kwh = bill / price if bill else None
    if kw is None:
        if not use_kwh:
            return {"geoid": y["geoid"], "error": "no household bill data; pass ?kw= to size the system"}
        kw = round(use_kwh / y["specific_yield_kwh_kw"], 1)

    kwh = y["specific_yield_kwh_kw"] * kw
    gross = kw * 1000 * cost_per_w
    net = gross * (1 - credit)
    savings = min(kwh, use_kwh or kwh) * price
    lifetime = sum(savings * (1 - DEGRADATION) ** t for t in range(LIFETIME_YEARS))
    return {
        "geoid": y["geoid"],
        "system_kw": kw,
        "annual_kwh": round(kwh),
        "household_use_kwh": round(use_kwh) if use_kwh else None,
        "gross_cost": round(gross),
        "net_cost": round(net),
        "annual_savings": round(savings),
        "simple_payback_years": round(net / savings, 1),
        "lifetime_net_savings": round(lifetime - net),
        "avg_household_electricity_cost": bill,
        "bill_offset_share": round(savings / bill, 3) if bill else None,
        # Payback barely varies statewide (sunlight is uniform); what varies is whether a
        # household can front the cost. This is the equity-relevant number.
        "net_cost_share_of_income": round(net / income, 3) if income else None,
    }


KW = Query(None, gt=0, le=1000, description="system size in kW(dc); default sizes to household use")
COST = Query(2.59, gt=0, description="installed $/W before incentives")
PRICE = Query(0.2177, gt=0, description="electricity price, $/kWh")
CREDIT = Query(0.0, ge=0, lt=1, description="share of cost covered by incentives")


@app.get("/health")
def health() -> dict:
    """Reports both upstreams, so a failure is attributable to the right service."""
    upstream = {}
    for name, url in (("yield", YIELD_URL), ("socioecon", SOCIOECON_URL), ("site", SITE_URL)):
        try:
            upstream[name] = requests.get(f"{url}/health", timeout=5).json().get("status", "unknown")
        except requests.RequestException as e:
            upstream[name] = f"unreachable: {e.__class__.__name__}"
    return {"status": "ok", "upstream": upstream}


# Declared before /roi/{geoid}: FastAPI would otherwise match "layout" as a GEOID and 422.
@app.get("/roi/layout")
def layout_roi(
    lat: float = Query(ge=37.88, le=39.73), lon: float = Query(ge=-79.49, le=-75.04),
    panels: int = Query(ge=1, le=100000), mount: str = Query("roof", pattern="^(roof|flat|ground)$"),
    azimuth: float = Query(180, ge=0, lt=360),
    cost_per_w: float = COST, price_per_kwh: float = PRICE, tax_credit: float = CREDIT,
) -> dict:
    """Cost, savings and payback of a panel-planner layout, rooftop or open land.

    Energy comes from SolarSiteService over REST; this service only adds the money, which
    is why the midterm planner can run without it. Every kWh is valued at price_per_kwh:
    there is no household here to cap savings at, unlike /roi/{geoid}.
    """
    e = fetch(SITE_URL, "/site/estimate", {"lat": lat, "lon": lon, "panels": panels,
                                           "mount": mount, "azimuth": azimuth})
    savings = e["annual_kwh"] * price_per_kwh
    gross = e["system_kw"] * 1000 * cost_per_w
    net = gross * (1 - tax_credit)
    lifetime = sum(savings * (1 - DEGRADATION) ** t for t in range(LIFETIME_YEARS))
    return {
        **e,
        "annual_savings": round(savings), "gross_cost": round(gross), "net_cost": round(net),
        "simple_payback_years": round(net / savings, 1),
        "lifetime_net_savings": round(lifetime - net), "lifetime_years": LIFETIME_YEARS,
        "assumptions": {**e["assumptions"], "price_per_kwh": price_per_kwh, "cost_per_w": cost_per_w,
                        "tax_credit": tax_credit, "degradation_per_year": DEGRADATION},
    }


@app.get("/roi")
def list_roi(county: str = Query(..., pattern=COUNTY, description="e.g. 24510"), kw: float | None = KW,
             cost_per_w: float = COST, price_per_kwh: float = PRICE, tax_credit: float = CREDIT) -> list[dict]:
    """One county per call; county is required to keep upstream payloads small."""
    yields = fetch(YIELD_URL, "/yield", {"county": county})
    socio = {r["geoid"]: r for r in fetch(SOCIOECON_URL, "/socioecon", {"county": county})}
    return [roi(y, socio.get(y["geoid"]), kw, cost_per_w, price_per_kwh, tax_credit) for y in yields]


@app.get("/roi/{geoid}")
def get_roi(geoid: str = Path(pattern=GEOID, description="e.g. 24510070100"), kw: float | None = KW,
            cost_per_w: float = COST, price_per_kwh: float = PRICE, tax_credit: float = CREDIT) -> dict:
    y = fetch(YIELD_URL, f"/yield/{geoid}")
    s = fetch(SOCIOECON_URL, f"/socioecon/{geoid}")
    return roi(y, s, kw, cost_per_w, price_per_kwh, tax_credit)
