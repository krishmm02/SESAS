# SESAS — Project Roadmap & Working Context

Solar Equity Suitability as a Service · CMSC 668 (Service-Oriented Computing), UMBC · Fall 2026

This file is the single source of truth for project direction. It records decisions that are
**already settled** so they don't get re-argued, and tracks what is built vs. outstanding.
Update the status table as units land.

---

## 1. What this is

A set of FastAPI microservices that answer two questions for Maryland, together:

1. **Where does solar physically work?** — irradiance, terrain, land cover
2. **Which communities are underserved relative to that potential?** — the equity gap

The second question is the contribution. Tools like Google Project Sunroof and NREL PVWatts
already answer the first. Nothing public answers the second at neighbourhood scale from open data.

**Team**
- **Krish Mehta** — data pipeline and preprocessing (`pipeline/`)
- **Rahul Badwaik** — ML models and SOA services (`services/`, `models/`)

---

## 2. Locked decisions — do not re-litigate

### 2.1 The physics / ML boundary

Energy yield from irradiance is a **closed-form equation**, not a prediction problem:

```
E = P_rated × (H_annual / 1000) × PR        PR ≈ 0.75–0.85
```

Training a model whose label derives from irradiance just reproduces this formula at expense.
A reviewer will catch it. ML is used **only** where no equation exists:

| Component | Kind | Why |
|---|---|---|
| Yield / specific yield | **Physics** — pvlib or PVWatts | Deterministic |
| Rooftop suitability | **ML** | Land cover + terrain + built form interact |
| Terrain developability | **ML** | Grid access, land use, zoning — no formula |
| Equity gap | **ML** | Human and economic behaviour |

State this split explicitly in every presentation. It pre-empts "isn't this just PVWatts?"

### 2.2 Grain

Two grains only. **No per-building modelling** — it was considered and rejected as out of scope.

| Grain | Count (MD) | Used by |
|---|---|---|
| Census tract | ~1,460 | Rooftop suitability, equity gap |
| Grid cell | TBD (~1–4 km) | Terrain / utility-scale suitability |

Join key across all tabular sources is the 11-digit **GEOID**:

```
24510070100
││ │  └─ tract  070100
││ └──── county 510 (Baltimore City)
└┴────── state  24  (Maryland)
```

Rasters have no GEOID — use **zonal statistics** (overlay polygon, aggregate pixels inside).

### 2.3 Only three datasets are labels

Everything else is a feature column or a direct lookup. This is where project risk concentrates.

| Role | Datasets |
|---|---|
| **A — Labels (`y`)** | USPVDB, DeepSolar, (optionally LBNL Tracking the Sun) |
| **B — Features (`X`)** | NASA POWER, Copernicus DEM, ESA WorldCover, Census ACS, NREL LEAD, OpenStreetMap |
| **C — Consumed as-is** | PVWatts / pvlib, EIA rates, Census TIGER geometry |

### 2.4 Equity gap is a two-stage counterfactual model

Naively fitting `adoption ~ f(income, renter_ratio, …)` and taking the residual **launders the
inequity into the baseline** — a poor tract with low adoption yields zero residual because the
model correctly "expected" low adoption. Do not do this.

**Stage 1** — expected adoption from the *physical solar resource only*. Tenure and housing
type belong in stage 2, not here:

```python
POTENTIAL = ["ghi", "clearness_index", "south_facing_frac", "slope_mean_deg"]
```

An earlier version also put `owner_rate` and `single_family_detached_rate` here, on the reasoning
that a renter cannot install panels so tenure is physical capacity. **That inverted the entire
result.** Baltimore City (owner_rate 0.48, detached 0.14) drew an expectation of 1.37 against an
actual of 1.06, so its gap was ~0 and only 2% of its tracts came out HIGH — against 22% in
Howard County, the wealthiest in the state. Tenure and housing type *are* the structural barrier;
conditioning on them launders the inequity into the baseline.

**Stage 1 does not score.** Across one state these features barely vary (GHI coefficient of
variation 0.017), and neither Ridge nor XGBoost shows county-CV skill (ρ 0.017 and 0.113, the
latter only by reading location off that 1.7% range). Scoring with a skill-less fit propagates
noise: the fitted ridge produced county median expectations spanning **2.20 to 14.75** and sent
Somerset, Wicomico and Dorchester to 100% HIGH priority on fractionally more sun. The expectation
is therefore the **statewide median** (3.16), with the fitted models reported only to evidence
the absence of skill.

```python
baseline = df[TARGET].median()        # median, not mean - target skew is 5.26
df["gap"] = baseline - df["actual_adoption"]
```

**Stage 2** — explain the gap with every structural factor stage 1 was not allowed to see:

```python
BARRIERS = ["owner_rate", "single_family_detached_rate", "median_year_built",
            "pop_density_km2", "median_household_income", "median_home_value",
            "poverty_rate", "energy_burden", "low_income_share", "renter_share",
            "pct_minority"]
```

Rank attributions by **signed** SHAP, not magnitude. `argmax(|shap|)` named whichever factor moved
the gap in either direction, so 98 of 364 HIGH tracts had a "primary barrier" that actually shrank
their gap. Only positive contributions widen a gap. Three factors are stored rather than one
because the barriers are heavily collinear — a tree splits on one of a correlated group and starves
the rest, which is why a single label read `pct_minority` in 77% of tracts.

A positive gap means: *this tract has the sunlight to support more solar than it has.*
Methodologically this mirrors Sunter et al. (Nature Sustainability, 2019) — cite it.

**Always return `expected_adoption` and `actual_adoption` alongside the gap**, never the
difference alone. A planner who cannot see both numbers cannot judge whether a high score means
real untapped potential or a bad expectation estimate.

### 2.5 Spatial cross-validation is mandatory

Neighbouring tracts are highly correlated. A random split leaks them across folds and reports
inflated metrics that mean nothing.

```python
from sklearn.model_selection import GroupKFold
groups = df["geoid"].str[:5]          # county FIPS
cv = GroupKFold(n_splits=5)
```

Expect **R² ≈ 0.4–0.6** on the tract model. Report that number and explain why it is lower than a
random split would show — the explanation earns more credit than the number costs.

### 2.6 Model choice

With ~1,460 tract rows, a default `max_depth=6` XGBoost will memorise the training set.

```python
XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
             subsample=0.7, colsample_bytree=0.7,
             min_child_weight=5, reg_lambda=2.0)
```

**Always run a Ridge baseline first.** If boosting doesn't clearly beat it under spatial CV, ship
Ridge — simpler, more defensible to a policy audience. The grid-cell terrain model has more rows
and can carry a deeper tree; the two models must not share hyperparameters.

### 2.7 Cut from scope

- **LSTM / forecasting** — cut. There is no time axis in the feature matrix. If forecasting ever
  returns, it belongs on the NASA POWER daily series `(n_locations, 365, n_vars)`, not here.
- **Per-building rooftop modelling** — cut. Tract grain only.

---

## 3. Feature tables

Two model-ready tables, both built by `pipeline/features.py`. Inspect the live schema before
modelling — the pipeline produces more derived columns than are listed here.

### `features.tract_features` — ~1,460 rows, rooftop + equity

| Group | Columns | Source |
|---|---|---|
| Key / geometry | `geoid`, `geometry`, `aland_m2` | Census TIGER |
| Irradiance | GHI climatology, clearness index, temp | NASA POWER |
| Terrain | slope, elevation, `south_facing_share`, WorldCover class shares | DEM + WorldCover |
| Housing | `owner_rate`, `single_family_rate`, `single_family_detached_rate` | ACS |
| Income / demographics | income, `poverty_rate`, `pct_minority`, `pct_nh_black`, `pct_hispanic` | ACS |
| Energy burden | `energy_cost`, electricity/gas/other splits, low-income + renter breakdowns | NREL LEAD |
| Derived | `pop_density_km2` | computed |
| **Label** | DeepSolar adoption counts | **`y`, not a feature** |

ACS carries race and ethnicity shares. That is deliberate — Sunter et al. (2019) is specifically
about racial disparity in rooftop PV, so these belong in the **stage-2 barrier model**, never in
the stage-1 counterfactual (§2.4).

### `features.grid_features` — 1 km cells, utility-scale terrain

| Columns | Notes |
|---|---|
| `cell_id`, `geometry` | 1 km grid on a fixed origin |
| terrain + WorldCover shares | from `features.grid_terrain` |
| `ghi_kwh_m2_day` | sampled at cell centroid |
| `dist_line_km`, `dist_substation_km` | OSM nearest-neighbour |
| `pv_frac`, `pv_first_year` | USPVDB polygon overlap |
| **`has_pv`** | **label** — `pv_frac >= 0.01` (1 ha of a 1 km² cell) |

`has_pv` means the cell *contains* PV, so negatives are every other cell in Maryland. That is a
far easier task than "would a developer build here." Expect a high AUC that mostly reflects
"is this flat open land away from a city." See the leakage warning in §6 before reporting metrics.

---

## 4. Build status

Update this table as units land. Last synced with `origin/master` at `7464d7b`.

| # | Unit | Owner | Status |
|---|---|---|---|
| 0 | Scaffold — Docker, PostGIS, conda env, `pipeline/db.py` | Krish | ✅ |
| 1 | Tract + state boundaries → `raw.tracts` (`census.run_boundaries`) | Krish | ✅ |
| 2 | Irradiance pipeline — NASA POWER climatology (`irradiance.run`, `run_tracts`) | Krish | ✅ |
| 3+4 | Terrain **and** land cover — DEM + WorldCover, merged into `terrain.py` | Krish | ✅ |
| 5 | ACS + LEAD ingestion (`census.run_acs`, `lead.run`) | Krish | ✅ |
| 6 | DeepSolar adoption (`deepsolar.run`) | Krish | ✅ |
| 7 | **`features.tract_features` assembled** (`features.run_tracts`) | Krish | ✅ |
| — | *Bonus, ahead of schedule:* 1 km grid, OSM power lines, USPVDB labels, `features.grid_features` | Krish | ✅ |
| 2s | `IrradianceService` FastAPI (8001) | Krish | ✅ |
| 3s | `TerrainService` FastAPI (8002) | Krish | ✅ |
| 5s | `SocioEconService` FastAPI (8003) | Rahul | ✅ |
| 6s | `AdoptionService` FastAPI (8004) | Rahul | ✅ |
| 8 | `EquityGapService` (8005) + `ml/equity_gap.py` two-stage model | Rahul | ✅ trained, stage 2 ρ=0.541 |
| 8v | `VisualizationService` (8006) — GeoJSON, composes over HTTP | Rahul | ✅ |
| 9 | Leaflet dashboard at `/app/` consuming `/geojson/equity` | Rahul | ✅ **midterm build complete** |
| 10 | `YieldService` — pvlib specific yield | Rahul | ☐ |
| 11 | `DevelopabilityService` — grid model on `has_pv` | Rahul | ☐ |
| 12 | `ROIService` — install cost, payback, EIA rates | Rahul | ☐ |
| 13 | API Gateway + Redis caching | Rahul | ☐ |
| 14 | Full dashboard — both layers, detail panel | Rahul | ☐ |

**Units 1–9 = midterm. Units 10–14 = endterm.**

The critical path (unit 7) is **cleared** — both `features.tract_features` and
`features.grid_features` exist, so all modelling work is unblocked. The remaining midterm work is
entirely ML and service layer. Unit 11's data is also ready early, which makes the terrain track
cheaper than originally planned.

### 4.1 Patterns already established — match these

**Pipeline steps** are functions registered in `scripts/run_pipeline.py`:

```bash
python scripts/run_pipeline.py              # all steps, in order
python scripts/run_pipeline.py osm terrain  # just these
python scripts/run_pipeline.py --list
```

**I/O helpers** — `pipeline/utils.py`: `download(url, dest)` caches to disk and skips if present;
`save(df, table, schema="raw")` replaces the table, keeping geometry for GeoDataFrames;
`load(table, schema="raw")` auto-detects geometry and returns a GeoDataFrame.

**Service layer** — `services/common.py` gives `tract_rows(table, county)`, `tract_row(table, geoid)`
(404s on miss) and `ping()`. Every service follows the same shape:

```python
app = FastAPI(title="…Service")

@app.get("/health")        def health() -> dict: return ping()
@app.get("/x")             # list, optional ?county=24005  (pattern=COUNTY)
@app.get("/x/{geoid}")     # one tract                     (pattern=GEOID)
```

`GEOID = r"^24\d{9}$"` and `COUNTY = r"^24\d{3}$"` validate at the route boundary. Keep new
services to this shape — consistency across services is part of what the SOA grade rewards.

**Config constants** live in `pipeline/config.py`: `CRS = EPSG:26985`, `ACS_YEAR = 2022`,
`GRID_SIZE = 1000` m, `FINE_RES = 100` m, `STATE_FIPS = "24"`.

### 4.2 Ports and run commands

```bash
uvicorn services.irradiance:app    --port 8001
uvicorn services.terrain:app       --port 8002
uvicorn services.socioecon:app     --port 8003
uvicorn services.adoption:app      --port 8004
uvicorn services.equity:app        --port 8005   # needs scripts/train_equity.py first
uvicorn services.visualization:app --port 8006   # needs 8005 running
```

`VisualizationService` reads `EQUITY_SERVICE_URL` (default `http://localhost:8005`). It is the
only service that calls another over REST rather than reading PostGIS directly — that composition
is the point, so keep it that way. Its `/health` reports upstream status so a failure is
attributable to the right service.

### 4.3 Running and testing

```bash
python scripts/run_services.py            # all six, Ctrl+C stops them
python scripts/run_services.py --list     # names and ports
pytest -v                                 # 39 tests
pytest -v -m "not live"                   # invariants only, no services needed
```

Dashboard: **http://localhost:8006/app/**

`tests/` holds integration tests, not unit tests, and that is deliberate. The services are
thin enough that mocking the database would test SQLAlchemy rather than this project, and
three of the four real bugs this project shipped were semantic — a ranking that inverted, an
expectation that leaked location — in code that was logically correct. **Every test in
`test_model_invariants.py` encodes a bug that actually shipped**, and its failure message
names what broke. Verified by mutation: corrupting the scores in the way each bug corrupted
them makes the matching test fail.

`test_api_contract.py` doubles as the pre-demo smoke test. If it is green, the dashboard works.

**Address search** suggests as you type (`/suggest`, backed by Photon over OpenStreetMap:
typo- and abbreviation-tolerant, Maryland only). A picked suggestion sends its coordinates to
`/lookup?lat=&lon=`, so the tract still comes from the Census geocoder (`/geocode/reverse`),
and its OSM id fetches the building footprint (`/geocode/building`) that the map outlines in
yellow on satellite imagery. Enter with no suggestion falls back to the exact Census match.

**Demo addresses — verified against the live Census geocoder.** Its address file is
incomplete (`9000 Rockville Pike, Bethesda, MD` fails even with its ZIP), so never improvise
an address on stage.

| Address | Tract | Tier | Gap | Note |
|---|---|---|---|---|
| 5200 Park Heights Ave, Baltimore, MD 21215 | 24510271801 | **HIGH** | +3.16 | zero installed solar |
| 2201 Argonne Dr, Baltimore, MD 21251 | 24510090200 | MEDIUM | +2.33 | Morgan State area |
| 1 E Pratt St, Baltimore, MD 21202 | 24510220100 | LOW | −1.66 | Inner Harbor, over-performing |
| 8600 Snowden River Pkwy, Columbia, MD 21045 | 24027606706 | LOW | −1.47 | Howard County contrast |
| 1000 Hilltop Circle, Baltimore, MD 21250 | 24005492500 | *unscored* | — | UMBC — under 50 households |

### 4.4 Training the equity model

```bash
python scripts/run_pipeline.py      # must have produced features.tract_features
python scripts/train_equity.py      # writes scores.tract_equity + models/equity_metrics.json
```

`ml/` is the ML package, kept separate from Krish's `pipeline/`. Training is offline and batch;
services only ever read what it wrote.

---

## 5. Deliverables

### Midterm — prove the pipeline and the equity model work

Build units 1–9. On stage, show:

- **The contrast table.** Two tracts, near-identical `ghi_annual`, wildly different
  `adoption_per_km2`. This is the whole thesis in three rows and lands harder than any metric.
- **Live query** — a Baltimore City county call, map colours in by gap score.
- **Click a tract** → score plus top SHAP drivers (`renter_ratio −0.34`).
- **Honest metrics** — spatial CV R², with the explanation of why it's lower than a random split.
- **Ridge vs XGBoost**, both reported.

### Endterm — prove it is a service-oriented platform

Build units 10–14. On stage, show:

- End to end: county in → ranked sites with cost and payback out
- Both tracks on one map, toggleable
- **Each service called independently** — hit one endpoint in isolation to prove the SOA claim
- Evaluation against a naive "highest irradiance wins" baseline

### Cut list, if behind

Drop in this order — each costs the least next:

1. Redis caching → direct DB reads
2. Terrain / USPVDB track → keep rooftop, closer to the pitch
3. ROI service → report kWh only, no dollars

**Never cut:** the equity gap model or SHAP. That is the project.

---

## 6. Gotchas

**⚠ Land-cover leakage in the grid model.** ESA WorldCover is the 2021 map; existing solar farms
may already be classified as `built` or `bare`. The grid label `has_pv` is derived from USPVDB
facility polygons, so a model given `built_share` / `bare_share` can partly read the answer off
its own features. Before trusting unit 11's metrics:

- check how `pv_frac` correlates with `built_share` and `bare_share`
- compare against `p_year` — cells whose facility predates 2021 are the leaky ones
- either drop those land-cover shares for the PV model, or restrict training to
  facilities built after the WorldCover vintage

Flagged by Krish in the `features.py` docstring. Treat it as a real threat to validity, not a note.

**Aspect encoding.** Never feed raw aspect degrees to a tree — 359° and 1° are nearly the same
direction but maximally far apart numerically.

```python
aspect_southness = np.cos(np.radians(aspect_deg - 180))   # 1.0 = due south
```

Note the pipeline already provides a `south_facing_share` per cell/tract rather than a raw mean
aspect, which sidesteps this — use that column rather than recomputing.

**Vintage alignment.** ACS 2018–2022 and NREL LEAD 2022 were chosen deliberately because both sit
on **2020 tract boundaries**. Do not mix in a source on 2010 boundaries without re-crosswalking.

**DeepSolar is stale.** Built from 2017 imagery; US solar has roughly tripled since. State this
limitation openly in the presentation — acknowledging it earns more than ignoring it. Optionally
supplement with Maryland PSC interconnection queue data.

**Census ACS returns strings.** Every value, including numerics. Cast explicitly.

**NASA POWER units.** `ALLSKY_SFC_SW_DWN` is kWh/m²/**day**. Sum 365 for annual.

**Hard negatives for USPVDB.** Sampling random wilderness as negatives makes the task trivial —
the model learns "is it land?" and hits 0.99 AUC meaninglessly. Sample negatives from parcels that
*look* viable (open land, ≥10 ha, unprotected) but weren't developed.

**API rate limits.** NREL ~1,000 req/hr. 1,460 tract calls → run overnight, cache to
`data/raw/`, and never re-pull in the request path.

**Projections.** Compute areas and distances in a projected CRS (`EPSG:26985`, Maryland State
Plane), not EPSG:4326.

---

## 7. Conventions

```
data/raw/          source downloads, gitignored — cache every API pull here
data/processed/    cleaned outputs, gitignored
pipeline/          ingestion scripts; db.py returns the PostGIS engine
services/          FastAPI microservices, one module per service
models/            trained model artifacts (.json), gitignored
notebooks/         exploration only — nothing load-bearing lives here
```

**DB schemas:** `raw` (as ingested) → `features` (engineered) → `scores` (model output).

**Service contract rule:** every service returns structured JSON and is callable on its own. No
service may require another service to have run first within a request — that coupling defeats the
whole SOA argument.

**Models load at startup, never train in the request path.** Training is offline and batch;
serving is online and read-only. One arrow between those planes, pointing one way.

---

## 8. Key references

| Purpose | Citation |
|---|---|
| Equity methodology | Sunter, Castellanos & Kammen, *Nature Sustainability* 2, 71–76, 2019 — https://doi.org/10.1038/s41893-018-0204-z |
| Adoption labels | Yu et al., "DeepSolar," *Joule* 2(12), 2018 — https://doi.org/10.1016/j.joule.2018.11.021 |
| Income equity in adoption | O'Shaughnessy et al., *Nature Energy* 6, 84–91, 2021 — https://doi.org/10.1038/s41560-020-00724-2 |
| Energy burden statistics | Drehobl & Ross, ACEEE U1602, 2016 — https://www.aceee.org/research-report/u1602 |
| Spatial CV justification | Roberts et al., *Ecography* 40(8), 913–929, 2017 — https://doi.org/10.1111/ecog.02881 |
| Model choice defence | Grinsztajn et al., NeurIPS 2022 — https://arxiv.org/abs/2207.08815 |
| XGBoost | Chen & Guestrin, KDD 2016 — https://arxiv.org/abs/1603.02754 |
| SHAP | Lundberg & Lee, NeurIPS 2017 — https://arxiv.org/abs/1705.07874 |

**Data portals:** NASA POWER `power.larc.nasa.gov` · Copernicus `dataspace.copernicus.eu` ·
WorldCover `esa-worldcover.org` · Census `census.gov/data/developers` · NREL LEAD
`lead.openei.org` · DeepSolar `web.stanford.edu/group/deepsolar` · USPVDB
`eerscmap.usgs.gov/uspvdb` · OSM `download.geofabrik.de`
