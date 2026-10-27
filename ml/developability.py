"""Utility-scale developability: which 1 km cells look like places solar farms get built.

Label is has_pv (USPVDB facility covers >= 1% of the cell): 200 positives in 28,258 cells,
0.7% prevalence. The score is a relative ranking, not a calibrated probability - class
weighting inflates the raw output, so consumers should read the percentile.

Leakage. ESA WorldCover is the 2021 map and USPVDB polygons are the label, so land-cover
classes can read existing farms off the imagery. Measured before modelling: cells whose
facility predates 2021 are 15.2% "built" against 9.0% for facilities built after, i.e.
WorldCover partly labels panels as built-up. lc_built_frac and lc_bare_frac are therefore
excluded. Cells touched by a facility below the 1% threshold are neither clean positives
nor clean negatives and are left out of training. Grid row/col are excluded: they are
location, not suitability. slope_p90_deg is dropped as near-collinear with slope_mean_deg:
in the linear model the pair took opposite signs, so "steep" read as a reason to build.
GHI is excluded from the model for the same reason as in the equity model: it spans 1.7%
statewide, so a fitted weight on it reads location, not physics (it came out as "less sun
is better"). It remains as the naive baseline.

Developed land. With built-up share out of the feature set, a city block looks like flat,
treeless land beside a substation, and a third of the HIGH tier came out >= 40% built-up
(104 HIGH cells in Baltimore City). Siting studies exclude developed land as a hard
constraint rather than score it, so cells at or above MAX_BUILT_FRAC are tier EXCLUDED and
kept out of training. A mask cannot leak the label the way a feature can: if anything it
drops leaky positives (panels mapped as built), costing 21 of 195 farm cells.

Evaluation, all out-of-fold:
- spatial CV: GroupKFold over 20 km blocks, so neighbouring cells never straddle folds
- temporal: train with every post-2021 facility (and 20% of blocks' negatives) removed,
  then ask how highly the model ranks the cells where those facilities were later built
- baselines: "highest irradiance wins" and "nearest substation wins", single features

Outputs:
    scores.grid_developability  one row per land cell (county is null for cells whose
                                centroid falls outside every tract, e.g. on the bay)
    models/developability_metrics.json
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline.config import ROOT
from pipeline.utils import load, save

MODEL_DIR = ROOT / "models"
TARGET = "has_pv"
WORLDCOVER_YEAR = 2021
BLOCK_CELLS = 20      # 20 km spatial CV blocks
MIN_LAND_FRAC = 0.5   # mostly-water cells are neither scored nor trained on
MAX_BUILT_FRAC = 0.4  # developed cells are excluded, not scored

FEATURES = [
    "slope_mean_deg", "flat_frac", "south_facing_frac",
    "elev_mean_m", "elev_range_m",
    "lc_tree_frac", "lc_grass_frac", "lc_crop_frac", "lc_water_frac", "lc_wetland_frac",
    "dist_line_km", "dist_substation_km",
]

# ~28,000 rows, so this can carry deeper trees than the tract model; deliberately not
# the equity model's hyperparameters.
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    eval_metric="aucpr",
    random_state=0,
    n_jobs=4,
)


def _logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))


def _xgb(y: pd.Series):
    return XGBClassifier(**XGB_PARAMS, scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))


def _scores(y: pd.Series, p: pd.Series) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4),
            "avg_precision": round(float(average_precision_score(y, p)), 4)}


def _oof(factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.Series:
    pred = pd.Series(np.nan, index=X.index)
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        model = factory(y.iloc[train_idx])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred.iloc[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return pred


def fit() -> tuple[pd.DataFrame, dict]:
    grid = load("grid_features", schema="features")
    grid = grid[grid["land_frac"] >= MIN_LAND_FRAC]
    # county of each cell's centroid, so services can filter by county without a spatial join
    tracts = load("tracts")[["geoid", "geometry"]]
    centroids = gpd.GeoDataFrame(grid[["cell_id"]], geometry=grid.centroid, crs=grid.crs)
    county = gpd.sjoin(centroids, tracts, predicate="within").drop_duplicates("cell_id")
    df = pd.DataFrame(grid.drop(columns="geometry"))
    df["county"] = df["cell_id"].map(county.set_index("cell_id")["geoid"].str[:5])

    developed = df["lc_built_frac"] >= MAX_BUILT_FRAC
    sliver = df["pv_first_year"].notna() & (df[TARGET] == 0)
    train = df[~sliver & ~developed]
    X, y = train[FEATURES], train[TARGET]
    blocks = (train["row"] // BLOCK_CELLS).astype(str) + "_" + (train["col"] // BLOCK_CELLS).astype(str)
    print(f"  {len(df):,} land cells, {int(developed.sum()):,} developed excluded, {len(train):,} trainable "
          f"({int(y.sum())} positive, {int((sliver & ~developed).sum())} sub-threshold slivers excluded), "
          f"{blocks.nunique()} CV blocks")

    # ---- spatial CV --------------------------------------------------------------
    models = {"logistic": lambda _: _logistic(), "xgboost": _xgb}
    spatial = {name: _scores(y, _oof(f, X, y, blocks)) for name, f in models.items()}
    spatial["baseline_irradiance"] = _scores(y, train["ghi_kwh_m2_day"])
    spatial["baseline_substation"] = _scores(y, -X["dist_substation_km"])
    spatial["prevalence"] = round(float(y.mean()), 4)
    for name, s in spatial.items():
        if isinstance(s, dict):
            print(f"  spatial CV  {name:20} AUC={s['roc_auc']:.3f}  AP={s['avg_precision']:.3f}")

    # Ship the tree only if it clearly beats the linear baseline on the rare-class metric.
    chosen = "xgboost" if spatial["xgboost"]["avg_precision"] > 1.1 * spatial["logistic"]["avg_precision"] else "logistic"

    # ---- temporal: forecast facilities built after the WorldCover snapshot --------
    # Negatives from a random 20% of spatial blocks are held out too, so neither side of
    # the test was seen in training.
    late = (train[TARGET] == 1) & (train["pv_first_year"] >= WORLDCOVER_YEAR)
    rng = np.random.default_rng(0)
    test_blocks = rng.choice(blocks.unique(), size=blocks.nunique() // 5, replace=False)
    held_neg = (train[TARGET] == 0) & blocks.isin(test_blocks)
    early = train[~late & ~held_neg]
    model = models[chosen](early[TARGET])
    model.fit(early[FEATURES], early[TARGET])
    holdout = train[late | held_neg]
    p = pd.Series(model.predict_proba(holdout[FEATURES])[:, 1], index=holdout.index)
    temporal = _scores(holdout[TARGET], p)
    temporal["n_late_sites"] = int(late.sum())
    temporal["late_sites_in_top_10pct"] = round(float((p.rank(pct=True)[late] >= 0.9).mean()), 3)
    print(f"  temporal    {chosen:20} AUC={temporal['roc_auc']:.3f}  "
          f"{temporal['late_sites_in_top_10pct']:.0%} of {temporal['n_late_sites']} post-{WORLDCOVER_YEAR} sites in top 10%")

    # ---- final model, all trainable cells; score every land cell -----------------
    final = models[chosen](y)
    final.fit(X, y)
    df["score"] = final.predict_proba(df[FEATURES])[:, 1].round(4)
    df.loc[developed, "score"] = np.nan
    df["score_percentile"] = df["score"].rank(pct=True).round(4)  # among undeveloped cells
    df["tier"] = pd.cut(df["score_percentile"], [0, 0.75, 0.95, 1.0],
                        labels=["LOW", "MEDIUM", "HIGH"], include_lowest=True).astype("object")
    df.loc[developed, "tier"] = "EXCLUDED"

    # Signed SHAP, top 3 positive drivers per cell, same convention as the equity model.
    if chosen == "xgboost":
        sv = shap.TreeExplainer(final).shap_values(df[FEATURES])
    else:
        Xs = final[:-1].transform(df[FEATURES])
        sv = Xs * final[-1].coef_[0]  # exact SHAP for a linear model on standardized inputs
    # A driver is named with its direction ("low slope_mean_deg"): a positive contribution
    # from slope means the cell is flatter than average, which the bare name hides.
    above = (df[FEATURES] > X.mean()).to_numpy()
    order = np.argsort(-sv, axis=1)
    rows = np.arange(len(df))
    for rank in range(3):
        idx = order[:, rank]
        val = sv[rows, idx].round(4)
        named = np.where(above[rows, idx], "high ", "low ") + np.array(FEATURES)[idx]
        keep = (val > 0) & ~developed.to_numpy()
        df[f"driver_{rank + 1}"] = np.where(keep, named, None)
        df[f"driver_{rank + 1}_shap"] = np.where(keep, val, np.nan)
    print("  top drivers:", df["driver_1"].value_counts().head(5).to_dict())

    out = df[["cell_id", "county", "has_pv", "pv_first_year", "score", "score_percentile", "tier",
              "driver_1", "driver_1_shap", "driver_2", "driver_2_shap", "driver_3", "driver_3_shap"]]

    metrics = {
        "n_land_cells": int(len(df)),
        "n_developed_excluded": int(developed.sum()),
        "developed_rule": f"lc_built_frac >= {MAX_BUILT_FRAC} -> tier EXCLUDED, not trained on",
        "n_trainable": int(len(train)),
        "n_positive": int(y.sum()),
        "target": f"{TARGET} (USPVDB facility covers >= 1% of a 1 km cell)",
        "features": FEATURES,
        "excluded_for_leakage": ["lc_built_frac", "lc_bare_frac", "row", "col"],
        "excluded_as_proxy_or_collinear": ["ghi_kwh_m2_day", "slope_p90_deg"],
        "leakage_evidence": "built share 0.152 in cells with pre-2021 facilities vs 0.090 post-2021",
        "cv": f"GroupKFold(5) over {BLOCK_CELLS} km spatial blocks",
        "spatial_cv": spatial,
        "temporal_holdout": temporal,
        "selected": chosen,
        "caveats": [
            "score is a ranking, not a probability: class weighting inflates raw output",
            "the label is where farms were built, which reflects developer choices and "
            "interconnection queues as much as physical suitability",
            "zoning, land price and parcel ownership are not in the feature set",
        ],
    }
    return out, metrics


def run() -> None:
    scores, metrics = fit()
    MODEL_DIR.mkdir(exist_ok=True)
    save(scores, "grid_developability", schema="scores")
    (MODEL_DIR / "developability_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("  wrote models/developability_metrics.json")
