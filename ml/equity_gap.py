"""Two-stage equity gap model.

Stage 1 predicts the rooftop adoption a tract *should* have from its solar potential and
housing stock alone, deliberately excluding income, tenure, race and energy burden. The
residual (expected - actual) is the gap. Stage 2 then explains that gap using only those
excluded barrier variables, and SHAP names the dominant one per tract.

Fitting stage 1 with the barrier variables included would launder the inequity into the
baseline: a low-income tract with low adoption would produce a zero residual, because the
model "correctly" expected little of it. Excluding them is the whole point.

Design tension worth knowing: owner_rate sits in stage 1 (a renter physically cannot
install) while renter_share sits in stage 2 (renter-dominated areas face a structural
barrier). They are near-complements from different sources, so part of the tenure effect
is absorbed before stage 2 sees it. Move owner_rate to BARRIERS to test the alternative.

Outputs:
    scores.tract_equity       one row per tract
    models/equity_stage1.json, equity_stage2.json, equity_metrics.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from pipeline.config import ROOT
from pipeline.utils import load, save

MODEL_DIR = ROOT / "models"

TARGET = "ds_res_systems_per_1000_hh"

# Stage 1 - PHYSICAL solar resource only.
#
# An earlier version also put owner_rate and single_family_detached_rate here, reasoning
# that a renter cannot install panels so tenure is "physical capacity". That was wrong and
# it broke the model: Baltimore City (owner_rate 0.48, detached 0.14) got an expectation of
# 1.37 against an actual of 1.06, so its gap was ~0 and only 2% of its tracts came out HIGH
# priority - against 22% in Howard County, the wealthiest in the state. Tenure and housing
# type ARE the structural barrier; conditioning on them launders the inequity into the
# baseline, which is the exact failure this two-stage design exists to avoid.
#
# Across one state these features barely vary (GHI has a coefficient of variation of 0.017),
# so expected adoption lands near the statewide mean for every tract. That is the finding,
# not a defect: Maryland's sunlight is uniform, its adoption is not, and the difference is
# structural rather than physical.
POTENTIAL = [
    "ghi",
    "clearness_index",
    "south_facing_frac",
    "slope_mean_deg",
]

# Stage 2 - every structural factor stage 1 was not allowed to see, tenure and housing
# type included.
BARRIERS = [
    "owner_rate",
    "single_family_detached_rate",
    "median_year_built",
    "pop_density_km2",
    "median_household_income",
    "median_home_value",
    "poverty_rate",
    "energy_burden",
    "low_income_share",
    "renter_share",
    "pct_minority",
]

# ~1,460 rows: shallow trees and strong regularization, or it memorizes the state.
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_lambda=2.0,
    random_state=0,
    n_jobs=4,
)


def _ridge():
    """Baseline. Trees must beat this under spatial CV to be worth the complexity."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  WARNING: columns absent from tract_features, dropped: {missing}")
    return [c for c in cols if c in df.columns]


def _oof(model_factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None, n_splits: int = 5) -> pd.Series:
    cv = GroupKFold(n_splits=n_splits) if groups is not None else KFold(n_splits=n_splits, shuffle=True, random_state=0)
    pred = pd.Series(np.nan, index=X.index)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred.iloc[test_idx] = model.predict(X.iloc[test_idx])
    return pred


def evaluate(model_factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """Both CV schemes, both metrics.

    Grouping by county is the honest test - neighbouring tracts are correlated, and county
    means span 0.46 to 23.54 systems per 1,000 households, so a random split leaks that
    structure across folds. Reporting the random split too makes the size of the leak
    visible, which is itself a result worth showing.

    Spearman leads because this is a prioritisation tool: ranking tracts correctly matters
    more than predicting levels. R2 on levels can be negative while the ranking is useful.
    """
    out = {}
    for label, g in (("county", groups), ("random", None)):
        pred = _oof(model_factory, X, y, g)
        out[label] = {
            "r2": round(float(r2_score(y, pred)), 4),
            "spearman": round(float(spearmanr(y, pred).statistic), 4),
        }
    return out


def fit() -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(load("tract_features", schema="features").drop(columns="geometry", errors="ignore"))
    potential = _present(df, POTENTIAL)
    barriers = _present(df, BARRIERS)

    # Census uses sentinels for "no data" on median_year_built.
    if "median_year_built" in df.columns:
        df["median_year_built"] = df["median_year_built"].where(df["median_year_built"] > 1800)

    trainable = df[TARGET].notna()
    counties = df["geoid"].str[:5]
    print(f"  {len(df):,} tracts, {int(trainable.sum()):,} with a usable adoption rate")

    # ---- stage 1: expected adoption from physical resource alone -----------------
    # Modelled on log1p: the target has skew 5.26 (median 3.2, max 237), so squared
    # error on a handful of extreme tracts otherwise dominates the fit.
    X1 = df.loc[trainable, potential]
    y1 = np.log1p(df.loc[trainable, TARGET])
    g1 = counties[trainable]

    ridge_eval = evaluate(_ridge, X1, y1, g1)
    xgb_eval = evaluate(lambda: XGBRegressor(**XGB_PARAMS), X1, y1, g1)
    print(f"  stage 1 county-CV  ridge rho={ridge_eval['county']['spearman']:.3f} "
          f"R2={ridge_eval['county']['r2']:+.3f} | "
          f"xgb rho={xgb_eval['county']['spearman']:.3f} R2={xgb_eval['county']['r2']:+.3f}")

    # Ridge by deliberate choice, not by score. XGBoost scores higher here (rho 0.113 vs
    # 0.017) and that is precisely the problem: with GHI spanning only 1.7% statewide, tree
    # splits on that range are not reading physics, they are reading LOCATION - and location
    # proxies for county, which proxies for wealth. Selecting on rho picked the model that
    # leaks hardest, and it put Baltimore City's expectation at 1.30 against Prince George's
    # 10.62 when both receive effectively identical sunlight. A regularised linear model on
    # near-constant inputs returns a near-constant expectation, which is the honest answer.
    stage1 = _ridge()
    stage1.fit(X1, y1)

    df["expected_adoption"] = np.clip(np.expm1(stage1.predict(df[potential])), 0, None).round(3)
    df["actual_adoption"] = df[TARGET]
    df["gap"] = (df["expected_adoption"] - df["actual_adoption"]).round(3)

    # ---- stage 2: which structural factor explains the gap -----------------------
    scored = df["gap"].notna()
    X2, y2 = df.loc[scored, barriers], df.loc[scored, "gap"]
    gap_eval = evaluate(lambda: XGBRegressor(**XGB_PARAMS), X2, y2, counties[scored])
    print(f"  stage 2 county-CV  rho={gap_eval['county']['spearman']:.3f} "
          f"R2={gap_eval['county']['r2']:+.3f}")

    stage2 = XGBRegressor(**XGB_PARAMS)
    stage2.fit(X2, y2)

    shap_values = shap.TreeExplainer(stage2).shap_values(X2)
    top = np.argmax(np.abs(shap_values), axis=1)
    df.loc[scored, "primary_barrier"] = [barriers[i] for i in top]
    df.loc[scored, "primary_barrier_shap"] = shap_values[np.arange(len(top)), top].round(4)

    # ---- tiers -------------------------------------------------------------------
    df["gap_percentile"] = df["gap"].rank(pct=True).round(4)
    df["priority_tier"] = pd.cut(
        df["gap_percentile"], [0, 0.5, 0.75, 1.0], labels=["LOW", "MEDIUM", "HIGH"], include_lowest=True
    ).astype("object")
    df.loc[df["gap"].isna(), "priority_tier"] = None

    out = df[[
        "geoid", "actual_adoption", "expected_adoption", "gap", "gap_percentile",
        "priority_tier", "primary_barrier", "primary_barrier_shap",
    ]].copy()

    metrics = {
        "n_tracts": int(len(df)),
        "n_trainable": int(trainable.sum()),
        "target": TARGET,
        "target_transform": "log1p (raw skew 5.26)",
        "headline_metric": "Spearman rho under county-grouped CV",
        "cv": "GroupKFold(5) by county FIPS; random KFold(5) reported for contrast",
        "stage1": {
            "features": potential,
            "ridge": ridge_eval,
            "xgboost": xgb_eval,
            "selected": "ridge (chosen on principle - see POTENTIAL note; xgboost leaks geography)",
        },
        "stage2": {
            "features": barriers,
            "xgboost": gap_eval,
            "mean_abs_shap": {
                f: round(float(v), 4)
                for f, v in sorted(zip(barriers, np.abs(shap_values).mean(0)), key=lambda kv: -kv[1])
            },
        },
        "caveats": [
            "Adoption is DeepSolar's 2016 baseline; Maryland's installed base has grown since.",
            "Stage 1 features barely vary statewide (GHI coefficient of variation 0.017), so "
            "expected adoption sits near the statewide mean. That uniformity is the finding: "
            "sunlight does not explain who has solar, structure does.",
            "County-grouped R2 is far below the random-split value. The difference measures how "
            "much of adoption is clustered by county, not model failure.",
        ],
    }
    return out, metrics


def run() -> None:
    scores, metrics = fit()
    MODEL_DIR.mkdir(exist_ok=True)
    save(scores, "tract_equity", schema="scores")
    (MODEL_DIR / "equity_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  wrote models/equity_metrics.json")
