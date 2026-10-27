"""Invariants on scores.tract_equity.

Every test here encodes a bug this project actually shipped. They are not generic sanity
checks - each one failed at some point, and the assertion message names what broke.

    pytest tests/test_model_invariants.py -v
"""

import numpy as np
import pandas as pd

BARRIER_COLS = [(f"barrier_{i}", f"barrier_{i}_shap") for i in (1, 2, 3)]

BALTIMORE_CITY = "24510"
HOWARD = "24027"


def high_share(df: pd.DataFrame, county: str) -> float:
    sub = df[df["geoid"].str.startswith(county)]
    return (sub["priority_tier"] == "HIGH").mean() * 100


# --------------------------------------------------------------------------- stage 2

def test_stored_barriers_only_widen_the_gap(scores):
    """Ranking by argmax(|shap|) named whichever factor moved the gap in either
    direction, so 98 of 364 HIGH tracts had a 'primary barrier' that shrank theirs."""
    for name_col, shap_col in BARRIER_COLS:
        vals = scores.loc[scores[name_col].notna(), shap_col]
        bad = (vals <= 0).sum()
        assert bad == 0, (
            f"{bad} tracts store a non-positive SHAP in {shap_col}. Only factors that "
            "widen the gap qualify as barriers - rank by signed value, not magnitude."
        )


def test_barriers_are_ranked_descending(scores):
    both = scores[scores["barrier_2_shap"].notna()]
    assert (both["barrier_1_shap"] >= both["barrier_2_shap"]).all(), \
        "barrier_1 must be the largest contribution"
    three = scores[scores["barrier_3_shap"].notna()]
    assert (three["barrier_2_shap"] >= three["barrier_3_shap"]).all(), \
        "barrier_2 must exceed barrier_3"


def test_primary_barrier_mirrors_barrier_1(scores):
    scored = scores[scores["barrier_1"].notna()]
    assert (scored["primary_barrier"] == scored["barrier_1"]).all()


def test_no_single_factor_dominates(scores):
    """pct_minority was the sole named barrier in 77% of tracts before the top-three
    change. The barriers are collinear, so a tree starves the rest of a correlated
    group; a single label creeping back past ~65% means that is happening again."""
    vc = scores["primary_barrier"].value_counts(normalize=True)
    if vc.empty:
        return
    top, share = vc.index[0], vc.iloc[0]
    assert share < 0.65, (
        f"'{top}' is the primary barrier in {share:.0%} of tracts. Check collinearity "
        "among the barrier features before presenting this."
    )


# --------------------------------------------------------------------------- stage 1

def test_expected_adoption_does_not_vary_by_location(scores):
    """Stage 1 has no county-CV skill, so any variation it emits is noise. Scoring with
    the fitted ridge gave county medians from 2.20 to 14.75 and sent three Eastern Shore
    counties to 100% HIGH priority on fractionally more sun."""
    distinct = scores["expected_adoption"].round(2).nunique()
    assert distinct == 1, (
        f"expected_adoption takes {distinct} distinct values. While stage 1 shows no "
        "skill it must be a constant, or it launders location into the ranking."
    )


def test_county_expectation_spread_is_flat(scores):
    med = scores.assign(cty=scores["geoid"].str[:5]).groupby("cty")["expected_adoption"].median()
    ratio = med.max() / max(med.min(), 0.01)
    assert ratio < 1.05, (
        f"county median expectation spans {ratio:.1f}x ({med.min():.2f} to {med.max():.2f}). "
        "This was 6.7x before the statewide-median fix."
    )


def test_gap_is_expected_minus_actual(scores):
    s = scores.dropna(subset=["gap", "expected_adoption", "actual_adoption"])
    diff = (s["expected_adoption"] - s["actual_adoption"] - s["gap"]).abs()
    assert (diff < 0.01).all(), "gap must equal expected_adoption - actual_adoption"


# --------------------------------------------------------------------------- semantics

def test_baltimore_city_outranks_howard_county(scores):
    """The invariant the original design violated. Baltimore City has roughly a ninth of
    Howard County's adoption on the same sunlight, so it cannot be the lower priority.
    Putting owner_rate in stage 1 produced 2% HIGH for Baltimore against 22% for Howard."""
    bc, ho = high_share(scores, BALTIMORE_CITY), high_share(scores, HOWARD)
    assert bc > ho, (
        f"Baltimore City is HIGH priority in {bc:.1f}% of tracts against Howard County's "
        f"{ho:.1f}%. A wealthier county with far more installed solar ranking higher means "
        "a structural barrier has leaked into the stage 1 expectation."
    )


# --------------------------------------------------------------------------- integrity

def test_tiers_are_valid_or_null(scores):
    allowed = {"HIGH", "MEDIUM", "LOW"}
    seen = set(scores["priority_tier"].dropna().unique())
    assert seen <= allowed, f"unexpected tier values: {seen - allowed}"


def test_tier_is_null_exactly_when_gap_is_null(scores):
    mismatch = (scores["priority_tier"].isna() != scores["gap"].isna()).sum()
    assert mismatch == 0, f"{mismatch} tracts have a tier without a gap, or the reverse"


def test_percentile_ordering_matches_tiers(scores):
    s = scores.dropna(subset=["priority_tier", "gap_percentile"])
    assert s.loc[s["priority_tier"] == "HIGH", "gap_percentile"].min() >= \
           s.loc[s["priority_tier"] == "MEDIUM", "gap_percentile"].max() - 1e-6, \
        "HIGH tracts must sit above MEDIUM on gap_percentile"


def test_geoids_are_unique_maryland_tracts(scores):
    assert scores["geoid"].is_unique, "duplicate geoid in scores.tract_equity"
    assert scores["geoid"].str.match(r"^24\d{9}$").all(), "non-Maryland or malformed geoid"
