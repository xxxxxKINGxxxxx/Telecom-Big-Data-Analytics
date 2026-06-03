"""
01_generate_dataset.py
======================
Synthetic telecom customer dataset generator for the Telecom Big Data
Analytics Platform.

Produces a deliberately *raw* (intentionally messy) CSV that downstream
cleaning (02) is responsible for fixing. Churn is generated from real
business drivers so that the ML layer (04) finds genuine signal.

Output: data/raw/telecom_customers.csv  (~50,000 rows)
Run from project root:  python scripts/01_generate_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.random import Generator

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
N_ROWS: Final[int] = 50_000
RANDOM_SEED: Final[int] = 42
TARGET_CHURN_RATE: Final[float] = 0.165  # 16.5% — mid-point of industry 15–18%

# Paths are resolved relative to THIS file, so the script works no matter
# which directory you run it from.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUTPUT_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "telecom_customers.csv"

# Categorical domains and their real-world imbalance (weights must sum to 1.0)
REGIONS: Final[tuple[str, ...]] = ("North", "South", "East", "West", "Central")
REGION_WEIGHTS: Final[tuple[float, ...]] = (0.30, 0.25, 0.15, 0.10, 0.20)

PLAN_TYPES: Final[tuple[str, ...]] = ("Prepaid", "Postpaid", "Business")
PLAN_WEIGHTS: Final[tuple[float, ...]] = (0.45, 0.40, 0.15)

# Per-region signal quality offset — some regions have worse infrastructure.
# This makes the Regional Performance dashboard (Step 8) show REAL differences.
REGION_SIGNAL_OFFSET: Final[dict[str, float]] = {
    "North": 0.20, "South": 0.00, "East": -0.30, "West": -0.40, "Central": 0.10,
}

# Final column order (must match the project brief exactly)
COLUMN_ORDER: Final[list[str]] = [
    "customer_id", "region", "plan_type", "tenure_months", "monthly_spend",
    "data_usage_gb", "call_minutes", "sms_count", "network_complaints",
    "signal_quality_score", "contract_end_months", "churn_flag",
    "last_upgrade_months",
]


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function. Clip the exponent to avoid float overflow warnings."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def calibrate_intercept(
    logit: np.ndarray,
    target_rate: float,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """
    Find the intercept `c` such that mean(sigmoid(logit + c)) == target_rate,
    using bisection. This lets us hit an exact overall churn rate while keeping
    each customer's relative risk driven by their features.
    """
    low, high = -20.0, 20.0
    midpoint = 0.0
    for _ in range(max_iter):
        midpoint = (low + high) / 2.0
        rate = float(sigmoid(logit + midpoint).mean())
        if abs(rate - target_rate) < tol:
            return midpoint
        if rate < target_rate:
            low = midpoint    # need higher probabilities -> raise intercept
        else:
            high = midpoint   # need lower probabilities  -> lower intercept
    return midpoint


# --------------------------------------------------------------------------- #
# Core generation
# --------------------------------------------------------------------------- #
def generate_clean_frame(rng: Generator, n: int) -> pd.DataFrame:
    """Generate a fully clean, feature-correlated customer dataset."""

    # --- Identifiers (vectorized, guaranteed unique) ---
    seq = (np.arange(n) + 1).astype(str)
    customer_id = np.char.add("CUST-", np.char.zfill(seq, 7))

    # --- Categorical columns ---
    region = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    plan_type = rng.choice(PLAN_TYPES, size=n, p=PLAN_WEIGHTS)

    # Boolean masks reused across columns (vectorized, no loops)
    is_prepaid = plan_type == "Prepaid"
    is_postpaid = plan_type == "Postpaid"
    is_business = plan_type == "Business"

    # --- Tenure: gamma distribution -> many newer customers, long tail ---
    tenure_months = np.clip(rng.gamma(shape=2.0, scale=18.0, size=n), 1, 120).astype(int)

    # --- Monthly spend (EUR): plan-driven base * lognormal spread ---
    spend_base = np.select(
        [is_prepaid, is_postpaid, is_business], [15.0, 42.0, 90.0]
    )
    monthly_spend = np.round(spend_base * rng.lognormal(mean=0.0, sigma=0.25, size=n), 2)

    # --- Data usage (GB): plan-driven, heavier on postpaid/business ---
    usage_base = np.select(
        [is_prepaid, is_postpaid, is_business], [3.0, 12.0, 25.0]
    )
    data_usage_gb = np.round(usage_base * rng.lognormal(mean=0.0, sigma=0.50, size=n), 2)

    # --- Call minutes & SMS (integer counts) ---
    call_minutes = np.clip(rng.gamma(shape=2.0, scale=120.0, size=n), 0, None).astype(int)
    sms_count = rng.poisson(lam=15.0, size=n).astype(int)

    # --- Signal quality (1.0–5.0): normal + per-region infrastructure offset ---
    region_offset = np.vectorize(REGION_SIGNAL_OFFSET.__getitem__, otypes=[float])(region)
    signal_quality_score = np.clip(
        rng.normal(loc=3.8, scale=0.6, size=n) + region_offset, 1.0, 5.0
    ).round(1)

    # --- Network complaints: Poisson whose rate RISES as signal falls ---
    complaint_lambda = np.clip(0.30 + (3.8 - signal_quality_score) * 0.40, 0.05, 3.0)
    network_complaints = rng.poisson(lam=complaint_lambda).astype(int)

    # --- Contract end (months): prepaid has no contract (0) ---
    contract_end_months = np.where(is_prepaid, 0, rng.integers(0, 25, size=n)).astype(int)

    # --- Last upgrade (months ago) ---
    last_upgrade_months = rng.integers(0, 49, size=n).astype(int)

    # --- CHURN: generated from real drivers, then calibrated to target rate ---
    contract_soon = ((contract_end_months > 0) & (contract_end_months <= 3)).astype(float)
    churn_logit = (
        -0.025 * tenure_months                       # loyalty reduces churn
        + 0.85 * network_complaints                  # pain increases churn
        - 0.55 * (signal_quality_score - 3.0)        # bad signal increases churn
        + 0.45 * is_prepaid.astype(float)            # prepaid churns more easily
        + 0.40 * contract_soon                       # contract ending = flight risk
        - 0.004 * monthly_spend                      # higher spenders stickier
        + rng.normal(0.0, 0.5, size=n)               # irreducible randomness
    )
    intercept = calibrate_intercept(churn_logit, TARGET_CHURN_RATE)
    churn_prob = sigmoid(churn_logit + intercept)
    churn_flag = (rng.random(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "customer_id": customer_id,
        "region": region,
        "plan_type": plan_type,
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "data_usage_gb": data_usage_gb,
        "call_minutes": call_minutes,
        "sms_count": sms_count,
        "network_complaints": network_complaints,
        "signal_quality_score": signal_quality_score,
        "contract_end_months": contract_end_months,
        "churn_flag": churn_flag,
        "last_upgrade_months": last_upgrade_months,
    })[COLUMN_ORDER]


def inject_data_quality_issues(df: pd.DataFrame, rng: Generator) -> pd.DataFrame:
    """
    Deliberately corrupt the clean frame to mimic real-world raw data.
    Step 02 (cleaning) is responsible for reversing every issue below.
    All issues are documented so cleaning is verifiable, not guesswork.
    """
    df = df.copy()
    n = len(df)

    # 1) Outliers / impossible values (entered BEFORE nulls to avoid clashing)
    neg_idx = rng.choice(n, size=5, replace=False)
    df.loc[neg_idx, "monthly_spend"] = [-12.5, -8.0, -45.0, -3.2, -19.9]

    hi_idx = rng.choice(n, size=5, replace=False)
    df.loc[hi_idx, "data_usage_gb"] = [9999.0, 8500.0, 12000.0, 7777.0, 10500.0]

    # 2) Missing values (NaN). Note: injecting NaN into int columns forces them
    #    to float — exactly the dtype mess Step 02 will have to re-cast.
    def blank_out(col: str, frac: float) -> None:
        k = int(n * frac)
        idx = rng.choice(n, size=k, replace=False)
        df.loc[idx, col] = np.nan

    blank_out("signal_quality_score", 0.015)
    blank_out("monthly_spend", 0.010)
    blank_out("data_usage_gb", 0.008)
    blank_out("tenure_months", 0.007)

    # 3) Inconsistent text casing in region ("North" -> "north")
    k_case = int(n * 0.010)
    case_idx = rng.choice(n, size=k_case, replace=False)
    df.loc[case_idx, "region"] = df.loc[case_idx, "region"].str.lower()

    # 4) Duplicate full rows (same customer_id appears twice) for dedup demo
    k_dupe = int(n * 0.002)
    dupe_idx = rng.choice(n, size=k_dupe, replace=False)
    df = pd.concat([df, df.iloc[dupe_idx]], ignore_index=True)

    return df


# --------------------------------------------------------------------------- #
# Reporting & entry point
# --------------------------------------------------------------------------- #
def print_summary(df: pd.DataFrame) -> None:
    """Print a verification report so you can sanity-check the output."""
    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total rows (incl. duplicates) : {len(df):,}")
    print(f"Unique customers              : {df['customer_id'].nunique():,}")
    print(f"Overall churn rate            : {df['churn_flag'].mean():.2%}")
    print("\nRegion distribution (raw, casing intentionally messy):")
    print(df["region"].value_counts().to_string())
    print("\nNull counts per column (intentional — Step 02 fixes these):")
    print(df.isna().sum().to_string())
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print("=" * 60 + "\n")


def main() -> int:
    """Generate, dirty, and persist the dataset. Returns a process exit code."""
    try:
        rng: Generator = np.random.default_rng(RANDOM_SEED)

        clean_df = generate_clean_frame(rng, N_ROWS)
        raw_df = inject_data_quality_issues(clean_df, rng)

        if raw_df.empty:
            print("ERROR: generated dataframe is empty — aborting.", file=sys.stderr)
            return 1

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(OUTPUT_PATH, index=False)

    except (OSError, PermissionError) as exc:
        print(f"ERROR: could not write CSV to {OUTPUT_PATH}\n  -> {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: invalid generation parameter\n  -> {exc}", file=sys.stderr)
        return 1

    print_summary(raw_df)
    print(f"SUCCESS: dataset written to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())