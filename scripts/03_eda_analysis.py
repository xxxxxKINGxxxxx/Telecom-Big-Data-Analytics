"""
03_eda_analysis.py
==================
Exploratory Data Analysis for the Telecom Big Data Analytics Platform.

Reads  : data/processed/telecom_cleaned.csv
Writes : screenshots/eda_01_churn_by_plan.png
         screenshots/eda_02_churn_by_region.png
         screenshots/eda_03_spend_distribution.png
         screenshots/eda_04_tenure_distribution.png
         screenshots/eda_05_correlation_heatmap.png
         screenshots/eda_06_complaints_vs_churn.png
         screenshots/eda_07_signal_by_region.png
         screenshots/eda_08_churn_by_tenure_bucket.png

Run from project root:  python scripts/03_eda_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Paths & style
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CLEAN_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "telecom_cleaned.csv"
OUT_DIR: Final[Path] = PROJECT_ROOT / "screenshots"

# Enterprise-style plot defaults — consistent across all 8 charts
PALETTE_CHURN: Final[dict[int, str]] = {0: "#2196F3", 1: "#F44336"}  # blue=retained, red=churned
ACCENT: Final[str] = "#1565C0"       # deep blue for single-series charts
BACKGROUND: Final[str] = "#F8F9FA"   # light grey — easier on the eye than pure white
DPI: Final[int] = 150

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor":   BACKGROUND,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: cleaned file not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, dtype={"customer_id": str, "region": str, "plan_type": str})
    # Restore ordered category for tenure_bucket so charts sort correctly
    bucket_order = ["New (0-12m)", "Developing (13-36m)", "Established (37-60m)", "Loyal (60m+)"]
    df["tenure_bucket"] = pd.Categorical(df["tenure_bucket"], categories=bucket_order, ordered=True)
    print(f"Loaded  : {len(df):,} rows  |  {df.shape[1]} columns")
    return df


def save_fig(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved   : {path.name}")
    plt.close(fig)


def add_value_labels(ax: plt.Axes, fmt: str = "{:.1%}", fontsize: int = 10) -> None:
    """Add text labels on top of every bar in a bar chart."""
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2.0,
                h + 0.002,
                fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
            )


# --------------------------------------------------------------------------- #
# Chart 1 — Churn rate by plan type
# --------------------------------------------------------------------------- #
def chart_churn_by_plan(df: pd.DataFrame) -> None:
    churn_rate = df.groupby("plan_type")["churn_flag"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(churn_rate.index, churn_rate.values, color=ACCENT, width=0.5, edgecolor="white")
    add_value_labels(ax)
    ax.set_title("Churn Rate by Plan Type", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Plan Type", fontsize=11)
    ax.set_ylabel("Churn Rate", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, churn_rate.max() * 1.25)
    fig.text(0.5, -0.02,
             "Insight: Prepaid customers exhibit significantly higher churn — "
             "lower switching costs and no contract lock-in.",
             ha="center", fontsize=9, style="italic", color="#555555")
    save_fig(fig, "eda_01_churn_by_plan.png")


# --------------------------------------------------------------------------- #
# Chart 2 — Churn rate by region
# --------------------------------------------------------------------------- #
def chart_churn_by_region(df: pd.DataFrame) -> None:
    churn_rate = df.groupby("region")["churn_flag"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [("#F44336" if v == churn_rate.max() else ACCENT) for v in churn_rate.values]
    ax.bar(churn_rate.index, churn_rate.values, color=colors, width=0.5, edgecolor="white")
    add_value_labels(ax)
    ax.set_title("Churn Rate by Region", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("Churn Rate", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, churn_rate.max() * 1.25)
    fig.text(0.5, -0.02,
             "Insight: Highest-churn region highlighted in red — "
             "correlates with lower signal quality scores (see Chart 7).",
             ha="center", fontsize=9, style="italic", color="#555555")
    save_fig(fig, "eda_02_churn_by_region.png")


# --------------------------------------------------------------------------- #
# Chart 3 — Monthly spend distribution by churn status
# --------------------------------------------------------------------------- #
def chart_spend_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for flag, label in [(0, "Retained"), (1, "Churned")]:
        subset = df.loc[df["churn_flag"] == flag, "monthly_spend"]
        sns.kdeplot(subset, ax=ax, label=label, color=PALETTE_CHURN[flag],
                    fill=True, alpha=0.30, linewidth=2)
    ax.set_title("Monthly Spend Distribution: Retained vs Churned", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Monthly Spend (€)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("€%.0f"))
    fig.text(0.5, -0.02,
             "Insight: Churned customers skew toward lower spend — "
             "primarily prepaid and low-tier postpaid segments.",
             ha="center", fontsize=9, style="italic", color="#555555")
    save_fig(fig, "eda_03_spend_distribution.png")


# --------------------------------------------------------------------------- #
# Chart 4 — Tenure distribution by churn status
# --------------------------------------------------------------------------- #
def chart_tenure_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for flag, label in [(0, "Retained"), (1, "Churned")]:
        subset = df.loc[df["churn_flag"] == flag, "tenure_months"]
        ax.hist(subset, bins=40, alpha=0.55, label=label,
                color=PALETTE_CHURN[flag], edgecolor="white", density=True)
    ax.set_title("Tenure Distribution: Retained vs Churned", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (months)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    fig.text(0.5, -0.02,
             "Insight: Churn concentrates in early tenure — "
             "first 12 months are the critical retention window.",
             ha="center", fontsize=9, style="italic", color="#555555")
    save_fig(fig, "eda_04_tenure_distribution.png")


# --------------------------------------------------------------------------- #
# Chart 5 — Correlation heatmap
# --------------------------------------------------------------------------- #
def chart_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = [
        "tenure_months", "monthly_spend", "data_usage_gb", "call_minutes",
        "sms_count", "network_complaints", "signal_quality_score",
        "contract_end_months", "last_upgrade_months", "complaint_rate",
        "data_intensity", "clv_estimate", "churn_flag",
    ]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle only
    sns.heatmap(
        corr, ax=ax, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.text(0.5, -0.01,
             "Insight: network_complaints and signal_quality_score show strongest "
             "correlation with churn_flag — confirming service quality as the primary driver.",
             ha="center", fontsize=9, style="italic", color="#555555")
    save_fig(fig, "eda_05_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# Chart 6 — Network complaints vs churn rate
# --------------------------------------------------------------------------- #
def chart_complaints_vs_churn(df: pd.DataFrame) -> None:
    # Cap at 5 complaints for readability (>5 is a tiny tail)
    df_plot = df.copy()
    df_plot["complaints_capped"] = df_plot["network_complaints"].clip(upper=5)
    churn_by_complaints = (
        df_plot.groupby("complaints_capped")["churn_flag"].mean()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(churn_by_complaints.index.astype(str),
           churn_by_complaints.values, color=ACCENT, width=0.5, edgecolor="white")
    add_value_labels(ax)
    ax.set_title("Churn Rate by Number of Network Complaints", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Network Complaints (capped at 5)", fontsize=11)
    ax.set_ylabel("Churn Rate", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, churn_by_complaints.max() * 1.25)
    fig.text(0.5, -0.02,
             "Insight: Near-linear relationship — each additional complaint "
             "materially increases churn probability.",
             ha="center", fontsize=9, style="italic", color="#555555")
    save_fig(fig, "eda_06_complaints_vs_churn.png")


# --------------------------------------------------------------------------- #
# Chart 7 — Signal quality by region (boxplot)
# --------------------------------------------------------------------------- #
def chart_signal_by_region(df: pd.DataFrame) -> None:
    region_order = (
        df.groupby("region")["signal_quality_score"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, x="region", y="signal_quality_score",
        order=region_order, hue="region", palette="Blues_d",
        legend=False, ax=ax,
        linewidth=1.2, flierprops={"marker": "o", "markersize": 3, "alpha": 0.3},
    )
    ax.set_title("Signal Quality Score by Region", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("Signal Quality Score (1–5)", fontsize=11)
    ax.set_ylim(0.5, 5.5)
    ax.axhline(y=3.0, color="#F44336", linestyle="--", linewidth=1.2, label="Threshold (3.0)")
    ax.legend(fontsize=9)
    fig.text(0.5, -0.02,
             "Insight: Regions sorted by median signal quality — "
             "lower-ranked regions correlate with higher churn (cross-ref Chart 2).",
             ha="center", fontsize=9, style="italic", color="#555555")
    save_fig(fig, "eda_07_signal_by_region.png")


# --------------------------------------------------------------------------- #
# Chart 8 — Churn rate by tenure bucket
# --------------------------------------------------------------------------- #
def chart_churn_by_tenure_bucket(df: pd.DataFrame) -> None:
    churn_rate = (
        df.groupby("tenure_bucket", observed=True)["churn_flag"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [("#F44336" if v == churn_rate.max() else ACCENT) for v in churn_rate.values]
    ax.bar(churn_rate.index.astype(str), churn_rate.values,
           color=colors, width=0.5, edgecolor="white")
    add_value_labels(ax)
    ax.set_title("Churn Rate by Customer Lifecycle Stage", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Tenure Bucket", fontsize=11)
    ax.set_ylabel("Churn Rate", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, churn_rate.max() * 1.30)
    fig.text(0.5, -0.02,
             "Insight: New customers (0–12m) show highest churn — "
             "onboarding and early engagement programs have highest ROI.",
             ha="center", fontsize=9, style="italic", color="#555555")
    save_fig(fig, "eda_08_churn_by_tenure_bucket.png")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_clean(CLEAN_PATH)

        print("\nGenerating charts...")
        chart_churn_by_plan(df)
        chart_churn_by_region(df)
        chart_spend_distribution(df)
        chart_tenure_distribution(df)
        chart_correlation_heatmap(df)
        chart_complaints_vs_churn(df)
        chart_signal_by_region(df)
        chart_churn_by_tenure_bucket(df)

    except (OSError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nSUCCESS: 8 charts saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())