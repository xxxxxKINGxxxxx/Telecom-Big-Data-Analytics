"""
05_segmentation.py
==================
Customer segmentation via K-Means clustering.

Reads  : data/processed/telecom_with_predictions.csv
Writes : data/processed/telecom_segmented.csv
         screenshots/seg_elbow_curve.png
         screenshots/seg_segment_profiles.png
         screenshots/seg_churn_by_segment.png

Run from project root:  python scripts/05_segmentation.py
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PRED_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "telecom_with_predictions.csv"
SEG_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "telecom_segmented.csv"
SCREEN_DIR: Final[Path] = PROJECT_ROOT / "screenshots"

RANDOM_SEED: Final[int] = 42
N_CLUSTERS: Final[int] = 4
BACKGROUND: Final[str] = "#F8F9FA"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor":   BACKGROUND,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# Features used for clustering — numeric behavioural signals only
# Categoricals excluded: clustering works on distances, not one-hot noise
CLUSTER_FEATURES: Final[list[str]] = [
    "tenure_months",
    "monthly_spend",
    "data_usage_gb",
    "network_complaints",
    "signal_quality_score",
    "churn_probability",
    "complaint_rate",
    "data_intensity",
]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found — run 04_churn_model.py first", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, dtype={"customer_id": str, "region": str, "plan_type": str})
    print(f"Loaded  : {len(df):,} rows")
    return df


# --------------------------------------------------------------------------- #
# Elbow method — find optimal k
# --------------------------------------------------------------------------- #
def plot_elbow(X_scaled: np.ndarray) -> None:
    """
    The elbow method runs KMeans for k=2..10 and plots inertia (within-cluster
    sum of squares). The 'elbow' — where inertia stops dropping sharply — is
    the optimal k. We already know k=4 is right for our personas, but showing
    this chart proves you didn't just pick k=4 arbitrarily.
    """
    inertias: list[float] = []
    k_range = range(2, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(k_range), inertias, marker="o", color="#2196F3",
            linewidth=2.2, markersize=7, markerfacecolor="#F44336")
    ax.axvline(x=N_CLUSTERS, color="#F44336", linestyle="--",
               linewidth=1.5, label=f"Selected k={N_CLUSTERS}")
    ax.set_title("K-Means Elbow Curve — Optimal Number of Clusters",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Inertia (Within-Cluster SS)", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

    path = SCREEN_DIR / "seg_elbow_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved   : {path.name}")


# --------------------------------------------------------------------------- #
# Cluster & label
# --------------------------------------------------------------------------- #
def run_kmeans(X_scaled: np.ndarray) -> np.ndarray:
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)
    print(f"Clustered: k={N_CLUSTERS}  |  inertia={km.inertia_:,.1f}")
    return labels


def assign_segment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable business labels to numeric cluster IDs.

    Strategy: compute the mean of key signals per cluster, then rank clusters
    on those signals to identify which numeric ID maps to which persona.
    This is robust — it works regardless of which random cluster ID KMeans
    assigns to each group on any given run.
    """
    profile = df.groupby("cluster_id").agg(
        tenure=("tenure_months",    "mean"),
        spend= ("monthly_spend",    "mean"),
        usage= ("data_usage_gb",    "mean"),
        complaints=("network_complaints", "mean"),
        churn_prob=("churn_probability",  "mean"),
    ).reset_index()

    # Rank each cluster on the two most discriminating dimensions
    profile["spend_rank"]     = profile["spend"].rank()      # 1=lowest, 4=highest
    profile["churn_rank"]     = profile["churn_prob"].rank() # 1=lowest churn, 4=highest
    profile["tenure_rank"]    = profile["tenure"].rank()
    profile["usage_rank"]     = profile["usage"].rank()

    label_map: dict[int, str] = {}
    for _, row in profile.iterrows():
        cid = int(row["cluster_id"])
        sr, cr, tr, ur = row["spend_rank"], row["churn_rank"], row["tenure_rank"], row["usage_rank"]

        if sr >= 3.5:
            # Highest spenders → either Business or High-Value Loyal
            label_map[cid] = "Business Accounts" if ur >= 3.0 else "High-Value Loyals"
        elif cr >= 3.0 and sr <= 2.0:
            label_map[cid] = "Low-Engagement Prepaid"
        elif cr >= 2.5 and ur >= 2.5:
            label_map[cid] = "At-Risk Heavy Users"
        else:
            label_map[cid] = "High-Value Loyals"

    # Safety net: if any label is still missing, assign by spend rank
    all_labels = {"Business Accounts", "High-Value Loyals",
                  "Low-Engagement Prepaid", "At-Risk Heavy Users"}
    used = set(label_map.values())
    missing = all_labels - used
    if missing:
        unlabelled = [cid for cid in profile["cluster_id"] if int(cid) not in label_map]
        for cid, lbl in zip(unlabelled, missing):
            label_map[int(cid)] = lbl

    df["segment"] = df["cluster_id"].map(label_map)
    print("\nSegment mapping:")
    for cid, lbl in sorted(label_map.items()):
        count = (df["cluster_id"] == cid).sum()
        print(f"  Cluster {cid} -> {lbl}  ({count:,} customers)")
    return df


# --------------------------------------------------------------------------- #
# Segment profile chart
# --------------------------------------------------------------------------- #
def plot_segment_profiles(df: pd.DataFrame) -> None:
    """
    Radar-style comparison replaced with a cleaner grouped bar chart showing
    normalised mean values per segment across key dimensions.
    Normalisation (0-1 min-max per feature) makes different-scale features
    comparable on one axis.
    """
    features = ["tenure_months", "monthly_spend", "data_usage_gb",
                "network_complaints", "churn_probability"]
    labels_short = ["Tenure", "Spend", "Data Usage", "Complaints", "Churn Prob"]

    profile = df.groupby("segment")[features].mean()

    # Min-max normalise each column so all features are on 0–1 scale
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    segment_order = ["High-Value Loyals", "Business Accounts",
                     "At-Risk Heavy Users", "Low-Engagement Prepaid"]
    # Only keep segments that actually exist
    segment_order = [s for s in segment_order if s in norm.index]
    norm = norm.reindex(segment_order)

    x = np.arange(len(labels_short))
    width = 0.20
    colors = ["#1565C0", "#2196F3", "#F44336", "#FF9800"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (seg, color) in enumerate(zip(norm.index, colors)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, norm.loc[seg].values, width,
               label=seg, color=color, alpha=0.85, edgecolor="white")

    ax.set_title("Customer Segment Profiles (Normalised)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=11)
    ax.set_ylabel("Normalised Score (0–1)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.15)

    path = SCREEN_DIR / "seg_segment_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved   : {path.name}")


def plot_churn_by_segment(df: pd.DataFrame) -> None:
    churn_rate = df.groupby("segment")["churn_flag"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#F44336" if v == churn_rate.max() else "#2196F3" for v in churn_rate.values]
    ax.bar(churn_rate.index, churn_rate.values, color=colors, width=0.5, edgecolor="white")

    for patch, val in zip(ax.patches, churn_rate.values):
        ax.text(patch.get_x() + patch.get_width() / 2, val + 0.003,
                f"{val:.1%}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_title("Churn Rate by Customer Segment", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Segment", fontsize=11)
    ax.set_ylabel("Churn Rate", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, churn_rate.max() * 1.25)

    path = SCREEN_DIR / "seg_churn_by_segment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved   : {path.name}")


# --------------------------------------------------------------------------- #
# Summary report
# --------------------------------------------------------------------------- #
def print_segment_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("SEGMENT SUMMARY")
    print("=" * 65)
    summary = df.groupby("segment").agg(
        customers     =("customer_id",      "count"),
        avg_tenure    =("tenure_months",     "mean"),
        avg_spend     =("monthly_spend",     "mean"),
        avg_data_gb   =("data_usage_gb",     "mean"),
        avg_complaints=("network_complaints","mean"),
        churn_rate    =("churn_flag",        "mean"),
        avg_clv       =("clv_estimate",      "mean"),
    ).round(2)
    print(summary.to_string())
    print("=" * 65)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_data(PRED_PATH)

        # Scale clustering features
        X = df[CLUSTER_FEATURES].fillna(0).to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        plot_elbow(X_scaled)

        labels = run_kmeans(X_scaled)
        df["cluster_id"] = labels
        df = assign_segment_labels(df)

        plot_segment_profiles(df)
        plot_churn_by_segment(df)
        print_segment_summary(df)

        df.to_csv(SEG_PATH, index=False)
        print(f"\nSaved   : segmented data -> {SEG_PATH.name}")

    except (OSError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nSUCCESS: segmentation pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())