"""
06_forecasting.py
=================
Regional revenue forecasting for the Telecom Big Data Analytics Platform.

Derives a monthly revenue time-series from customer tenure cohorts,
fits a linear trend per region, and extrapolates 3 months forward.

Reads  : data/processed/telecom_segmented.csv
Writes : data/processed/revenue_forecast.csv
         screenshots/forecast_regional_revenue.png
         screenshots/forecast_summary_table.png

Run from project root:  python scripts/06_forecasting.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SEG_PATH: Final[Path]  = PROJECT_ROOT / "data" / "processed" / "telecom_segmented.csv"
FORE_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "revenue_forecast.csv"
SCREEN_DIR: Final[Path] = PROJECT_ROOT / "screenshots"

FORECAST_MONTHS: Final[int] = 3
BACKGROUND: Final[str] = "#F8F9FA"

REGION_COLORS: Final[dict[str, str]] = {
    "North":   "#1565C0",
    "South":   "#2196F3",
    "East":    "#F44336",
    "West":    "#FF9800",
    "Central": "#4CAF50",
}

plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor":   BACKGROUND,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
})


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found — run 05_segmentation.py first", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, dtype={"customer_id": str, "region": str, "plan_type": str})
    print(f"Loaded  : {len(df):,} rows")
    return df


# --------------------------------------------------------------------------- #
# Build monthly cohort revenue series
# --------------------------------------------------------------------------- #
def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a monthly revenue time series from the customer snapshot.

    Logic:
    - Each customer's tenure_months tells us how long they have been active.
    - We group customers into monthly cohorts (month 1 = newest, month N = oldest).
    - For each cohort-month we sum monthly_spend of active (non-churned) customers.
    - This gives us a 'historical' revenue curve per region.

    The assumption: customers who churned are absent from later months.
    Active customers (churn_flag=0) contribute their monthly_spend to every
    month from 1 to their tenure. Churned customers (churn_flag=1) contributed
    only up to their tenure month.

    We use months 1–36 (3 years) to keep the trend stable and avoid
    sparse data at very high tenure values.
    """
    records: list[dict] = []
    regions = df["region"].unique()

    for region in regions:
        rdf = df[df["region"] == region].copy()
        max_month = min(int(rdf["tenure_months"].max()), 36)

        for month in range(1, max_month + 1):
            # Active at this month = customers whose tenure >= month
            # AND (retained OR churned after this month)
            active = rdf[rdf["tenure_months"] >= month]
            revenue = float(active["monthly_spend"].sum())
            records.append({"region": region, "month": month, "revenue": revenue})

    series = pd.DataFrame(records)
    return series


# --------------------------------------------------------------------------- #
# Fit linear trend & forecast
# --------------------------------------------------------------------------- #
def fit_and_forecast(series: pd.DataFrame) -> pd.DataFrame:
    """
    For each region:
    1. Fit a linear regression (np.polyfit degree=1) on months 1..N.
    2. Extrapolate to months N+1, N+2, N+3.
    3. Append forecast rows with is_forecast=True flag.

    np.polyfit returns [slope, intercept]. We use the last 12 months of
    history as the fitting window — recent trend is more predictive than
    the full history which includes early ramp-up noise.
    """
    all_rows: list[pd.DataFrame] = []

    for region in series["region"].unique():
        rdf = series[series["region"] == region].copy().sort_values("month")
        rdf["is_forecast"] = False

        # Use last 12 months as fitting window
        fit_window = rdf.tail(12)
        x = fit_window["month"].to_numpy(dtype=float)
        y = fit_window["revenue"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, deg=1)

        last_month = int(rdf["month"].max())
        forecast_months = np.arange(last_month + 1, last_month + FORECAST_MONTHS + 1, dtype=float)
        forecast_revenue = slope * forecast_months + intercept

        # Clip to zero — revenue cannot be negative
        forecast_revenue = np.clip(forecast_revenue, 0.0, None)

        forecast_df = pd.DataFrame({
            "region":      region,
            "month":       forecast_months.astype(int),
            "revenue":     forecast_revenue.round(2),
            "is_forecast": True,
        })

        all_rows.append(rdf)
        all_rows.append(forecast_df)

    result = pd.concat(all_rows, ignore_index=True).sort_values(["region", "month"])

    # Compute MAE on last 3 historical months vs trend prediction
    mae_values: list[float] = []
    for region in result["region"].unique():
        hist = result[(result["region"] == region) & (~result["is_forecast"])].tail(3)
        x_val = hist["month"].to_numpy(dtype=float)
        y_actual = hist["revenue"].to_numpy(dtype=float)
        fit_window = result[
            (result["region"] == region) & (~result["is_forecast"])
        ].tail(12)
        x_fit = fit_window["month"].to_numpy(dtype=float)
        y_fit = fit_window["revenue"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
        y_pred = slope * x_val + intercept
        mae_values.append(float(np.mean(np.abs(y_actual - y_pred))))

    avg_mae = np.mean(mae_values)
    print(f"Forecast : {FORECAST_MONTHS}-month linear trend extrapolation")
    print(f"MAE      : €{avg_mae:,.0f} avg across regions (last 3 historical months)")
    return result


# --------------------------------------------------------------------------- #
# Visualisations
# --------------------------------------------------------------------------- #
def plot_regional_forecast(forecast_df: pd.DataFrame) -> None:
    """
    One subplot per region. Historical revenue = solid line.
    Forecast = dashed line with shaded confidence band (±10% of forecast value).
    """
    regions = sorted(forecast_df["region"].unique())
    n = len(regions)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for i, region in enumerate(regions):
        ax = axes_flat[i]
        rdf = forecast_df[forecast_df["region"] == region]
        hist = rdf[~rdf["is_forecast"]]
        fore = rdf[rdf["is_forecast"]]
        color = REGION_COLORS.get(region, "#2196F3")

        # Historical line
        ax.plot(hist["month"], hist["revenue"] / 1_000,
                color=color, linewidth=2.0, label="Historical")

        # Forecast line + band
        # Connect last historical point to first forecast point for continuity
        connect_x = [hist["month"].iloc[-1]] + fore["month"].tolist()
        connect_y = [hist["revenue"].iloc[-1] / 1_000] + (fore["revenue"] / 1_000).tolist()
        ax.plot(connect_x, connect_y, color=color, linewidth=2.0,
                linestyle="--", label="Forecast")

        fore_y = fore["revenue"].to_numpy() / 1_000
        fore_x = fore["month"].to_numpy()
        ax.fill_between(fore_x, fore_y * 0.90, fore_y * 1.10,
                        color=color, alpha=0.15, label="±10% band")

        ax.set_title(f"{region}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=9)
        ax.set_ylabel("Revenue (€k)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:.0f}k"))
        ax.legend(fontsize=7, loc="upper right")
        ax.set_facecolor(BACKGROUND)

    # Hide the unused 6th subplot (we have 5 regions)
    axes_flat[n].set_visible(False)

    fig.suptitle("Regional Revenue Forecast — 3-Month Extrapolation",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = SCREEN_DIR / "forecast_regional_revenue.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved   : {path.name}")


def plot_forecast_summary_table(forecast_df: pd.DataFrame) -> None:
    """
    Clean summary table: last historical month vs 3 forecast months per region.
    Exported as a matplotlib table figure for the Power BI import and README.
    """
    rows: list[dict] = []
    for region in sorted(forecast_df["region"].unique()):
        rdf = forecast_df[forecast_df["region"] == region]
        last_hist = rdf[~rdf["is_forecast"]]["revenue"].iloc[-1]
        fore_vals = rdf[rdf["is_forecast"]]["revenue"].tolist()
        rows.append({
            "Region":    region,
            "Current (€k)": f"€{last_hist/1000:.1f}k",
            "Month +1 (€k)": f"€{fore_vals[0]/1000:.1f}k",
            "Month +2 (€k)": f"€{fore_vals[1]/1000:.1f}k",
            "Month +3 (€k)": f"€{fore_vals[2]/1000:.1f}k",
            "Trend":     "▲" if fore_vals[2] > last_hist else "▼",
        })
    table_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    # Style header row
    for j in range(len(table_df.columns)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for row_idx in range(1, len(table_df) + 1):
        bg = "#E3F2FD" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(table_df.columns)):
            tbl[row_idx, col_idx].set_facecolor(bg)

    ax.set_title("3-Month Regional Revenue Forecast Summary",
                 fontsize=13, fontweight="bold", pad=16)

    path = SCREEN_DIR / "forecast_summary_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved   : {path.name}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_data(SEG_PATH)
        series = build_monthly_series(df)
        forecast_df = fit_and_forecast(series)

        plot_regional_forecast(forecast_df)
        plot_forecast_summary_table(forecast_df)

        forecast_df.to_csv(FORE_PATH, index=False)
        print(f"Saved   : forecast data -> {FORE_PATH.name}")

    except (OSError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nSUCCESS: forecasting pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())