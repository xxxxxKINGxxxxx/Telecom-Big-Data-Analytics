"""
07_ai_insights.py
=================
AI-generated executive summary for the Telecom Big Data Analytics Platform.

Reads KPIs from processed data, assembles a structured prompt, calls the
Anthropic API, and saves the response to report/executive_summary.md.

Run from project root:  python scripts/07_ai_insights.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import anthropic
import pandas as pd
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SEG_PATH: Final[Path]  = PROJECT_ROOT / "data" / "processed" / "telecom_segmented.csv"
FORE_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "revenue_forecast.csv"
REPORT_PATH: Final[Path] = PROJECT_ROOT / "report" / "executive_summary.md"

# --------------------------------------------------------------------------- #
# Load environment
# --------------------------------------------------------------------------- #
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------------------------------- #
# Build KPI snapshot from processed data
# --------------------------------------------------------------------------- #
def build_kpi_snapshot(seg_df: pd.DataFrame, fore_df: pd.DataFrame) -> str:
    """
    Extract the most important KPIs from the processed datasets and format
    them as a structured text block for the prompt.
    """
    total = len(seg_df)
    churn_rate = seg_df["churn_flag"].mean()
    avg_arpu = seg_df["monthly_spend"].mean()
    avg_clv = seg_df["clv_estimate"].mean()
    high_risk = (seg_df["risk_segment"] == "High Risk").sum()
    high_value_churners = seg_df[
        (seg_df["is_high_value"] == 1) & (seg_df["churn_predicted"] == 1)
    ].shape[0]

    # Regional churn
    regional_churn = (
        seg_df.groupby("region")["churn_flag"]
        .mean()
        .sort_values(ascending=False)
    )

    # Segment summary
    segment_summary = (
        seg_df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            churn_rate=("churn_flag", "mean"),
            avg_spend=("monthly_spend", "mean"),
            avg_clv=("clv_estimate", "mean"),
        )
        .round(2)
    )

    # Forecast — last historical vs month+3 per region
    forecast_lines: list[str] = []
    for region in sorted(fore_df["region"].unique()):
        rdf = fore_df[fore_df["region"] == region]
        last_hist = rdf[~rdf["is_forecast"]]["revenue"].iloc[-1]
        last_fore = rdf[rdf["is_forecast"]]["revenue"].iloc[-1]
        direction = "▲" if last_fore > last_hist else "▼"
        forecast_lines.append(
            f"  {region}: current €{last_hist/1000:.1f}k → forecast €{last_fore/1000:.1f}k {direction}"
        )

    snapshot = f"""
TELECOM BIG DATA ANALYTICS PLATFORM — KPI SNAPSHOT
====================================================

OVERALL METRICS
- Total customers analysed : {total:,}
- Overall churn rate       : {churn_rate:.2%}
- Average ARPU             : €{avg_arpu:.2f}/month
- Average CLV              : €{avg_clv:.2f}
- High-risk customers      : {high_risk:,} ({high_risk/total:.1%} of base)
- High-value customers at risk of churn : {high_value_churners:,}

CHURN RATE BY REGION (highest to lowest)
{chr(10).join(f"  {r}: {v:.2%}" for r, v in regional_churn.items())}

CUSTOMER SEGMENTS
{segment_summary.to_string()}

3-MONTH REVENUE FORECAST BY REGION
{chr(10).join(forecast_lines)}

MODEL PERFORMANCE (Random Forest)
- ROC-AUC  : 0.7235
- Recall   : 70.3% (captures 7 in 10 churners)
- Net value delivered by retention campaign : €254,203
"""
    return snapshot.strip()


# --------------------------------------------------------------------------- #
# Call Anthropic API
# --------------------------------------------------------------------------- #
def generate_executive_summary(kpi_snapshot: str) -> str:
    """
    Send the KPI snapshot to Claude and request a structured executive summary.
    Uses claude-haiku-3-5 for speed and cost efficiency — this is a
    text-generation task, not a reasoning task, so Haiku is the right model.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    system_prompt = """You are a senior analytics consultant preparing an executive 
summary for the Chief Commercial Officer of a major European telecom operator.

Your writing style is:
- Direct, confident, and data-driven
- Free of jargon and academic language  
- Focused on business impact and actionable recommendations
- Professional but not stiff

You always lead with the most important finding, quantify impact in euros where 
possible, and end with prioritised recommendations."""

    user_prompt = f"""Based on the following KPI snapshot from our telecom analytics 
platform, write a professional executive summary with exactly three sections:

1. **Situation Summary** — What the data tells us about the current state of 
   the customer base. Lead with the most alarming finding.

2. **Key Risk Areas** — The two or three specific areas requiring immediate 
   attention, with supporting data points and estimated financial impact.

3. **Recommended Actions** — Four prioritised, specific actions the business 
   should take in the next 90 days. Each action must include an expected outcome.

Keep the total length to 350–450 words. Write for a CCO who reads this in 
90 seconds between meetings.

KPI SNAPSHOT:
{kpi_snapshot}"""

    print("Calling Anthropic API...")
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system=system_prompt,
    )

    response_text = message.content[0].text
    tokens_used = message.usage.input_tokens + message.usage.output_tokens
    print(f"API call complete  |  tokens used: {tokens_used:,}")
    return response_text


# --------------------------------------------------------------------------- #
# Save report
# --------------------------------------------------------------------------- #
def save_report(summary: str, kpi_snapshot: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# Executive Summary
*Telecom Big Data Analytics Platform — AI-Generated Insights*
*Model: claude-haiku-4-5 | Generated by 07_ai_insights.py*

---

{summary}

---

## Raw KPI Snapshot (Input to AI)

"""
    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"Saved   : {REPORT_PATH}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    # Validate inputs exist
    for path in (SEG_PATH, FORE_PATH):
        if not path.exists():
            print(f"ERROR: {path.name} not found — run previous scripts first",
                  file=sys.stderr)
            return 1

    try:
        print("Loading data...")
        seg_df = pd.read_csv(SEG_PATH, dtype={"customer_id": str,
                                               "region": str,
                                               "plan_type": str})
        fore_df = pd.read_csv(FORE_PATH)

        kpi_snapshot = build_kpi_snapshot(seg_df, fore_df)
        print("KPI snapshot built.")
        print("\n" + kpi_snapshot + "\n")

        summary = generate_executive_summary(kpi_snapshot)

        print("\n" + "=" * 60)
        print("AI-GENERATED EXECUTIVE SUMMARY")
        print("=" * 60)
        print(summary)
        print("=" * 60 + "\n")

        save_report(summary, kpi_snapshot)

    except anthropic.AuthenticationError:
        print("ERROR: Invalid API key — check your .env file", file=sys.stderr)
        return 1
    except anthropic.RateLimitError:
        print("ERROR: Rate limit hit — wait 60 seconds and retry", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("ERROR: No internet connection or API unreachable", file=sys.stderr)
        return 1
    except (OSError, PermissionError) as exc:
        print(f"ERROR: File I/O failure\n  -> {exc}", file=sys.stderr)
        return 1

    print("SUCCESS: executive summary written to report/executive_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())