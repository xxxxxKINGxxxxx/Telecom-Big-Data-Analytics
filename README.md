# Telecom Big Data Analytics Platform

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

> An end-to-end enterprise-grade telecom analytics platform simulating 
> real-world customer intelligence operations — churn prediction, customer 
> segmentation, regional revenue forecasting, and AI-generated executive 
> reporting.

---

## Business Problem

Telecom operators lose 15–25% of their customer base annually to churn, 
representing millions in lost recurring revenue. Without predictive 
intelligence, retention efforts are reactive, expensive, and poorly targeted.

This platform simulates the analytics infrastructure an operator like Orange 
would deploy to: identify at-risk customers before they leave, understand 
customer value by segment, diagnose regional performance gaps, and deliver 
AI-generated executive intelligence automatically.

---

## Key Results

| Metric | Value |
|--------|-------|
| Customers analysed | 50,000 |
| Churn rate detected | 16.58% |
| Model ROC-AUC | 0.72 |
| Churners correctly identified | 1,166 of 1,658 |
| Net retention campaign value | **€254,203** |
| High-risk customers flagged | 7,095 (14.2%) |
| Average ARPU | €38.20/month |
| Average CLV | €933.66 |

---

## Architecture

Layer 1 — Data Generation
└── 50,000 synthetic customer records with realistic churn drivers
Layer 2 — ETL Cleaning Pipeline
└── Null imputation, outlier capping, deduplication, feature engineering
Layer 3 — Analytics & ML
├── Churn prediction (Logistic Regression + Random Forest)
├── Customer segmentation (K-Means, k=4)
└── Revenue forecasting (linear trend extrapolation)
Layer 4 — Output & Reporting
├── Power BI 4-page executive dashboard
└── AI-generated executive summary (Anthropic Claude API)

---

## Dashboard Pages

### Page 1 — Executive Overview
KPI cards (total customers, churn rate, ARPU), risk distribution donut, 
churn rate by region.

### Page 2 — Churn Intelligence  
Churn by plan type, churn by customer lifecycle stage, predicted churn 
distribution.

### Page 3 — Customer Segments
4 K-Means personas profiled by count, CLV, churn rate, and monthly spend.

### Page 4 — Regional Performance
Signal quality by region, churn rate by region, ARPU by region, 
3-month revenue forecast.

---

## Customer Segments Discovered

| Segment | Customers | Churn Rate | Avg CLV |
|---------|-----------|------------|---------|
| Low-Engagement Prepaid | 21,354 | 16% | €409 |
| High-Value Loyals | 11,570 | 5% | €1,539 |
| At-Risk Heavy Users | 9,127 | 39% | €442 |
| Business Accounts | 7,949 | 10% | €2,027 |

---

## Key Findings

- **West and East regions** show 21% and 20% churn — driven by below-average 
  signal quality scores (median 3.4 vs 4.0 in North)
- **New customers (0–12 months)** churn at 24.9% — 4× the rate of Loyal 
  customers (6.6%)
- **Each additional network complaint** increases churn probability by ~15 
  percentage points (from 11.6% at 0 complaints to 100% at 5)
- **Prepaid customers** churn at 20.2% vs 12.7% for Business — no contract 
  lock-in is the primary driver
- **At-Risk Heavy Users** are engaged customers being lost to service quality 
  failures, not price sensitivity

---

## Business Recommendations

1. Deploy retention campaigns targeting 7,095 High Risk customers — 
   estimated **€254,203 net value** at 30% conversion
2. Prioritise network infrastructure investment in West and East regions
3. Introduce contract migration incentives for Prepaid customers
4. Implement 90-day onboarding programme for all new customers
5. Assign dedicated account management to Business Account segment

---

## Technologies Used

| Category | Technologies |
|----------|-------------|
| Language | Python 3.14 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Logistic Regression, Random Forest, K-Means) |
| Visualisation | Matplotlib, Seaborn, Power BI Desktop |
| AI Integration | Anthropic Claude API |
| Version Control | Git, GitHub |
| Environment | python-dotenv |

---

## Project Structure

telecom-big-data-analytics/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── scripts/
│   ├── 01_generate_dataset.py
│   ├── 02_data_cleaning.py
│   ├── 03_eda_analysis.py
│   ├── 04_churn_model.py
│   ├── 05_segmentation.py
│   ├── 06_forecasting.py
│   └── 07_ai_insights.py
├── dashboard/
│   └── telecom_dashboard.pbix
├── report/
│   ├── executive_summary.md
│   └── model_evaluation.md
└── screenshots/

---

## Setup & Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/telecom-big-data-analytics.git
cd telecom-big-data-analytics

# Install dependencies
python -m pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your Anthropic API key to .env

# Run the full pipeline in order
python scripts/01_generate_dataset.py
python scripts/02_data_cleaning.py
python scripts/03_eda_analysis.py
python scripts/04_churn_model.py
python scripts/05_segmentation.py
python scripts/06_forecasting.py
python scripts/07_ai_insights.py
```

---

## Future Improvements

- Real-time churn scoring via Apache Kafka stream ingestion
- Azure Synapse deployment for production-scale processing
- AutoML hyperparameter optimisation (Optuna)
- Power BI embedded publishing with row-level security
- SHAP explainability layer for model interpretability

---

## Target Industry

This project is designed to demonstrate enterprise analytics capability 
relevant to telecom operators (Orange), HR analytics platforms (SD Worx, 
Dayforce), and enterprise data teams requiring scalable BI + ML + AI 
integration.

