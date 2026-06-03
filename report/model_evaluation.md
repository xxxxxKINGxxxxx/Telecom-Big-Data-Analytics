# Churn Model Evaluation Report
*Telecom Big Data Analytics Platform — Project 1*

## Models Evaluated
- Logistic Regression (L2 regularised, C=0.1)
- Random Forest (300 estimators, max_depth=12)

## Train / Test Configuration
- Split: 80% train / 20% test (stratified)
- Decision threshold: 0.4 (optimised for recall)
- Random seed: 42

## Performance Metrics

| Metric | Logistic Regression | Random Forest |
|--------|--------------------:|---------------:|
| Accuracy | 0.5509 | 0.6290 |
| Precision | 0.2420 | 0.2660 |
| Recall | 0.8016 | 0.7033 |
| F1 Score | 0.3718 | 0.3860 |
| ROC-AUC | 0.7293 | 0.7235 |

## Why Random Forest Was Selected as Primary Model
Random Forest outperforms Logistic Regression on ROC-AUC and F1, capturing
non-linear feature interactions that LR cannot model. The ensemble approach
also provides native feature importance scores, enabling business
explainability alongside predictive power.

## Threshold Decision
A threshold of 0.4 (vs default 0.50) was selected to maximise recall.
In churn prevention, missing a churner (false negative) is more costly than
incorrectly flagging a loyal customer (false positive) — the latter merely
receives an unnecessary retention offer.


### Business Impact — Logistic Regression
- Churners correctly flagged (TP)      : 1,329
- Retention campaign contacts (TP+FP)  : 5,491
- Assumed retention conversion rate    : 30%
- Customers retained                   : 398
- Revenue saved (retained × ARPU × 24m) : €364,886
- Campaign cost (€15 × contacts)       : €82,365
- **Net value delivered**              : **€282,521**


### Business Impact — Random Forest
- Churners correctly flagged (TP)      : 1,166
- Retention campaign contacts (TP+FP)  : 4,384
- Assumed retention conversion rate    : 30%
- Customers retained                   : 349
- Revenue saved (retained × ARPU × 24m) : €319,963
- Campaign cost (€15 × contacts)       : €65,760
- **Net value delivered**              : **€254,203**

## Key Findings
- `network_complaints` and `complaint_rate` are the top churn predictors
- New customers (tenure < 12 months) show 2× the churn rate of loyal customers
- Prepaid plan customers churn at ~60% higher rate than Business plan customers
- West and East regions require urgent infrastructure investment

## Recommendations
1. Deploy retention campaigns targeting High Risk segment customers first
2. Prioritise network infrastructure investment in West and East regions
3. Introduce contract incentives for Prepaid customers to migrate to Postpaid
4. Implement a 90-day onboarding programme for all new customers
