# CodeAlpha Machine Learning Internship - Task 1

## Credit Scoring Model

A machine learning model that predicts whether a person will default on a loan within the next 2 years, based on their financial history.

---

## Problem Statement

Banks and financial institutions need to assess whether a customer is likely to default on a loan. This model takes a customer's financial and demographic data as input and predicts whether they are creditworthy.

This is a **binary classification** problem:
- `0` = Will NOT default
- `1` = Will default within 2 years

---

## Dataset

- **Source:** [Give Me Some Credit - Kaggle](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)
- **Size:** 150,000 samples, 11 columns (10 features + 1 target)
- **Target:** `SeriousDlqin2yrs` (seriously delinquent in 2 years)
- **Class imbalance:** 93.3% No Default / 6.7% Default

---

## Approach

1. Data Loading & Exploration
2. Missing Value Handling (median fill for MonthlyIncome and NumberOfDependents)
3. Train-Test Split (80/20, stratified)
4. Feature Scaling with StandardScaler
5. Model 1: Logistic Regression (baseline)
6. Model 2: Random Forest with class_weight='balanced'
7. Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
8. Feature Importance Analysis

---

## Results

| Metric | Logistic Regression | Random Forest |
|--------|--------------------:|--------------:|
| Accuracy | 0.9340 | 0.8570 |
| Precision (Default) | 0.5806 | 0.2635 |
| Recall (Default) | 0.0449 | **0.6349** |
| F1-Score (Default) | 0.0833 | 0.3724 |
| ROC-AUC | 0.7143 | **0.8557** |

### Winner: Random Forest

Random Forest achieved **14x better Recall** on the Default class. Even though its overall accuracy dropped from 93.4% to 85.7%, this is expected and desirable because:

- The dataset is highly imbalanced (93% vs 7%)
- A trivial "always predict No Default" model would already get 93% accuracy
- In credit scoring, missing a defaulter (False Negative) is far more costly to a bank than rejecting a good customer (False Positive)

---

## Top 3 Most Important Features

1. **RevolvingUtilizationOfUnsecuredLines** (31.78%)
2. **NumberOfTimes90DaysLate** (14.14%)
3. **NumberOfTime30-59DaysPastDueNotWorse** (13.15%)

---

## Technologies Used

- Python 3.14
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook (VS Code)

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter