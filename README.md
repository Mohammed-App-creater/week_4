# Credit Risk Probability Model (BNPL)

[![CI/CD](https://github.com/Mohammed-App-creater/week_4/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohammed-App-creater/week_4/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow)](https://mlflow.org/)

A production-grade, regulator-aware Credit Risk Probability Model built using alternative eCommerce data for a Buy-Now-Pay-Later (BNPL) use case. This system identifies high-risk customers using behavioral proxies and provides real-time credit scoring via a containerized REST API.

---

## 1️⃣ Project Overview

### Business Context
In the Buy-Now-Pay-Later (BNPL) industry, providing credit to "thin-file" or "unbanked" customers is a core value proposition. Traditional credit bureau data is often unavailable for these segments, necessitating the use of **alternative data** (e.g., transaction history, behavioral patterns) to assess creditworthiness.

### The Challenge
Without historical "default" labels, we must rely on behavioral proxies to identify risk. This project implements an end-to-end pipeline: from transaction-level raw data to a deployed machine learning model that predicts the probability of a customer being "high risk."

### High-Level Architecture
1.  **Data Processing**: Aggregate transaction data to customer-level features.
2.  **Proxy Labeling**: RFM analysis + K-Means clustering to define a "High Risk" proxy target.
3.  **Feature Engineering**: Temporal extraction, Scaling, and Weight of Evidence (WoE) transformation.
4.  **Modeling**: Training interpretable models (Logistic Regression) and ensemble methods (Random Forest) with MLflow tracking.
5.  **Deployment**: FastAPI service wrapped in Docker with CI/CD via GitHub Actions.

---

## 2️⃣ Credit Scoring Business Understanding

### Basel II Influence
Under **Basel II**, models used for regulatory capital calculation (Internal Ratings-Based approach) must be transparent and explainable. This implementation prioritizes interpretability (e.g., via WoE and Logistic Regression) to ensure that risk drivers can be audited and communicated to regulators.

### Proxy Target Variable (`is_high_risk`)
Traditional lending uses "90 days past due" (DPD) as a default label. In this BNPL context, we use a **proxy target** derived from financial engagement:
-   **Rationale**: Customers with low transaction frequency, low monetary value, and high recency (dormancy) are behaviorally similar to those who have stopped paying or are disengaging due to financial distress.
-   **Risks**: Proxy labels may misclassify simply "inactive" users as "high risk" (Type I error).

### Trade-offs: Interpretability vs. performance
| Model | Interpretability | Complexity | Business Fit |
| :--- | :--- | :--- | :--- |
| **Logistic Regression + WoE** | High (Industry Standard) | Low | Regulatory approval & Scorecards |
| **Gradient Boosting / RF** | Low (Black Box) | High | Maximum predictive power |

---

## 3️⃣ Dataset Description

-   **Source**: Xente eCommerce transaction dataset.
-   **Granularity**: Originally transaction-level, aggregated to customer-level for scoring.
-   **Key Features**: Transaction amount (Value), Transaction count, Recency, Frequency, Monetary (RFM), and temporal identifiers.
-   **Limitations**: Data is a snapshot; it lacks long-term historical performance and actual repayment outcome labels.

---

## 4️⃣ Exploratory Data Analysis (EDA)

-   **Structure**: The dataset contains high-frequency transactions with varied amounts.
-   **Skewness**: Transaction amounts (Value/Amount) are highly right-skewed, requiring log transformation for stable modeling.
-   **Missing Values**: Minimal missing values in core transactional fields; imputation handled via median strategy for robustness.
-   **Key Insights**:
    1.  **Extreme Outliers**: A small percentage of transactions account for the majority of the total volume.
    2.  **Class Imbalance**: Fraudulent transactions (where present) are extremely rare (<1%), mirroring the suspected distribution of default events.
    3.  **Behavioral Clusters**: Clear segmentation exists between "Power Users" and "Inactive/High-Risk" users.

---

## 5️⃣ Feature Engineering

-   **Aggregation**: Transactions are grouped by `CustomerId` to create behavioral profiles.
-   **Temporal Features**: Extraction of `hour`, `day`, and `month` from timestamps to capture cyclical trends in spending.
-   **RFM Design**:
    -   **Recency**: Days since last purchase.
    -   **Frequency**: Count of purchases.
    -   **Monetary**: Total spend.
-   **WoE & IV**: Weight of Evidence (WoE) transformation is used to linearize relationships and handle categories, while Information Value (IV) ranks feature predictive power.

---

## 6️⃣ Proxy Target Variable (RFM Clustering)

Since "default" is not explicitly labeled, we use **K-Means Clustering** on RFM metrics to segment customers.
-   **The "High Risk" Cluster**: Identified as the cluster with high average Recency (inactive) and low average Frequency/Monetary (low engagement).
-   **Assumptions**: We assume Behavioral Disengagement $\approx$ Default Risk.

---

## 7️⃣ Model Training & Evaluation

-   **Models**: Logistic Regression and Random Forest.
-   **Tuning**: `GridSearchCV` used for hyperparameter optimization (e.g., regularization strength `C`).
-   **Metrics**: ROC-AUC is the primary metric, ensuring the model can distinguish between risk classes regardless of the probability threshold.
-   **MLflow**: All experiments, parameters, and metrics are logged. The best model is registered to the MLflow Model Registry as `CreditRiskModel`.

---

## 8️⃣ Model Deployment

### FastAPI Design
The API provides a high-performance REST interface for real-time scoring.
-   **Endpoint**: `/predict` (POST)
-   **Logic**:
    1.  Receive customer features.
    2.  Model calculates probability $P(Default)$.
    3.  Convert $P$ to a score: $Score = 850 - (P \times 550)$.
-   **Example Request/Response**:
    ```json
    // POST /predict
    { "total_transaction_amount": 5000, "avg_transaction_amount": 500, "transaction_count": 10, "std_transaction_amount": 50 }

    // RESPONSE
    { "risk_probability": 0.21, "credit_score": 734, "risk_label": "LOW_RISK" }
    ```

---

## 9️⃣ CI/CD Pipeline

-   **GitHub Actions**: Automates linting (flake8/black) and unit testing (pytest) on every push.
-   **Importance**: In financial ML, CI ensures that code changes do not break scoring logic, maintaining the integrity of credit decisions.

---

## 🔟 Project Structure

```bash
├── .github/workflows/  # CI/CD pipelines
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory Data Analysis (EDA)
├── src/
│   ├── api/            # FastAPI implementation & Pydantic models
│   ├── data_processing.py # Feature engineering & Proxy labeling
│   ├── train.py        # Model training & MLflow logging
├── tests/              # Unit tests for API and Processing
├── Dockerfile          # Container definition
├── docker-compose.yml  # Multi-container orchestration (API + MLflow)
└── requirements.txt    # Production dependencies
```

---

## 1️⃣1️⃣ How to Run the Project

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Tests**:
    ```bash
    pytest
    ```
3.  **Start API (Docker)**:
    ```bash
    docker-compose up --build
    ```

---

## 1️⃣2️⃣ Limitations & Future Improvements

-   **Proxy Label Risks**: Real default data (e.g., 90+ DPD) should replace RFM clusters as soon as it becomes available.
-   **Model Drift**: Financial behavior changes over time; periodic retraining and monitoring of score stability (PSI) are required.
-   **Fairness**: Continuous auditing for disparate impact across demographic groups (if data is collected).

---

## 1️⃣3️⃣ References

-   [Basel II Framework](https://www.bis.org/publ/bcbs128.htm)
-   [World Bank: Credit Scoring for SMEs](https://www.worldbank.org/)
-   [Scikit-learn Pipelines Documentation](https://scikit-learn.org/stable/modules/compose.html)
