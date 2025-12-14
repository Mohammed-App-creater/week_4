# Credit Risk Probability Model

## Credit Scoring Business Understanding

### Regulatory Compliance and Model Transparency (Basel II)
The Basel II Capital Accord fundamentally shifts credit risk management from simple capital allocation to a risk-sensitive framework. Under the Internal Ratings-Based (IRB) approach, financial institutions must calculate regulatory capital based on precise estimates of probability of default (PD), loss given default (LGD), and exposure at default (EAD). This regulatory environment mandates that credit models be not only predictive but also fully transparent, audit-ready, and interpretable. A "black box" model, regardless of its accuracy, poses a significant compliance risk because it prevents regulators and internal auditors from validating the specific drivers of capital requirements. Therefore, interpretability is not merely a technical preference; it is a prerequisite for ensuring capital adequacy and adhering to strict governance standards.

### Defining Default with Alternative Data
Unlike traditional lending products with established historical performance, this Buy-Now-Pay-Later (BNPL) service relies on transactional behavioral data where a strict "default" label is often absent or delayed. Consequently, we must engineer a proxy target variable using Recency, Frequency, and Monetary (RFM) analysis and days-past-due metrics to infer creditworthiness. While this allows for modeling in the absence of bureau data, it introduces specific business and ethical risks. Behavior that mimics default (e.g., a dormant user) may be distinct from actual insolvency, leading to potential Type I errors (rejecting good customers) or Type II errors (lending to defaulters). Furthermore, relying on behavioral proxies requires rigorous validation to ensure the model does not inadvertently encode biases present in spending patterns, which could violate fair lending practices.

### Model Selection: Interpretability vs. Complexity
In a regulated financial environment, the choice between a simple model (e.g., Logistic Regression with Weight of Evidence) and a complex ensemble model (e.g., Gradient Boosting) involves distinct trade-offs.
*   **Logistic Regression** offers superior explainability and is the industry standard for regulatory approval. Its coefficients are directly mappable to a scorecard, making it easy to explain adverse actions to customers and auditors. However, it may struggle to capture complex, non-linear relationships in high-dimensional alternative data.
*   **Ensemble Methods (e.g., Random Forest, XGBoost)** typically provide higher predictive performance and better differentiate risky borrowers. However, they present significant governance challenges regarding "reason codes" for loan denials and stability over time.
*   **Strategic Decision**: For this project, we prioritize a balance where model complexity is introduced only if the performance lift significantly outweighs the operational risk and cost of explainability (e.g., using SHAP values for governance). This ensures the model remains robust, compliant, and defensible.

## Project Structure

```
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA and prototyping
├── src/                # Source code for scripts and modules
│   ├── data/           # Data processing scripts
│   ├── features/       # Feature engineering scripts
│   ├── models/         # Model training and evaluation scripts
│   └── visualization/  # Plotting scripts
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/credit-risk-bnpl.git
    cd credit-risk-bnpl
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## deployment

### Docker (Recommended)

To run the API in a containerized environment (ensures reproducibility):

1.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```
    The API will be available at `http://localhost:8000`.

2.  **Test the API**:
    Visit `http://localhost:8000/docs` for the interactive Swagger UI, or run:
    ```bash
    curl http://localhost:8000/health
    ```

### Local Development

1.  **Train the Model**:
    The training pipeline is defined in `src/models/train_model.py`.
    ```bash
    python src/models/train_model.py
    ```
    This will save the trained model to `src/models/model.pkl`.

2.  **Run the API**:
    ```bash
    uvicorn src.app:app --reload
    ```

