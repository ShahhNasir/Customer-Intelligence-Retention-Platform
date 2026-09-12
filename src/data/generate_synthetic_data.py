"""
Generates synthetic customer churn data: a customers table and an
interactions table (support tickets / chat logs), written to data/raw/.

The churn label is NOT random. It's computed from a weighted combination
of realistic churn drivers, run through a sigmoid, so a model trained on
this data has genuine, learnable (but imperfect) signal to find.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# A seeded random generator makes this script reproducible: same seed,
# same output, every run. Critical for debugging and for anyone re-running
# your project and expecting the numbers in your README to match.
rng = np.random.default_rng(seed=42)

N_CUSTOMERS = 200_000


def generate_customers(n: int) -> pd.DataFrame:
    # --- Step 1: generate the independent (causal) features first ---
    tenure_months = rng.integers(0, 73, size=n)  # 0 to 6 years

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],  # month-to-month is realistically the most common
    )

    monthly_charges = np.round(rng.normal(loc=65, scale=25, size=n).clip(18, 150), 2)

    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n,
        p=[0.35, 0.20, 0.22, 0.23],
    )

    internet_service = rng.choice(["DSL", "Fiber optic", "None"], size=n, p=[0.35, 0.45, 0.20])
    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
    online_security = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])

    # Mixture distribution for support tickets: most customers low-touch,
    # a 15% minority high-touch. This is what creates the realistic long tail.
    is_high_touch = rng.random(n) < 0.15
    num_support_tickets = np.where(
        is_high_touch,
        rng.poisson(lam=9, size=n),
        rng.poisson(lam=0.8, size=n),
    )

    # Satisfaction is itself influenced by tickets: more tickets -> lower
    # satisfaction on average, plus noise. Scale clipped to a realistic 1-5.
    avg_satisfaction_score = np.round(
        (4.2 - 0.15 * num_support_tickets + rng.normal(0, 0.6, size=n)).clip(1, 5), 2
    )

    total_charges = np.round(
        tenure_months * monthly_charges + rng.normal(0, 50, size=n), 2
    ).clip(0, None)

    # --- Step 2: compute the churn score from the causal features ---
    is_month_to_month = (contract_type == "Month-to-month").astype(int)
    is_electronic_check = (payment_method == "Electronic check").astype(int)
    has_tech_support = (tech_support == "Yes").astype(int)
    has_online_security = (online_security == "Yes").astype(int)

    score = (
        -0.045 * tenure_months
        + 0.90 * is_month_to_month
        + 0.020 * monthly_charges
        + 0.35 * num_support_tickets
        - 0.55 * avg_satisfaction_score
        + 0.45 * is_electronic_check
        - 0.30 * has_tech_support
        - 0.25 * has_online_security
        + rng.normal(0, 1.5, size=n)  # irreducible noise
        - 0.3  # intercept: tunes baseline churn rate
    )

    churn_probability = 1 / (1 + np.exp(-score))
    churned = (rng.random(n) < churn_probability).astype(int)

    # --- Step 3: derive customer lifetime value segment ---
    clv_raw = tenure_months * monthly_charges
    clv_segment = pd.cut(
        clv_raw,
        bins=[-1, 500, 2000, 5000, np.inf],
        labels=["Low", "Medium", "High", "VIP"],
    )

    signup_date = [
        (datetime.now() - timedelta(days=int(t * 30.4))).date() for t in tenure_months
    ]

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": signup_date,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "num_support_tickets": num_support_tickets,
            "avg_satisfaction_score": avg_satisfaction_score,
            "clv_segment": clv_segment,
            "churned": churned,
        }
    )


if __name__ == "__main__":
    customers = generate_customers(N_CUSTOMERS)

    print(customers.head())
    print("\nShape:", customers.shape)
    print("\nChurn rate:", customers["churned"].mean().round(4))
    print("\nClass balance:\n", customers["churned"].value_counts())

    customers.to_csv("data/raw/customers.csv", index=False)
    print("\nSaved to data/raw/customers.csv")
