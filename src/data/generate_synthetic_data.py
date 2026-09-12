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
        - 0.3  # intercept: tunes baseline churn rate (calibrated to ~28%)
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


INTERACTION_TEMPLATES = {
    "billing_issue": [
        "I was charged ${amount} but my plan should only cost ${expected}. Can you explain this?",
        "My bill this month is higher than usual. I don't understand the extra ${amount} charge.",
        "Please refund the ${amount} duplicate charge from {date}.",
    ],
    "technical_problem": [
        "My {service} has been down since {date}. This is affecting my work.",
        "I'm getting constant connection drops with my {service} service.",
        "The {service} outage lasted for hours and no one from support followed up.",
    ],
    "complaint": [
        "I've called {n} times about the same issue and nothing has been resolved.",
        "Very disappointed with the service quality over the past {months} months.",
        "This is the third time I've had to explain my issue to a different agent.",
    ],
    "cancellation_request": [
        "I want to cancel my subscription. It's no longer worth the ${amount}/month.",
        "Please cancel my account. I've found a better deal elsewhere.",
        "I'm switching providers due to the repeated {service} issues.",
    ],
    "positive_feedback": [
        "Just wanted to say the support agent today was extremely helpful!",
        "Really happy with the {service} upgrade, works much better now.",
        "Quick resolution on my last ticket, appreciate the fast response.",
    ],
    "general_inquiry": [
        "What are the differences between the current plan and the premium tier?",
        "Does my plan include {service}? Just want to confirm before renewing.",
        "How do I update my payment method on file?",
    ],
}

# Which sentiment each interaction_type tends toward - a probability
# distribution, not a fixed mapping, so there's still realistic variation
# (e.g. a billing_issue can occasionally resolve into neutral, not just negative).
SENTIMENT_BY_TYPE = {
    "billing_issue": {"negative": 0.7, "neutral": 0.25, "positive": 0.05},
    "technical_problem": {"negative": 0.65, "neutral": 0.3, "positive": 0.05},
    "complaint": {"negative": 0.85, "neutral": 0.13, "positive": 0.02},
    "cancellation_request": {"negative": 0.9, "neutral": 0.08, "positive": 0.02},
    "positive_feedback": {"negative": 0.02, "neutral": 0.08, "positive": 0.9},
    "general_inquiry": {"negative": 0.1, "neutral": 0.8, "positive": 0.1},
}

SERVICES = ["internet", "fiber", "streaming add-on", "tech support plan", "mobile hotspot"]


def generate_interactions(customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    interaction_id = 1

    for cust_id, tenure, n_tickets in zip(
        customers["customer_id"], customers["tenure_months"], customers["num_support_tickets"]
    ):
        if n_tickets == 0:
            continue  # most customers: no interaction history at all

        # interaction_type distribution: we pick uniformly across the 6
        # types per row. High-ticket customers naturally end up with MORE
        # negative-leaning rows overall simply because they have more rows,
        # not because we hand-bias the type selection itself.
        types = rng.choice(list(INTERACTION_TEMPLATES.keys()), size=n_tickets)

        for interaction_type in types:
            sentiment_probs = SENTIMENT_BY_TYPE[interaction_type]
            sentiment = rng.choice(
                list(sentiment_probs.keys()), p=list(sentiment_probs.values())
            )

            template = rng.choice(INTERACTION_TEMPLATES[interaction_type])
            text = template.format(
                amount=rng.integers(5, 80),
                expected=rng.integers(20, 100),
                date=f"{rng.integers(1, 28)}/{rng.integers(1, 12)}",
                service=rng.choice(SERVICES),
                n=rng.integers(2, 6),
                months=rng.integers(1, 6),
            )

            days_ago = rng.integers(0, max(tenure * 30, 1))
            timestamp = datetime.now() - timedelta(days=int(days_ago))

            rows.append(
                {
                    "interaction_id": interaction_id,
                    "customer_id": cust_id,
                    "timestamp": timestamp,
                    "channel": rng.choice(["chat", "email", "phone"], p=[0.5, 0.3, 0.2]),
                    "interaction_type": interaction_type,
                    "sentiment": sentiment,
                    "text": text,
                }
            )
            interaction_id += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    customers = generate_customers(N_CUSTOMERS)

    print(customers.head())
    print("\nShape:", customers.shape)
    print("\nChurn rate:", customers["churned"].mean().round(4))
    print("\nClass balance:\n", customers["churned"].value_counts())

    customers.to_csv("data/raw/customers.csv", index=False)
    print("\nSaved to data/raw/customers.csv")

    interactions = generate_interactions(customers)
    print("\nInteractions shape:", interactions.shape)
    print(interactions.head())
    print(
        "\nSentiment distribution:\n",
        interactions["sentiment"].value_counts(normalize=True).round(3),
    )

    interactions.to_csv("data/raw/interactions.csv", index=False)
    print("\nSaved to data/raw/interactions.csv")
