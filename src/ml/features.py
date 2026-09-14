"""
Single source of truth for which columns feed the churn model, and how
they're split into categorical vs numeric. Imported by BOTH
train_churn_model.py (training) and the FastAPI prediction endpoint
(serving) - duplicating these lists in two places would risk them
silently drifting apart, which is exactly the training/serving
consistency problem this project has been careful about since Phase 3's
encoder design.
"""

FEATURE_COLUMNS = [
    "tenure_months",
    "contract_type",
    "monthly_charges",
    "total_charges",
    "payment_method",
    "internet_service",
    "tech_support",
    "online_security",
    "paperless_billing",
    "num_support_tickets",
    "avg_satisfaction_score",
    "clv_segment",
]
CATEGORICAL_COLUMNS = [
    "contract_type",
    "payment_method",
    "internet_service",
    "tech_support",
    "online_security",
    "paperless_billing",
    "clv_segment",
]
NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "avg_satisfaction_score",
]
