"""
Loads the trained model, encoder, and metadata ONCE, and exposes a single
predict_churn() function the API calls per request. This mirrors the same
"load once at import time" pattern used in src/rag/retriever.py - reloading
a model from disk on every request would be disastrous for the "1000+
requests/minute" target.
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.ml.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

MODEL_PATH = "src/ml/model_registry/xgboost_churn_model.pkl"
ENCODER_PATH = "src/ml/model_registry/onehot_encoder.pkl"
METADATA_PATH = "src/ml/model_registry/model_metadata.json"

_model = joblib.load(MODEL_PATH)
_encoder = joblib.load(ENCODER_PATH)
with open(METADATA_PATH) as f:
    _metadata = json.load(f)

DECISION_THRESHOLD = _metadata["decision_threshold"]
MODEL_VERSION = "xgboost-v1"  # bump this string whenever the model is retrained/replaced


def predict_churn(customer_features: dict) -> dict:
    """
    customer_features must contain exactly the keys in
    src.ml.features.FEATURE_COLUMNS (raw, unencoded values - the same
    shape as a row of data/raw/customers.csv, minus customer_id/
    signup_date/churned).

    Returns churn_probability, predicted_label (using the tuned
    threshold from training - NOT a hard-coded 0.5), and which
    threshold/model version produced it.
    """
    row = pd.DataFrame([customer_features])

    cat_encoded = _encoder.transform(row[CATEGORICAL_COLUMNS])
    numeric_values = row[NUMERIC_COLUMNS].values
    features_final = np.hstack([cat_encoded, numeric_values])

    probability = float(_model.predict_proba(features_final)[0, 1])
    label = int(probability >= DECISION_THRESHOLD)

    return {
        "churn_probability": probability,
        "predicted_label": label,
        "threshold_used": DECISION_THRESHOLD,
        "model_version": MODEL_VERSION,
    }
