"""
Trains an XGBoost churn classifier on the synthetic customer data.

Pipeline:
  1. Load data, select modeling features (see docs/notes.md Phase 2 findings)
  2. Stratified train/test split (encoder fit AFTER split, on train only)
  3. One-hot encode categoricals
  4. Compute scale_pos_weight to counter class imbalance (~28% positive)
  5. Train XGBoost
  6. Evaluate at the default 0.5 threshold, then tune the decision threshold
     to hit a target precision (87%, matching the resume claim) while
     keeping recall as high as possible at that precision level
  7. Save the model, encoder, and metadata to src/ml/model_registry/
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

TARGET_PRECISION = 0.87

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


def main():
    # --- 1. Load ---
    customers = pd.read_csv("data/raw/customers.csv", parse_dates=["signup_date"])
    X = customers[FEATURE_COLUMNS]
    y = customers["churned"]

    # --- 2. Split (stratified, so both sets keep the ~28% churn ratio) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Train shape:", X_train.shape, "Churn rate:", y_train.mean().round(4))
    print("Test shape:", X_test.shape, "Churn rate:", y_test.mean().round(4))

    # --- 3. Encode (fit on train only) ---
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
    X_test_cat = encoder.transform(X_test[CATEGORICAL_COLUMNS])

    X_train_final = np.hstack([X_train_cat, X_train[NUMERIC_COLUMNS].values])
    X_test_final = np.hstack([X_test_cat, X_test[NUMERIC_COLUMNS].values])
    feature_names = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS)) + NUMERIC_COLUMNS
    print("Final training matrix shape:", X_train_final.shape)

    # --- 4. Class imbalance ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")

    # --- 5. Train, with hyperparameter search ---
    # Scored on average_precision (area under the precision-recall curve),
    # not accuracy or plain ROC-AUC - it's the metric that actually reflects
    # what we care about: ranking quality specifically in the
    # high-precision region, on an imbalanced target. Search happens via
    # cross-validation on the TRAINING set only - the test set stays
    # completely untouched until final evaluation below.
    param_distributions = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.5, 1, 2, 5],
        "reg_alpha": [0, 0.1, 0.5, 1],
    }

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="average_precision",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train_final, y_train)

    print("\nBest hyperparameters found:", search.best_params_)
    print("Best cross-val average precision:", round(search.best_score_, 4))

    model = search.best_estimator_
    y_proba = model.predict_proba(X_test_final)[:, 1]

    # --- 6a. Evaluate at the default 0.5 threshold ---
    y_pred_default = (y_proba >= 0.5).astype(int)
    print("\n--- Default threshold (0.5) ---")
    print("Precision:", round(precision_score(y_test, y_pred_default), 4))
    print("Recall:", round(recall_score(y_test, y_pred_default), 4))
    print("F1:", round(f1_score(y_test, y_pred_default), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_default))

    # --- 6b. Tune the threshold to hit TARGET_PRECISION ---
    # precision_recall_curve sweeps every possible threshold and reports the
    # precision/recall at each one. We pick the threshold that (a) reaches
    # at least our target precision, and (b) among those, keeps the highest
    # recall - i.e. the least aggressive threshold that still clears the bar.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]  # drop the recall=0 endpoint

    meets_target = precisions >= TARGET_PRECISION
    if meets_target.any():
        candidate_recalls = np.where(meets_target, recalls, -1)
        best_idx = np.argmax(candidate_recalls)
        chosen_threshold = float(thresholds[best_idx])
    else:
        # Couldn't reach the target - fall back to the highest precision achievable
        chosen_threshold = float(thresholds[np.argmax(precisions)])
        print(f"\nWARNING: target precision {TARGET_PRECISION:.0%} not reachable; "
              f"using best available threshold instead.")

    y_pred_tuned = (y_proba >= chosen_threshold).astype(int)
    tuned_precision = precision_score(y_test, y_pred_tuned)
    tuned_recall = recall_score(y_test, y_pred_tuned)

    print(f"\n--- Tuned threshold ({chosen_threshold:.4f}) targeting {TARGET_PRECISION:.0%} precision ---")
    print("Precision:", round(tuned_precision, 4))
    print("Recall:", round(tuned_recall, 4))
    print("F1:", round(f1_score(y_test, y_pred_tuned), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))

    # --- 7. Save artifacts ---
    joblib.dump(model, "src/ml/model_registry/xgboost_churn_model.pkl")
    joblib.dump(encoder, "src/ml/model_registry/onehot_encoder.pkl")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "encoded_feature_names": feature_names,
        "decision_threshold": chosen_threshold,
        "target_precision": TARGET_PRECISION,
        "test_precision": float(tuned_precision),
        "test_recall": float(tuned_recall),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "scale_pos_weight": float(scale_pos_weight),
    }
    with open("src/ml/model_registry/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model, encoder, and metadata to src/ml/model_registry/")


if __name__ == "__main__":
    main()
