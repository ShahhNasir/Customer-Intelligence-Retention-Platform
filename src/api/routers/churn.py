"""
Churn prediction endpoint. Scores a real customer from Postgres using the
trained XGBoost model, persists the result as an audit trail, and returns
it - the "identified high-risk customers for targeted interventions"
resume claim, made concrete.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import PredictionResponse
from src.db.database import get_db
from src.db.models import Customer, Prediction
from src.ml.features import FEATURE_COLUMNS
from src.ml.predict import predict_churn

router = APIRouter(prefix="/customers", tags=["churn"])


@router.post("/{customer_id}/predict", response_model=PredictionResponse)
def predict_customer_churn(customer_id: int, db: Session = Depends(get_db)):
    customer = db.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    # Pull exactly the columns the model expects, straight off the ORM
    # object - FEATURE_COLUMNS is the same list train_churn_model.py used,
    # imported from the shared src/ml/features.py (see Phase 7 notes on
    # why that refactor happened).
    features = {col: getattr(customer, col) for col in FEATURE_COLUMNS}
    result = predict_churn(features)

    prediction = Prediction(
        customer_id=customer_id,
        churn_probability=result["churn_probability"],
        predicted_label=result["predicted_label"],
        threshold_used=result["threshold_used"],
        model_version=result["model_version"],
        created_at=datetime.utcnow(),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)  # populates prediction.prediction_id from the DB sequence

    return PredictionResponse(
        prediction_id=prediction.prediction_id,
        customer_id=customer_id,
        churn_probability=prediction.churn_probability,
        predicted_label=prediction.predicted_label,
        threshold_used=prediction.threshold_used,
        model_version=prediction.model_version,
        created_at=prediction.created_at,
    )
