"""
Retention recommendation endpoint. Only ever called against an EXISTING
prediction (not a bare customer_id) - a recommendation is generated
IN RESPONSE TO a churn-risk assessment, which is exactly why
Recommendation is foreign-keyed to Prediction in the schema (see
docs/notes.md Phase 4).
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import RecommendationRequest, RecommendationResponse
from src.config import settings
from src.db.database import get_db
from src.db.models import Prediction, Recommendation
from src.rag.generate_recommendation import generate_recommendation

router = APIRouter(prefix="/predictions", tags=["recommendations"])


@router.post("/{prediction_id}/recommend", response_model=RecommendationResponse)
def create_recommendation(
    prediction_id: int, request: RecommendationRequest, db: Session = Depends(get_db)
):
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    result = generate_recommendation(
        customer_id=prediction.customer_id, query=request.query, top_k=request.top_k
    )

    # Persisted regardless of guardrail outcome - passed_guardrails and
    # guardrail_reason are what let a caller (or a human reviewer) know
    # NOT to trust a failed recommendation, rather than it silently
    # vanishing with no audit trail.
    recommendation = Recommendation(
        prediction_id=prediction_id,
        recommendation_text=result.recommendation_text,
        passed_guardrails=result.passed_guardrails,
        guardrail_reason=result.guardrail_reason,
        llm_provider=settings.llm_provider,
        created_at=datetime.utcnow(),
    )
    db.add(recommendation)
    db.commit()
    db.refresh(recommendation)

    return RecommendationResponse(
        recommendation_id=recommendation.recommendation_id,
        prediction_id=prediction_id,
        customer_id=prediction.customer_id,
        recommendation_text=recommendation.recommendation_text,
        passed_guardrails=recommendation.passed_guardrails,
        guardrail_reason=recommendation.guardrail_reason,
        llm_provider=recommendation.llm_provider,
        created_at=recommendation.created_at,
    )
