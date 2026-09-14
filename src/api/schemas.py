"""
Pydantic request/response models. These are what FastAPI uses to
validate incoming requests and shape outgoing JSON - separate from the
SQLAlchemy models in src/db/models.py, which describe database rows.
Keeping them separate matters: not every DB column should be exposed over
the API, and not every API field maps 1:1 to a column.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction_id: int
    customer_id: int
    churn_probability: float
    predicted_label: int
    threshold_used: float
    model_version: str
    created_at: datetime


class RecommendationRequest(BaseModel):
    query: str = Field(
        default="Why might this customer churn, and what should the support team do?",
        description="The question guiding retrieval - which of the customer's own interactions to focus on.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class RecommendationResponse(BaseModel):
    recommendation_id: int
    prediction_id: int
    customer_id: int
    recommendation_text: str
    passed_guardrails: bool
    guardrail_reason: str
    llm_provider: str
    created_at: datetime
