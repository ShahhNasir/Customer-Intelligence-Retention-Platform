"""
SQLAlchemy ORM models for the four tables:

  customers      - one row per customer (mirrors data/raw/customers.csv)
  interactions   - one row per support contact (mirrors interactions.csv)
  predictions    - one row per churn-risk score the API ever computes
  recommendations - one row per LLM-generated retention recommendation

Relationships: customer -> many interactions, customer -> many predictions,
prediction -> at most one recommendation.
"""

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.database import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signup_date: Mapped[date] = mapped_column(Date)
    tenure_months: Mapped[int] = mapped_column(Integer)
    contract_type: Mapped[str] = mapped_column(String(20))
    monthly_charges: Mapped[float] = mapped_column(Float)
    total_charges: Mapped[float] = mapped_column(Float)
    payment_method: Mapped[str] = mapped_column(String(30))
    internet_service: Mapped[str] = mapped_column(String(30))
    tech_support: Mapped[str] = mapped_column(String(5))
    online_security: Mapped[str] = mapped_column(String(5))
    paperless_billing: Mapped[str] = mapped_column(String(5))
    num_support_tickets: Mapped[int] = mapped_column(Integer)
    avg_satisfaction_score: Mapped[float] = mapped_column(Float)
    clv_segment: Mapped[str] = mapped_column(String(10))
    churned: Mapped[int] = mapped_column(Integer)  # ground-truth label, from training data

    interactions: Mapped[list["Interaction"]] = relationship(back_populates="customer")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="customer")


class Interaction(Base):
    __tablename__ = "interactions"

    interaction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    channel: Mapped[str] = mapped_column(String(10))
    interaction_type: Mapped[str] = mapped_column(String(30))
    sentiment: Mapped[str] = mapped_column(String(10))
    text: Mapped[str] = mapped_column(Text)

    customer: Mapped["Customer"] = relationship(back_populates="interactions")


class Prediction(Base):
    """
    One row per churn-risk score the API computes. This is the audit
    trail behind "identified high-risk customers for targeted
    interventions" - every scoring decision is durably recorded, not
    just returned and forgotten.
    """

    __tablename__ = "predictions"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    churn_probability: Mapped[float] = mapped_column(Float)
    predicted_label: Mapped[int] = mapped_column(Integer)  # 0/1, using the tuned threshold
    threshold_used: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    customer: Mapped["Customer"] = relationship(back_populates="predictions")
    recommendation: Mapped["Recommendation"] = relationship(
        back_populates="prediction", uselist=False
    )


class Recommendation(Base):
    """
    One row per LLM-generated retention recommendation. Exists only in
    response to a Prediction that flagged a customer as at-risk - hence
    the foreign key to predictions, not directly to customers.
    """

    __tablename__ = "recommendations"

    recommendation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.prediction_id"))
    recommendation_text: Mapped[str] = mapped_column(Text)
    passed_guardrails: Mapped[bool] = mapped_column(Boolean)
    guardrail_reason: Mapped[str] = mapped_column(Text)
    llm_provider: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped["Prediction"] = relationship(back_populates="recommendation")
