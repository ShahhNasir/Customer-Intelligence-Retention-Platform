"""
FastAPI application entry point. Run with:
    uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI

from src.api.routers import churn, recommendations

app = FastAPI(
    title="Customer Intelligence & Retention Platform",
    description="Churn prediction + LLM-powered retention recommendations.",
    version="1.0.0",
)

app.include_router(churn.router)
app.include_router(recommendations.router)


@app.get("/health", tags=["health"])
def health_check():
    """Basic liveness check - does NOT verify the database or model are reachable."""
    return {"status": "ok"}
