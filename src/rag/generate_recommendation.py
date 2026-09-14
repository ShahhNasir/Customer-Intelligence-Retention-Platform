"""
End-to-end retention recommendation generation: retrieve a customer's own
interaction history (src/rag/retriever.py), generate a grounded
recommendation from it (src/rag/llm_client.py), and run the result
through hallucination guardrails (src/rag/guardrails.py) before returning.

This is the full RAG pipeline the resume bullet describes - Phase 7 wires
this into a FastAPI endpoint and persists the result to Postgres.
"""

from dataclasses import dataclass

from src.rag.guardrails import run_all_guardrails
from src.rag.llm_client import get_llm_client
from src.rag.retriever import retrieve_customer_context

SYSTEM_PROMPT = (
    "You are a customer retention specialist. You will be given a customer's "
    "actual recent support interaction history. Write ONE short, specific, "
    "actionable retention recommendation for the support team. Base your "
    "recommendation ONLY on the facts given below - do not invent details, "
    "dates, amounts, or prior actions that are not present in the provided "
    "history. Do NOT propose specific discounts, refunds, compensation "
    "percentages, or promised timelines unless the customer's history "
    "already mentions one - offering unauthorized concessions is a bigger "
    "business risk than a vague recommendation. Prefer general next steps "
    "such as escalating to a specialist, reviewing the account manually, or "
    "having a manager call the customer to discuss their specific concerns. "
    "If the history is empty or too sparse to say anything specific, "
    "recommend a general check-in instead of inventing details."
)


@dataclass
class RecommendationResult:
    customer_id: int
    recommendation_text: str
    passed_guardrails: bool
    guardrail_reason: str
    context_used: list[dict]


def _format_context(interactions: list[dict]) -> str:
    if not interactions:
        return "(No interaction history available for this customer.)"
    lines = [
        f"{i}. [{item['interaction_type']}, {item['sentiment']}] {item['text']}"
        for i, item in enumerate(interactions, 1)
    ]
    return "\n".join(lines)


def generate_recommendation(customer_id: int, query: str, top_k: int = 5) -> RecommendationResult:
    context_items = retrieve_customer_context(customer_id, query, top_k=top_k)
    context_text = _format_context(context_items)

    client = get_llm_client()
    recommendation = client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            f"Customer's interaction history:\n{context_text}\n\n"
            "Write the retention recommendation now."
        ),
    )

    guardrail_result = run_all_guardrails(recommendation, context_text)

    return RecommendationResult(
        customer_id=customer_id,
        recommendation_text=recommendation,
        passed_guardrails=guardrail_result.passed,
        guardrail_reason=guardrail_result.reason,
        context_used=context_items,
    )


if __name__ == "__main__":
    import sys

    # Windows' default console codepage (cp1252) can't print some Unicode
    # characters LLMs commonly produce (curly quotes, em-dashes, narrow
    # no-break spaces). Force UTF-8 stdout so printing generated text
    # doesn't crash - purely a display fix, unrelated to pipeline logic.
    sys.stdout.reconfigure(encoding="utf-8")

    result = generate_recommendation(
        customer_id=1,
        query="why might this customer be at risk of churning, and what should we do?",
    )
    print("Customer:", result.customer_id)
    print("\nRecommendation:\n", result.recommendation_text)
    print("\nPassed guardrails:", result.passed_guardrails)
    print("Guardrail reason:", result.guardrail_reason)
    print(f"\nContext used ({len(result.context_used)} interactions):")
    for item in result.context_used:
        print(f"  - [{item['interaction_type']}] {item['text']}")
