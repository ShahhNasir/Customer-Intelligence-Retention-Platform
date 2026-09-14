"""
Guardrails against hallucination - checked before a generated
recommendation is ever shown to a user or persisted to the database.

Three layers, cheapest first, short-circuiting on the first failure:
  1. Structural checks (no API call): non-empty, sane length.
  2. Numeric grounding (no API call): any dollar amount mentioned in the
     recommendation must actually appear in the retrieved context -
     otherwise the model likely invented a specific number, a classic
     hallucination pattern.
  3. LLM-as-judge groundedness check (one extra Groq call): asks the
     model itself whether the recommendation introduces any claim not
     supported by the given context.
"""

import re
from dataclasses import dataclass

from src.rag.llm_client import get_llm_client

MIN_LENGTH = 20
MAX_LENGTH = 1000

DOLLAR_AMOUNT_PATTERN = re.compile(r"\$\d+(?:\.\d{2})?")


@dataclass
class GuardrailResult:
    passed: bool
    reason: str


def check_structural(recommendation: str, context_text: str = "") -> GuardrailResult:
    if not recommendation or not recommendation.strip():
        return GuardrailResult(False, "Recommendation is empty.")
    if len(recommendation) < MIN_LENGTH:
        return GuardrailResult(False, f"Recommendation too short ({len(recommendation)} chars).")
    if len(recommendation) > MAX_LENGTH:
        return GuardrailResult(False, f"Recommendation too long ({len(recommendation)} chars).")
    return GuardrailResult(True, "Structural checks passed.")


def check_numeric_grounding(recommendation: str, context_text: str) -> GuardrailResult:
    """
    Any dollar amount the recommendation claims must appear somewhere in
    the retrieved context - otherwise it was likely fabricated.
    """
    claimed_amounts = set(DOLLAR_AMOUNT_PATTERN.findall(recommendation))
    if not claimed_amounts:
        return GuardrailResult(True, "No dollar amounts claimed.")

    context_amounts = set(DOLLAR_AMOUNT_PATTERN.findall(context_text))
    ungrounded = claimed_amounts - context_amounts
    if ungrounded:
        return GuardrailResult(
            False, f"Recommendation mentions amount(s) not found in context: {sorted(ungrounded)}"
        )
    return GuardrailResult(True, "All claimed dollar amounts are grounded in context.")


def check_llm_groundedness(recommendation: str, context_text: str) -> GuardrailResult:
    """
    Standard "LLM-as-judge" verification: ask the model to check its own
    (or another call's) output against the source context.
    """
    client = get_llm_client()
    verdict = client.generate(
        system_prompt=(
            "You are a strict fact-checker for a customer retention system. "
            "You will be given CONTEXT (a customer's real interaction "
            "history) and a RECOMMENDATION written for the support team.\n\n"
            "IMPORTANT: the recommendation is ALLOWED to propose new actions "
            "not mentioned in the context (e.g. 'escalate to a specialist', "
            "'have a manager call the customer', 'review the account') - "
            "proposing a next step is the whole point of a recommendation "
            "and is NOT a hallucination.\n\n"
            "Flag it as UNGROUNDED only if it does one of these:\n"
            "(a) states a FACT about the customer or their history that "
            "isn't supported by the context (e.g. claims an action was "
            "already taken, invents a date/amount not in the context, or "
            "claims the customer said something they didn't), or\n"
            "(b) proposes a specific unauthorized concession - a discount "
            "percentage, refund amount, or guaranteed compensation - that "
            "isn't already present in the context.\n\n"
            "A general escalation, callback, or account review is GROUNDED "
            "even though the customer never asked for it by name.\n\n"
            "Reply with exactly one word on the first line: GROUNDED or "
            "UNGROUNDED. On the next line, briefly explain why in one "
            "sentence."
        ),
        user_prompt=f"CONTEXT:\n{context_text}\n\nRECOMMENDATION:\n{recommendation}",
    )
    verdict_line = verdict.strip().splitlines()[0].strip().upper()
    passed = verdict_line.startswith("GROUNDED")
    return GuardrailResult(passed, verdict.strip())


GUARDRAIL_CHECKS = (check_structural, check_numeric_grounding, check_llm_groundedness)


def run_all_guardrails(recommendation: str, context_text: str) -> GuardrailResult:
    """Runs every guardrail in order, cheapest first; stops at the first failure."""
    for check in GUARDRAIL_CHECKS:
        result = check(recommendation, context_text)
        if not result.passed:
            return result
    return GuardrailResult(True, "All guardrails passed.")
