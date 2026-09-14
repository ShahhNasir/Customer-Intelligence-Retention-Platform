"""
Provider-abstracted LLM client. The rest of the RAG pipeline calls one
function - generate() - without caring whether Groq or Anthropic answers
it. Which provider actually runs is chosen once via LLM_PROVIDER in .env.

Groq is the default (free tier, used for real usage). Anthropic is kept
as a fully working, swappable alternative - same interface, different
backend - specifically so the pattern itself is demonstrated, not just
used.
"""

from abc import ABC, abstractmethod

from src.config import settings


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the model's text completion for the given prompts."""
        raise NotImplementedError


class GroqClient(LLMClient):
    # Verified against the live model list (client.models.list()) as of
    # this project's build date - Groq's lineup changes over time, so if
    # this ever 404s, re-check available models the same way.
    MODEL = "openai/gpt-oss-20b"

    def __init__(self):
        from groq import Groq  # imported lazily - only required if this provider is actually used

        self._client = Groq(api_key=settings.groq_api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # lower temperature: we want grounded, consistent output, not creative variety
            # gpt-oss-20b is a reasoning model: part of the token budget is
            # spent on an internal reasoning trace BEFORE the final answer.
            # Too low a max_tokens can exhaust the whole budget on
            # reasoning alone, returning empty content with
            # finish_reason="length" - confirmed happening at max_tokens=300.
            max_tokens=800,
        )
        choice = response.choices[0]
        content = choice.message.content

        if not content:
            raise RuntimeError(
                f"Groq returned empty content (finish_reason={choice.finish_reason!r}). "
                "Likely exhausted max_tokens on internal reasoning before producing an answer."
            )
        return content


class AnthropicClient(LLMClient):
    MODEL = "claude-haiku-4-5-20251001"  # cheapest current Claude model

    def __init__(self):
        from anthropic import Anthropic  # imported lazily - only required if this provider is actually used

        self._client = Anthropic(api_key=settings.anthropic_api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self.MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


def get_llm_client() -> LLMClient:
    """Factory: returns the configured provider's client based on LLM_PROVIDER."""
    if settings.llm_provider == "groq":
        return GroqClient()
    elif settings.llm_provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


if __name__ == "__main__":
    client = get_llm_client()
    print(f"Using provider: {settings.llm_provider} ({client.MODEL})")

    reply = client.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say hello in exactly five words.",
    )
    print("Response:", reply)
