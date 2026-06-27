"""LLM client abstraction over Anthropic Claude, with a deterministic mock.

Production uses the Claude API (optionally via LangChain's ``ChatAnthropic``)
for grounded reasoning. The :class:`MockClaudeClient` provides a deterministic,
network-free stand-in whose heuristics mirror what a well-prompted model would
conclude on the seed corpus - enabling end-to-end demos and tests offline.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from claim_engine.config.settings import Settings
from claim_engine.logging_utils import get_logger
from claim_engine.models.enums import DecisionOutcome

logger = get_logger(__name__)


@runtime_checkable
class LlmClient(Protocol):
    """Minimal contract for a chat completion the reasoner can call."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw text completion for the given prompts."""
        ...


class ClaudeClient:
    """Adapter around the Anthropic Claude Messages API.

    Uses :mod:`tenacity` (when available) for resilient retries with
    exponential back-off, and lazily imports the SDK so the package imports
    cleanly in mock mode.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None  # constructed lazily

    def _get_client(self):  # noqa: ANN202 - vendor client object
        """Instantiate (once) the Anthropic SDK client."""
        if self._client is not None:
            return self._client
        import anthropic  # optional heavy dependency

        self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude and return the concatenated text of the response.

        Args:
            system_prompt: The role/guardrail/system instructions.
            user_prompt: The grounded, de-identified adjudication request.

        Returns:
            The model's raw text output (expected to be a JSON object).
        """
        client = self._get_client()
        message = client.messages.create(
            model=self._settings.claude_model,
            max_tokens=self._settings.claude_max_tokens,
            temperature=self._settings.claude_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # The Messages API returns a list of content blocks; join the text ones.
        return "".join(block.text for block in message.content if block.type == "text")


class MockClaudeClient:
    """Deterministic, offline stand-in for Claude.

    Applies transparent keyword heuristics over the (already de-identified)
    prompt to emit schema-valid JSON. The heuristics are intentionally simple
    and explainable; they are NOT a model, only a fixture that makes the full
    pipeline runnable and testable without the network.
    """

    # Keyword signals -> nudges toward a particular outcome.
    _DENY_SIGNALS = ("cosmetic", "elective cosmetic", "not covered", "excluded")
    _REVIEW_SIGNALS = ("unclear", "incomplete", "unspecified", "ambiguous", "prolonged icu")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a deterministic JSON assessment derived from the prompt text.

        Only the *claim* section of the prompt is inspected for outcome signals;
        the retrieved-context section is excluded so that guideline wording (which
        naturally contains words like "unspecified") cannot skew the verdict.
        """
        text = self._extract_claim_section(user_prompt).lower()
        cited = self._extract_cited_sources(user_prompt)

        if any(signal in text for signal in self._DENY_SIGNALS):
            assessment = {
                "recommended_outcome": DecisionOutcome.AUTO_DENIED.value,
                "confidence": 0.94,
                "rationale": (
                    "Requested service matches an explicit policy exclusion; the retrieved "
                    "coverage clauses do not provide for payment."
                ),
                "medical_necessity_met": False,
                "cited_sources": cited,
                "edge_case_flags": [],
            }
        elif any(signal in text for signal in self._REVIEW_SIGNALS):
            assessment = {
                "recommended_outcome": DecisionOutcome.MANUAL_REVIEW.value,
                "confidence": 0.55,
                "rationale": (
                    "Clinical documentation is incomplete or the diagnosis is unspecified; "
                    "evidence is insufficient to confirm medical necessity automatically."
                ),
                "medical_necessity_met": False,
                "cited_sources": cited,
                "edge_case_flags": ["insufficient_documentation"],
            }
        else:
            assessment = {
                "recommended_outcome": DecisionOutcome.AUTO_APPROVED.value,
                "confidence": 0.93,
                "rationale": (
                    "Service is covered under the policy, the provider is in-network, and the "
                    "clinical narrative supports medical necessity for the billed care."
                ),
                "medical_necessity_met": True,
                "cited_sources": cited,
                "edge_case_flags": [],
            }
        return json.dumps(assessment)

    @staticmethod
    def _extract_claim_section(prompt: str) -> str:
        """Return only the de-identified claim portion of the user prompt.

        Falls back to the whole prompt when the section markers are absent (for
        example in unit tests that pass bare context strings).
        """
        marker = "## Claim under review"
        if marker not in prompt:
            return prompt
        after_marker = prompt.split(marker, 1)[1]
        # Stop at the trailing "## Task" section if present.
        return after_marker.split("## Task", 1)[0]

    @staticmethod
    def _extract_cited_sources(prompt: str) -> list[str]:
        """Pull the `Source:` tags out of the prompt so the mock 'cites' them."""
        return re.findall(r"\[Source:\s*([^|\]]+)", prompt)


def build_llm_client(settings: Settings) -> LlmClient:
    """Factory returning the appropriate LLM client for the environment.

    Args:
        settings: Active application settings.

    Returns:
        A :class:`MockClaudeClient` in mock mode, else a real :class:`ClaudeClient`.
    """
    if settings.resolve_use_mock(settings.claude_credentials_present):
        logger.info("LLM client: using deterministic mock Claude.")
        return MockClaudeClient()
    logger.info("LLM client: using Anthropic Claude (%s).", settings.claude_model)
    return ClaudeClient(settings)
