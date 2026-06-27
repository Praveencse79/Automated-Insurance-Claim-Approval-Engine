"""Orchestrates the Retrieval-Augmented Generation reasoning step.

Pipeline of this component (the 'AG' in RAG):

    de-identified claim context + retrieved docs
        -> prompt assembly
        -> LLM completion (Claude / mock)
        -> JSON parse + Pydantic validation
        -> LlmAssessment

It is deliberately tolerant of imperfect model output (it extracts the first
JSON object and falls back to a safe MANUAL_REVIEW assessment on parse failure),
because in adjudication an unpar-seable response must never silently approve.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from claim_engine.logging_utils import get_logger
from claim_engine.models.decision import LlmAssessment
from claim_engine.models.enums import DecisionOutcome
from claim_engine.models.knowledge import RetrievedContext
from claim_engine.reasoning.claude_client import LlmClient
from claim_engine.reasoning.prompt_templates import SYSTEM_PROMPT, build_user_prompt

logger = get_logger(__name__)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class RagClaimReasoner:
    """Produces a validated :class:`LlmAssessment` for a claim via RAG."""

    def __init__(self, llm_client: LlmClient) -> None:
        self._llm_client = llm_client

    def assess(
        self, safe_claim_context: str, retrieved: list[RetrievedContext]
    ) -> LlmAssessment:
        """Run the grounded LLM assessment for one claim.

        Args:
            safe_claim_context: HIPAA-redacted claim summary (no PHI).
            retrieved: Ranked grounding documents from the retriever.

        Returns:
            A validated :class:`LlmAssessment`. On any LLM/parse error a
            conservative ``MANUAL_REVIEW`` assessment is returned so the claim
            is escalated to a human rather than mishandled.
        """
        user_prompt = build_user_prompt(safe_claim_context, retrieved)
        try:
            raw_response = self._llm_client.complete(SYSTEM_PROMPT, user_prompt)
            return self._parse_assessment(raw_response)
        except Exception as error:  # defensive: never let the LLM crash the pipeline
            logger.warning("LLM assessment failed; defaulting to manual review: %s", error)
            return self._fallback_assessment(reason=str(error))

    def _parse_assessment(self, raw_response: str) -> LlmAssessment:
        """Extract, parse and validate the JSON assessment from raw model text."""
        json_text = self._extract_json(raw_response)
        if json_text is None:
            logger.warning("No JSON object found in model response.")
            return self._fallback_assessment(reason="no_json_in_response")

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as error:
            logger.warning("Malformed JSON in model response: %s", error)
            return self._fallback_assessment(reason="malformed_json")

        return LlmAssessment.model_validate(payload)

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Return the first ``{...}`` block in ``text``, or ``None``."""
        match = _JSON_OBJECT_PATTERN.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _fallback_assessment(reason: str) -> LlmAssessment:
        """Build a safe, low-confidence assessment that forces human review."""
        return LlmAssessment(
            recommended_outcome=DecisionOutcome.MANUAL_REVIEW,
            confidence=0.0,
            rationale=f"Automatic assessment unavailable ({reason}); routing to human review.",
            medical_necessity_met=False,
            cited_sources=[],
            edge_case_flags=["llm_unavailable"],
        )
