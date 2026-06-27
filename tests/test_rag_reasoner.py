"""Tests for the RAG reasoner and its resilient JSON parsing."""

from __future__ import annotations

from claim_engine.models.enums import DecisionOutcome
from claim_engine.reasoning.claude_client import MockClaudeClient
from claim_engine.reasoning.rag_reasoner import RagClaimReasoner


class _BrokenClient:
    """An LLM client that returns non-JSON text, to exercise the fallback."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "I'm sorry, I cannot help with that."


def test_mock_client_approves_clean_context() -> None:
    """Clean context yields an approval recommendation from the mock client."""
    reasoner = RagClaimReasoner(MockClaudeClient())
    assessment = reasoner.assess(
        "Claim type: INPATIENT. Covered pneumonia care, in-network provider.", retrieved=[]
    )
    assert assessment.recommended_outcome == DecisionOutcome.AUTO_APPROVED
    assert assessment.confidence >= 0.9


def test_cosmetic_context_is_denied() -> None:
    """Context mentioning a cosmetic exclusion yields a denial recommendation."""
    reasoner = RagClaimReasoner(MockClaudeClient())
    assessment = reasoner.assess("Elective cosmetic rhinoplasty requested.", retrieved=[])
    assert assessment.recommended_outcome == DecisionOutcome.AUTO_DENIED


def test_unparseable_response_falls_back_to_review() -> None:
    """A non-JSON model response must safely degrade to MANUAL_REVIEW."""
    reasoner = RagClaimReasoner(_BrokenClient())
    assessment = reasoner.assess("anything", retrieved=[])
    assert assessment.recommended_outcome == DecisionOutcome.MANUAL_REVIEW
    assert assessment.confidence == 0.0
    assert "llm_unavailable" in assessment.edge_case_flags
