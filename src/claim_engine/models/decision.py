"""Models describing the engine's reasoning artefacts and final decision."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from claim_engine.models.enums import DecisionOutcome, RuleSeverity


class RuleFinding(BaseModel):
    """The outcome of evaluating a single deterministic business rule.

    Deterministic rules act as guardrails around the probabilistic LLM: a
    ``BLOCKER`` finding can veto an approval the model would otherwise make.
    """

    rule_id: str = Field(..., description="Stable identifier of the rule, e.g. 'POLICY_EXPIRED'.")
    severity: RuleSeverity
    passed: bool = Field(..., description="True if the rule's condition was satisfied (no issue).")
    message: str = Field(..., description="Human-readable explanation of the finding.")


class LlmAssessment(BaseModel):
    """Structured output parsed from the Claude RAG response.

    The prompt instructs the model to return strict JSON; this model is the
    typed landing zone for that JSON after validation.
    """

    recommended_outcome: DecisionOutcome = Field(
        ..., description="The model's recommendation before rule arbitration."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's self-reported confidence in its recommendation."
    )
    rationale: str = Field(..., description="Concise, grounded justification for the recommendation.")
    medical_necessity_met: bool = Field(
        ..., description="Whether the clinical evidence supports medical necessity."
    )
    cited_sources: list[str] = Field(
        default_factory=list, description="Knowledge-base sources the model relied upon."
    )
    edge_case_flags: list[str] = Field(
        default_factory=list,
        description="Anomalies the model noticed (e.g. 'ambiguous_diagnosis').",
    )


class DecisionTrace(BaseModel):
    """Full, replayable audit trail of how a decision was reached.

    Persisted verbatim so any decision can be explained to a regulator,
    auditor or the member months later - a hard requirement in regulated
    health-insurance adjudication.
    """

    retrieved_sources: list[str] = Field(default_factory=list)
    rule_findings: list[RuleFinding] = Field(default_factory=list)
    llm_assessment: Optional[LlmAssessment] = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    used_mock_components: list[str] = Field(default_factory=list)


class ClaimDecision(BaseModel):
    """The engine's final, authoritative ruling on a claim."""

    claim_id: str
    outcome: DecisionOutcome
    confidence: float = Field(..., ge=0.0, le=1.0)
    approved_amount: float = Field(default=0.0, ge=0.0, description="Amount approved for payout (INR).")
    requires_human_review: bool = Field(default=False)
    summary: str = Field(..., description="One-line, member-facing explanation of the outcome.")
    trace: DecisionTrace = Field(default_factory=DecisionTrace)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_auto_resolved(self) -> bool:
        """True when the engine resolved the claim without a human."""
        return self.outcome in (DecisionOutcome.AUTO_APPROVED, DecisionOutcome.AUTO_DENIED)
