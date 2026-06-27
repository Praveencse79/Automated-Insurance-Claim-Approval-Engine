"""Tests for the final approval arbitration logic."""

from __future__ import annotations

from claim_engine.config.settings import Settings
from claim_engine.decisioning.approval_engine import ApprovalEngine
from claim_engine.decisioning.rules_engine import DeterministicRulesEngine
from claim_engine.models.claim import Claim
from claim_engine.models.decision import DecisionTrace, LlmAssessment
from claim_engine.models.enums import DecisionOutcome
from claim_engine.models.policy import Policy


def _approve_assessment(confidence: float = 0.95) -> LlmAssessment:
    """An LLM assessment recommending approval at the given confidence."""
    return LlmAssessment(
        recommended_outcome=DecisionOutcome.AUTO_APPROVED,
        confidence=confidence,
        rationale="covered and medically necessary",
        medical_necessity_met=True,
        cited_sources=["guideline:CGL-RESP-01"],
        edge_case_flags=[],
    )


def test_high_confidence_clean_claim_is_auto_approved(
    settings: Settings, clean_claim: Claim, sample_policy: Policy
) -> None:
    """Consensus + high confidence + low risk => AUTO_APPROVED with payout."""
    findings = DeterministicRulesEngine().evaluate(clean_claim, sample_policy)
    decision = ApprovalEngine(settings).decide(
        clean_claim, sample_policy, findings, _approve_assessment(), DecisionTrace()
    )
    assert decision.outcome == DecisionOutcome.AUTO_APPROVED
    # 40,000 billed * (1 - 0.10 co-pay) = 36,000 payable.
    assert decision.approved_amount == 36_000.0


def test_low_confidence_routes_to_manual_review(
    settings: Settings, clean_claim: Claim, sample_policy: Policy
) -> None:
    """Confidence below the auto-approve threshold => MANUAL_REVIEW."""
    findings = DeterministicRulesEngine().evaluate(clean_claim, sample_policy)
    decision = ApprovalEngine(settings).decide(
        clean_claim, sample_policy, findings, _approve_assessment(confidence=0.50), DecisionTrace()
    )
    assert decision.outcome == DecisionOutcome.MANUAL_REVIEW
    assert decision.requires_human_review is True


def test_blocker_failure_forces_auto_denial(
    settings: Settings, clean_claim: Claim, sample_policy: Policy
) -> None:
    """A failed blocker rule overrides an approving LLM and denies the claim."""
    # Make the policy inactive on the service date to trip POLICY_ACTIVE.
    expired_policy = sample_policy.model_copy(
        update={"expiry_date": clean_claim.service_date.replace(year=2024)}
    )
    findings = DeterministicRulesEngine().evaluate(clean_claim, expired_policy)
    decision = ApprovalEngine(settings).decide(
        clean_claim, expired_policy, findings, _approve_assessment(), DecisionTrace()
    )
    assert decision.outcome == DecisionOutcome.AUTO_DENIED
    assert decision.approved_amount == 0.0


def test_llm_denial_without_blocker_escalates(
    settings: Settings, clean_claim: Claim, sample_policy: Policy
) -> None:
    """An LLM-only denial is never auto-applied; it escalates to a human."""
    findings = DeterministicRulesEngine().evaluate(clean_claim, sample_policy)
    deny_assessment = LlmAssessment(
        recommended_outcome=DecisionOutcome.AUTO_DENIED,
        confidence=0.9,
        rationale="model thinks not necessary",
        medical_necessity_met=False,
    )
    decision = ApprovalEngine(settings).decide(
        clean_claim, sample_policy, findings, deny_assessment, DecisionTrace()
    )
    assert decision.outcome == DecisionOutcome.MANUAL_REVIEW
