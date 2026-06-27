"""End-to-end pipeline tests over the seeded sample claims."""

from __future__ import annotations

from claim_engine.models.enums import DecisionOutcome
from claim_engine.pipeline.claim_pipeline import ClaimApprovalPipeline


def test_clean_claim_is_auto_approved(pipeline: ClaimApprovalPipeline) -> None:
    """The seeded clean claim flows through to an automatic approval."""
    decision = pipeline.process_claim_by_id("CLM-APPROVE-001")
    assert decision.outcome == DecisionOutcome.AUTO_APPROVED
    assert decision.approved_amount > 0
    assert decision.trace.latency_ms >= 0


def test_excluded_claim_is_auto_denied(pipeline: ClaimApprovalPipeline) -> None:
    """The seeded cosmetic claim is hard-denied by a blocker rule."""
    decision = pipeline.process_claim_by_id("CLM-DENY-001")
    assert decision.outcome == DecisionOutcome.AUTO_DENIED
    assert decision.approved_amount == 0.0


def test_ambiguous_high_value_claim_routes_to_review(pipeline: ClaimApprovalPipeline) -> None:
    """The seeded ambiguous high-value claim requires human review."""
    decision = pipeline.process_claim_by_id("CLM-REVIEW-001")
    assert decision.outcome == DecisionOutcome.MANUAL_REVIEW
    assert decision.requires_human_review is True


def test_metrics_reflect_processed_claims(pipeline: ClaimApprovalPipeline) -> None:
    """Processing all three claims updates the metrics snapshot accordingly."""
    for claim_id in ("CLM-APPROVE-001", "CLM-DENY-001", "CLM-REVIEW-001"):
        pipeline.process_claim_by_id(claim_id)
    snapshot = pipeline.metrics.snapshot()
    assert snapshot.total_claims == 3
    assert snapshot.auto_resolution_rate > 0.0


def test_unknown_claim_raises(pipeline: ClaimApprovalPipeline) -> None:
    """Referencing a missing claim raises a clear lookup error."""
    import pytest

    from claim_engine.pipeline.claim_pipeline import ClaimNotFoundError

    with pytest.raises(ClaimNotFoundError):
        pipeline.process_claim_by_id("DOES-NOT-EXIST")
