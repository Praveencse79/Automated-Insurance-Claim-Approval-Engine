"""Final arbitration: combine deterministic rules + LLM assessment into a ruling.

This is where the system's risk policy lives. The core principles encoded here:

1. **Rules can veto, the LLM cannot hard-deny alone.** A ``BLOCKER`` rule
   failure produces an immediate ``AUTO_DENIED``. But if the LLM recommends a
   denial without a supporting blocker, the claim is escalated to a human
   rather than auto-denied - denials carry regulatory and reputational risk.
2. **Auto-approval requires consensus + high confidence + low risk.** A claim
   is auto-approved only when no blocker failed, the LLM recommends approval,
   confidence clears the threshold, and the claim is not high-value.
3. **When in doubt, route to a human.** Every uncertain path resolves to
   ``MANUAL_REVIEW`` - safety over coverage.
"""

from __future__ import annotations

from claim_engine.config.settings import Settings
from claim_engine.logging_utils import get_logger
from claim_engine.models.claim import Claim
from claim_engine.models.decision import ClaimDecision, DecisionTrace, LlmAssessment, RuleFinding
from claim_engine.models.enums import DecisionOutcome, RuleSeverity
from claim_engine.models.policy import Policy

logger = get_logger(__name__)

# Each unresolved WARNING shaves this much off the effective confidence.
_WARNING_CONFIDENCE_PENALTY = 0.10


class ApprovalEngine:
    """Arbitrates the final :class:`ClaimDecision` from all available signals."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def decide(
        self,
        claim: Claim,
        policy: Policy,
        rule_findings: list[RuleFinding],
        assessment: LlmAssessment,
        trace: DecisionTrace,
    ) -> ClaimDecision:
        """Produce the authoritative decision for a claim.

        Args:
            claim: The claim under adjudication.
            policy: The governing policy.
            rule_findings: Output of the deterministic rules engine.
            assessment: Validated LLM assessment.
            trace: Mutable audit trace to attach to the decision.

        Returns:
            A fully-populated :class:`ClaimDecision`.
        """
        blockers = [f for f in rule_findings if f.severity == RuleSeverity.BLOCKER and not f.passed]
        warnings = [f for f in rule_findings if f.severity == RuleSeverity.WARNING and not f.passed]

        # 1) Hard denial: any failed blocker rule vetoes the claim outright.
        if blockers:
            return self._build_decision(
                claim=claim,
                outcome=DecisionOutcome.AUTO_DENIED,
                confidence=0.99,
                approved_amount=0.0,
                requires_human_review=False,
                summary=f"Denied: {blockers[0].message}",
                trace=trace,
            )

        effective_confidence = self._compute_effective_confidence(assessment, warnings)
        is_high_value = claim.total_billed_amount >= self._settings.high_value_threshold

        # 2) LLM-recommended denial without a blocker -> escalate, never auto-deny.
        if assessment.recommended_outcome == DecisionOutcome.AUTO_DENIED:
            return self._build_decision(
                claim=claim,
                outcome=DecisionOutcome.MANUAL_REVIEW,
                confidence=effective_confidence,
                approved_amount=0.0,
                requires_human_review=True,
                summary="Model recommends denial without a hard policy block; human review required.",
                trace=trace,
            )

        # 3) Auto-approval gate: consensus + confidence + acceptable risk.
        can_auto_approve = (
            assessment.recommended_outcome == DecisionOutcome.AUTO_APPROVED
            and assessment.medical_necessity_met
            and effective_confidence >= self._settings.auto_approve_threshold
            and not is_high_value
            and not assessment.edge_case_flags
        )
        if can_auto_approve:
            approved_amount = self._compute_approved_amount(claim, policy)
            return self._build_decision(
                claim=claim,
                outcome=DecisionOutcome.AUTO_APPROVED,
                confidence=effective_confidence,
                approved_amount=approved_amount,
                requires_human_review=False,
                summary="Approved automatically: covered, medically necessary, high confidence.",
                trace=trace,
            )

        # 4) Everything else routes to a human, with a reason for the escalation.
        return self._build_decision(
            claim=claim,
            outcome=DecisionOutcome.MANUAL_REVIEW,
            confidence=effective_confidence,
            approved_amount=0.0,
            requires_human_review=True,
            summary=self._review_reason(effective_confidence, is_high_value, warnings, assessment),
            trace=trace,
        )

    # ------------------------------------------------------------------ helpers
    def _compute_effective_confidence(
        self, assessment: LlmAssessment, warnings: list[RuleFinding]
    ) -> float:
        """Discount the model's confidence by unresolved warning findings."""
        penalty = _WARNING_CONFIDENCE_PENALTY * len(warnings)
        return round(max(0.0, min(1.0, assessment.confidence - penalty)), 4)

    def _compute_approved_amount(self, claim: Claim, policy: Policy) -> float:
        """Apply co-payment and cap the payout at the remaining sum insured."""
        member_share = claim.total_billed_amount * policy.co_payment_rate
        payable = claim.total_billed_amount - member_share
        capped = min(payable, policy.remaining_sum_insured)
        return round(max(capped, 0.0), 2)

    def _review_reason(
        self,
        confidence: float,
        is_high_value: bool,
        warnings: list[RuleFinding],
        assessment: LlmAssessment,
    ) -> str:
        """Compose a precise, human-readable reason for manual escalation."""
        reasons: list[str] = []
        if is_high_value:
            reasons.append("high-value claim")
        if confidence < self._settings.auto_approve_threshold:
            reasons.append(f"confidence {confidence:.2f} below auto-approve threshold")
        if warnings:
            reasons.append(f"{len(warnings)} policy warning(s)")
        if assessment.edge_case_flags:
            reasons.append("edge-case flags: " + ", ".join(assessment.edge_case_flags))
        return "Routed to manual review (" + "; ".join(reasons or ["insufficient confidence"]) + ")."

    def _build_decision(
        self,
        claim: Claim,
        outcome: DecisionOutcome,
        confidence: float,
        approved_amount: float,
        requires_human_review: bool,
        summary: str,
        trace: DecisionTrace,
    ) -> ClaimDecision:
        """Assemble and log the final decision object."""
        decision = ClaimDecision(
            claim_id=claim.claim_id,
            outcome=outcome,
            confidence=confidence,
            approved_amount=approved_amount,
            requires_human_review=requires_human_review,
            summary=summary,
            trace=trace,
        )
        logger.info(
            "Decision for %s: %s (confidence=%.2f, payout=%.2f)",
            claim.claim_id,
            outcome.value,
            confidence,
            approved_amount,
        )
        return decision
