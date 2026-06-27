"""Deterministic policy & eligibility rules.

These rules are the **non-negotiable guardrails** around the probabilistic LLM.
They encode contractual and regulatory facts that must never be left to a
language model: policy validity, coverage scope, exclusions, waiting periods,
sum-insured limits and prior-authorisation requirements.

A ``BLOCKER`` finding can veto an approval the LLM would otherwise make; a
``WARNING`` lowers confidence and can trigger manual review. This hybrid
(symbolic + neural) design is the industry-standard pattern for trustworthy
automated adjudication.
"""

from __future__ import annotations

from claim_engine.logging_utils import get_logger
from claim_engine.models.claim import Claim
from claim_engine.models.decision import RuleFinding
from claim_engine.models.enums import RuleSeverity
from claim_engine.models.policy import Policy

logger = get_logger(__name__)


class DeterministicRulesEngine:
    """Evaluates a claim against its policy and returns structured findings."""

    def evaluate(self, claim: Claim, policy: Policy) -> list[RuleFinding]:
        """Run every rule and collect their findings.

        Args:
            claim: The claim under adjudication.
            policy: The policy the claim is filed against.

        Returns:
            One :class:`RuleFinding` per rule, in evaluation order.
        """
        findings = [
            self._check_policy_active(claim, policy),
            self._check_claim_type_covered(claim, policy),
            self._check_not_excluded_procedure(claim, policy),
            self._check_waiting_period(claim, policy),
            self._check_within_sum_insured(claim, policy),
            self._check_prior_authorization(claim, policy),
            self._check_provider_in_network(claim),
        ]
        failed = [f.rule_id for f in findings if not f.passed]
        if failed:
            logger.debug("Claim %s failed rules: %s", claim.claim_id, failed)
        return findings

    # ------------------------------------------------------------- individual rules
    def _check_policy_active(self, claim: Claim, policy: Policy) -> RuleFinding:
        """Service date must fall within the policy's active coverage window."""
        active = policy.is_active_on(claim.service_date)
        return RuleFinding(
            rule_id="POLICY_ACTIVE",
            severity=RuleSeverity.BLOCKER,
            passed=active,
            message=(
                "Policy active on service date."
                if active
                else "Service date is outside the policy's coverage period."
            ),
        )

    def _check_claim_type_covered(self, claim: Claim, policy: Policy) -> RuleFinding:
        """The claim type must be within the policy's covered categories."""
        covered = policy.covers_claim_type(claim.claim_type)
        return RuleFinding(
            rule_id="CLAIM_TYPE_COVERED",
            severity=RuleSeverity.BLOCKER,
            passed=covered,
            message=(
                f"Claim type {claim.claim_type.value} is covered."
                if covered
                else f"Claim type {claim.claim_type.value} is not covered by this policy."
            ),
        )

    def _check_not_excluded_procedure(self, claim: Claim, policy: Policy) -> RuleFinding:
        """No billed procedure code may appear in the policy's exclusion list."""
        excluded = {code.upper() for code in policy.excluded_procedure_codes}
        offending = [
            item.procedure_code
            for item in claim.line_items
            if item.procedure_code.upper() in excluded
        ]
        passed = not offending
        return RuleFinding(
            rule_id="NOT_EXCLUDED_PROCEDURE",
            severity=RuleSeverity.BLOCKER,
            passed=passed,
            message=(
                "No excluded procedures billed."
                if passed
                else f"Billed procedure(s) are policy-excluded: {', '.join(offending)}."
            ),
        )

    def _check_waiting_period(self, claim: Claim, policy: Policy) -> RuleFinding:
        """Service must occur after the policy's initial waiting period."""
        days_since_effective = (claim.service_date - policy.effective_date).days
        passed = days_since_effective >= policy.waiting_period_days
        return RuleFinding(
            rule_id="WAITING_PERIOD_SATISFIED",
            severity=RuleSeverity.BLOCKER,
            passed=passed,
            message=(
                "Waiting period satisfied."
                if passed
                else (
                    f"Service rendered {days_since_effective} days after inception; "
                    f"waiting period is {policy.waiting_period_days} days."
                )
            ),
        )

    def _check_within_sum_insured(self, claim: Claim, policy: Policy) -> RuleFinding:
        """Billed amount must not exceed the remaining annual sum insured."""
        passed = claim.total_billed_amount <= policy.remaining_sum_insured
        return RuleFinding(
            rule_id="WITHIN_SUM_INSURED",
            severity=RuleSeverity.WARNING,
            passed=passed,
            message=(
                "Within remaining sum insured."
                if passed
                else (
                    f"Billed {claim.total_billed_amount} exceeds remaining sum insured "
                    f"{policy.remaining_sum_insured}; payout will be capped."
                )
            ),
        )

    def _check_prior_authorization(self, claim: Claim, policy: Policy) -> RuleFinding:
        """High-value services require a prior-authorisation reference."""
        threshold = policy.requires_prior_auth_above
        needs_auth = threshold is not None and claim.total_billed_amount > threshold
        has_auth = bool(claim.prior_authorization_number)
        passed = (not needs_auth) or has_auth
        return RuleFinding(
            rule_id="PRIOR_AUTH_PRESENT",
            severity=RuleSeverity.WARNING,
            passed=passed,
            message=(
                "Prior authorisation present or not required."
                if passed
                else (
                    f"Amount exceeds {threshold} and requires prior authorisation, "
                    "but none was supplied."
                )
            ),
        )

    def _check_provider_in_network(self, claim: Claim) -> RuleFinding:
        """Out-of-network providers reduce confidence (but are not a hard block)."""
        return RuleFinding(
            rule_id="PROVIDER_IN_NETWORK",
            severity=RuleSeverity.WARNING,
            passed=claim.provider_in_network,
            message=(
                "Provider is in-network."
                if claim.provider_in_network
                else "Provider is out-of-network; higher cost-share / scrutiny applies."
            ),
        )
