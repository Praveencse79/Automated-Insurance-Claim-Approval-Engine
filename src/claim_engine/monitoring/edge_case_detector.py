"""Heuristic edge-case detector for proactive human-in-the-loop routing.

Even when the rules pass and the model is confident, certain *shapes* of claim
warrant extra caution (novel/unspecified diagnoses, statistical cost outliers,
out-of-network high-value care, sparse documentation). Surfacing these as
explicit flags lets the approval engine and on-call operators catch problems
the happy path would miss.
"""

from __future__ import annotations

from claim_engine.models.claim import Claim
from claim_engine.models.policy import Policy

# Diagnosis codes that are inherently non-specific and merit closer review.
_UNSPECIFIED_DIAGNOSIS_CODES = {"R69", "R68.89", "Z00.00"}

# A claim billed above this multiple of the policy's average is an outlier.
_COST_OUTLIER_MULTIPLE = 5.0
_ASSUMED_AVERAGE_CLAIM_INR = 50_000.0


class EdgeCaseDetector:
    """Flags claims that should bypass full automation for human attention."""

    def detect(self, claim: Claim, policy: Policy) -> list[str]:
        """Return a list of snake_case edge-case flags for a claim.

        Args:
            claim: The claim under adjudication.
            policy: The governing policy (for context-relative checks).

        Returns:
            Zero or more flag strings; empty means no edge cases detected.
        """
        flags: list[str] = []

        if self._has_unspecified_diagnosis(claim):
            flags.append("unspecified_diagnosis")
        if self._is_cost_outlier(claim):
            flags.append("cost_outlier")
        if self._is_sparse_documentation(claim):
            flags.append("sparse_documentation")
        if self._is_out_of_network_high_value(claim):
            flags.append("out_of_network_high_value")
        if self._exhausts_sum_insured(claim, policy):
            flags.append("near_sum_insured_exhaustion")

        return flags

    # ----------------------------------------------------------- detectors
    def _has_unspecified_diagnosis(self, claim: Claim) -> bool:
        """True if every diagnosis code is missing or inherently unspecified."""
        if not claim.diagnosis_codes:
            return True
        return all(code in _UNSPECIFIED_DIAGNOSIS_CODES for code in claim.diagnosis_codes)

    def _is_cost_outlier(self, claim: Claim) -> bool:
        """True if the billed amount is a gross statistical outlier."""
        return claim.total_billed_amount > _COST_OUTLIER_MULTIPLE * _ASSUMED_AVERAGE_CLAIM_INR

    def _is_sparse_documentation(self, claim: Claim) -> bool:
        """True if the clinical narrative is too thin to support automation."""
        return len(claim.clinical_notes.strip()) < 40

    def _is_out_of_network_high_value(self, claim: Claim) -> bool:
        """True for expensive care delivered by an out-of-network provider."""
        return (not claim.provider_in_network) and claim.total_billed_amount > 100_000.0

    def _exhausts_sum_insured(self, claim: Claim, policy: Policy) -> bool:
        """True if approving this claim would consume >90% of remaining cover."""
        remaining = policy.remaining_sum_insured
        if remaining <= 0:
            return True
        return claim.total_billed_amount >= 0.90 * remaining
