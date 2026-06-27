"""Tests for the deterministic rules engine."""

from __future__ import annotations

from datetime import date

from claim_engine.decisioning.rules_engine import DeterministicRulesEngine
from claim_engine.models.claim import Claim, ClaimLineItem, Member
from claim_engine.models.enums import ClaimType
from claim_engine.models.policy import Policy


def _finding(findings, rule_id):
    """Helper: return the finding with the given rule id."""
    return next(f for f in findings if f.rule_id == rule_id)


def test_clean_claim_passes_all_blockers(clean_claim: Claim, sample_policy: Policy) -> None:
    """A clean claim should not trip any blocker rule."""
    findings = DeterministicRulesEngine().evaluate(clean_claim, sample_policy)
    assert all(f.passed for f in findings if f.severity.value == "BLOCKER")


def test_excluded_procedure_is_blocked(sample_member: Member, sample_policy: Policy) -> None:
    """A billed exclusion code must fail the NOT_EXCLUDED_PROCEDURE blocker."""
    claim = Claim(
        claim_id="CLM-X",
        member=sample_member,
        claim_type=ClaimType.OUTPATIENT,
        diagnosis_codes=["Z41.1"],
        line_items=[
            ClaimLineItem(
                procedure_code="COSMETIC-001",
                description="Elective cosmetic procedure",
                unit_amount=100_000.0,
            )
        ],
        provider_id="PRV-9",
        provider_in_network=False,
        service_date=date(2025, 7, 1),
    )
    findings = DeterministicRulesEngine().evaluate(claim, sample_policy)
    assert _finding(findings, "NOT_EXCLUDED_PROCEDURE").passed is False


def test_waiting_period_violation(sample_member: Member, sample_policy: Policy) -> None:
    """Service within the waiting period must fail the waiting-period rule."""
    claim = Claim(
        claim_id="CLM-W",
        member=sample_member,
        claim_type=ClaimType.INPATIENT,
        diagnosis_codes=["J18.9"],
        line_items=[ClaimLineItem(procedure_code="99223", description="care", unit_amount=10_000.0)],
        provider_id="PRV-1",
        service_date=date(2025, 1, 10),  # only 9 days after effective date
    )
    findings = DeterministicRulesEngine().evaluate(claim, sample_policy)
    assert _finding(findings, "WAITING_PERIOD_SATISFIED").passed is False
