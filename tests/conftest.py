"""Shared pytest fixtures.

Every fixture forces mock mode so the suite is hermetic: no network, no
credentials, fully deterministic.
"""

from __future__ import annotations

from datetime import date

import pytest

from claim_engine.config.settings import Settings
from claim_engine.models.claim import Claim, ClaimLineItem, Member
from claim_engine.models.enums import ClaimType, Gender
from claim_engine.models.policy import Policy
from claim_engine.pipeline.claim_pipeline import ClaimApprovalPipeline, build_pipeline


@pytest.fixture()
def settings() -> Settings:
    """Return a settings instance pinned to mock mode."""
    return Settings(mock_mode=True, environment="local")


@pytest.fixture()
def pipeline(settings: Settings) -> ClaimApprovalPipeline:
    """A freshly-composed pipeline (with seeded fixtures) per test."""
    return build_pipeline(settings)


@pytest.fixture()
def sample_policy() -> Policy:
    """A standard active policy used across decisioning tests."""
    return Policy(
        policy_id="POL-TEST-1",
        product_name="Test Plan",
        effective_date=date(2025, 1, 1),
        expiry_date=date(2026, 12, 31),
        annual_sum_insured=1_000_000.0,
        amount_consumed_ytd=0.0,
        covered_claim_types=[ClaimType.INPATIENT, ClaimType.OUTPATIENT],
        excluded_procedure_codes=["COSMETIC-001"],
        waiting_period_days=30,
        co_payment_rate=0.10,
        requires_prior_auth_above=200_000.0,
    )


@pytest.fixture()
def sample_member() -> Member:
    """A representative insured member."""
    return Member(
        member_id="MEM-TEST-1",
        full_name="Test Member",
        date_of_birth=date(1990, 1, 1),
        gender=Gender.OTHER,
        policy_id="POL-TEST-1",
    )


@pytest.fixture()
def clean_claim(sample_member: Member) -> Claim:
    """A clearly-approvable in-network inpatient claim."""
    return Claim(
        claim_id="CLM-CLEAN-1",
        member=sample_member,
        claim_type=ClaimType.INPATIENT,
        diagnosis_codes=["J18.9"],
        line_items=[
            ClaimLineItem(
                procedure_code="99223",
                description="Initial hospital inpatient care",
                quantity=1,
                unit_amount=40_000.0,
            )
        ],
        provider_id="PRV-1",
        provider_in_network=True,
        service_date=date(2025, 6, 1),
        clinical_notes=(
            "Patient admitted with confirmed pneumonia on chest imaging; IV antibiotics "
            "started and clinical course documented in full."
        ),
    )
