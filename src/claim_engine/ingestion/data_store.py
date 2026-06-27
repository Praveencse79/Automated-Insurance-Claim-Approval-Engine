"""Abstract data store plus an in-memory implementation and a factory.

The engine depends only on the :class:`ClaimDataStore` *protocol*, never on a
concrete database. This is the Dependency-Inversion principle in action and is
what lets the entire pipeline run against deterministic fixtures in CI while
talking to Snowflake in production - with zero changes to business logic.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from claim_engine.config.settings import Settings
from claim_engine.logging_utils import get_logger
from claim_engine.models.claim import Claim, ClaimLineItem, Member
from claim_engine.models.decision import ClaimDecision
from claim_engine.models.enums import ClaimType, Gender
from claim_engine.models.policy import Policy

logger = get_logger(__name__)


@runtime_checkable
class ClaimDataStore(Protocol):
    """Read/write contract for claim, policy and decision persistence."""

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Return the claim with ``claim_id`` or ``None`` if it does not exist."""
        ...

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Return the policy with ``policy_id`` or ``None`` if not found."""
        ...

    def save_decision(self, decision: ClaimDecision) -> None:
        """Durably persist a finalised decision for audit and downstream payout."""
        ...


class InMemoryClaimDataStore:
    """Deterministic, dependency-free data store backed by Python dicts.

    Seeded with a small, representative set of claims and policies that
    exercise every decision path (clean approval, hard denial, edge case).
    Used automatically whenever Snowflake credentials are absent or mock mode
    is enabled.
    """

    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}
        self._policies: dict[str, Policy] = {}
        self._decisions: dict[str, ClaimDecision] = {}
        self._seed_reference_data()

    # ------------------------------------------------------------- read API
    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Look up a seeded claim by id."""
        return self._claims.get(claim_id)

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Look up a seeded policy by id."""
        return self._policies.get(policy_id)

    # ------------------------------------------------------------ write API
    def save_decision(self, decision: ClaimDecision) -> None:
        """Store the decision in-memory (a real store would INSERT a row)."""
        self._decisions[decision.claim_id] = decision
        logger.debug("Persisted decision for claim %s", decision.claim_id)

    # ----------------------------------------------------------- test hooks
    def add_claim(self, claim: Claim) -> None:
        """Register an additional claim (used by tests and the demo script)."""
        self._claims[claim.claim_id] = claim

    def add_policy(self, policy: Policy) -> None:
        """Register an additional policy (used by tests and the demo script)."""
        self._policies[policy.policy_id] = policy

    def get_saved_decision(self, claim_id: str) -> Optional[ClaimDecision]:
        """Retrieve a previously-saved decision (test/inspection helper)."""
        return self._decisions.get(claim_id)

    # --------------------------------------------------------- seed fixtures
    def _seed_reference_data(self) -> None:
        """Populate the store with canonical sample policies and claims."""
        gold_policy = Policy(
            policy_id="POL-GOLD-001",
            product_name="HealthGuard Gold",
            effective_date=date(2025, 1, 1),
            expiry_date=date(2026, 12, 31),
            annual_sum_insured=1_000_000.0,
            amount_consumed_ytd=150_000.0,
            covered_claim_types=[
                ClaimType.INPATIENT,
                ClaimType.OUTPATIENT,
                ClaimType.DIAGNOSTIC,
                ClaimType.EMERGENCY,
                ClaimType.MATERNITY,
            ],
            excluded_procedure_codes=["COSMETIC-001"],
            waiting_period_days=30,
            co_payment_rate=0.10,
            requires_prior_auth_above=200_000.0,
        )
        self.add_policy(gold_policy)

        member = Member(
            member_id="MEM-1001",
            full_name="Asha Verma",
            date_of_birth=date(1986, 4, 17),
            gender=Gender.FEMALE,
            policy_id=gold_policy.policy_id,
            email="asha.verma@example.com",
            phone="+91 98765 43210",
        )

        # 1) A clean, clearly-approvable inpatient claim.
        self.add_claim(
            Claim(
                claim_id="CLM-APPROVE-001",
                member=member,
                claim_type=ClaimType.INPATIENT,
                diagnosis_codes=["J18.9"],  # Pneumonia, unspecified organism
                line_items=[
                    ClaimLineItem(
                        procedure_code="99223",
                        description="Initial hospital inpatient care, high complexity",
                        quantity=1,
                        unit_amount=45_000.0,
                    ),
                    ClaimLineItem(
                        procedure_code="J7050",
                        description="IV fluids and supportive care",
                        quantity=3,
                        unit_amount=2_500.0,
                    ),
                ],
                provider_id="PRV-NET-22",
                provider_in_network=True,
                service_date=date(2025, 6, 10),
                clinical_notes=(
                    "Patient Mrs. Asha Verma admitted with high fever, productive cough "
                    "and chest X-ray confirming right lower lobe pneumonia. IV antibiotics "
                    "administered. Contact: asha.verma@example.com, +91 98765 43210."
                ),
            )
        )

        # 2) A claim that must be hard-denied: excluded cosmetic procedure.
        self.add_claim(
            Claim(
                claim_id="CLM-DENY-001",
                member=member,
                claim_type=ClaimType.OUTPATIENT,
                diagnosis_codes=["Z41.1"],  # Encounter for cosmetic surgery
                line_items=[
                    ClaimLineItem(
                        procedure_code="COSMETIC-001",
                        description="Elective cosmetic rhinoplasty",
                        quantity=1,
                        unit_amount=180_000.0,
                    )
                ],
                provider_id="PRV-OON-09",
                provider_in_network=False,
                service_date=date(2025, 7, 1),
                clinical_notes="Elective cosmetic procedure requested by patient.",
            )
        )

        # 3) An ambiguous, high-value edge case -> manual review.
        self.add_claim(
            Claim(
                claim_id="CLM-REVIEW-001",
                member=member,
                claim_type=ClaimType.INPATIENT,
                diagnosis_codes=["R69"],  # Illness, unspecified
                line_items=[
                    ClaimLineItem(
                        procedure_code="99291",
                        description="Critical care, complex multi-system evaluation",
                        quantity=1,
                        unit_amount=620_000.0,
                    )
                ],
                provider_id="PRV-NET-22",
                provider_in_network=True,
                service_date=date(2025, 8, 5),
                clinical_notes=(
                    "Prolonged ICU stay with unclear primary diagnosis; multiple "
                    "specialist consults ongoing. Documentation incomplete."
                ),
            )
        )


class SnowflakeClaimDataStore:
    """Production data store backed by Snowflake.

    Lazily imports the ``snowflake-connector-python`` driver so the package can
    be imported (and mock mode can run) without the heavy dependency installed.
    Queries are parameterised to prevent SQL injection, and the connection is
    pooled/reused across calls.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._connection = None  # established lazily on first use

    def _get_connection(self):  # noqa: ANN202 - vendor connection object
        """Open (once) and return the Snowflake connection."""
        if self._connection is not None:
            return self._connection
        import snowflake.connector  # local import: optional heavy dependency

        self._connection = snowflake.connector.connect(
            account=self._settings.snowflake_account,
            user=self._settings.snowflake_user,
            password=self._settings.snowflake_password,
            warehouse=self._settings.snowflake_warehouse,
            database=self._settings.snowflake_database,
            schema=self._settings.snowflake_schema,
        )
        logger.info("Opened Snowflake connection to %s", self._settings.snowflake_account)
        return self._connection

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Fetch and hydrate a claim (and its line items) from Snowflake.

        Implementation note: in a real deployment this joins CLAIMS,
        CLAIM_LINE_ITEMS and MEMBERS. The mapping logic is intentionally
        omitted here; the in-memory store is the executable reference.
        """
        raise NotImplementedError(
            "Connect real Snowflake schema mapping here; use mock mode for local runs."
        )

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Fetch and hydrate a policy from Snowflake."""
        raise NotImplementedError(
            "Connect real Snowflake schema mapping here; use mock mode for local runs."
        )

    def save_decision(self, decision: ClaimDecision) -> None:
        """Persist a decision row into the DECISIONS table (parameterised)."""
        cursor = self._get_connection().cursor()
        try:
            cursor.execute(
                """
                INSERT INTO DECISIONS
                    (CLAIM_ID, OUTCOME, CONFIDENCE, APPROVED_AMOUNT,
                     REQUIRES_HUMAN_REVIEW, SUMMARY, DECIDED_AT)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    decision.claim_id,
                    decision.outcome.value,
                    decision.confidence,
                    decision.approved_amount,
                    decision.requires_human_review,
                    decision.summary,
                    decision.decided_at or datetime.now(timezone.utc),
                ),
            )
        finally:
            cursor.close()


def build_data_store(settings: Settings) -> ClaimDataStore:
    """Factory selecting the appropriate data store for the environment.

    Returns the in-memory store when Snowflake credentials are missing or mock
    mode is on; otherwise returns the Snowflake-backed store.

    Args:
        settings: The active application settings.

    Returns:
        An object satisfying the :class:`ClaimDataStore` protocol.
    """
    if settings.resolve_use_mock(settings.snowflake_credentials_present):
        logger.info("Data store: using in-memory mock (Snowflake disabled).")
        return InMemoryClaimDataStore()
    logger.info("Data store: using Snowflake adapter.")
    return SnowflakeClaimDataStore(settings)
