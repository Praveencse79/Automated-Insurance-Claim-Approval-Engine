"""Models describing an inbound insurance claim and its supporting entities."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from claim_engine.models.enums import ClaimType, Gender


class Member(BaseModel):
    """The insured individual (policy member) a claim is filed against.

    Contains Protected Health Information (PHI); instances must only be logged
    or sent to an LLM *after* passing through the compliance redaction layer.
    """

    member_id: str = Field(..., description="Stable internal identifier for the member.")
    full_name: str = Field(..., description="Member legal name (PHI).")
    date_of_birth: date = Field(..., description="Used to derive age-based eligibility.")
    gender: Gender = Field(default=Gender.UNKNOWN)
    policy_id: str = Field(..., description="Identifier of the member's active policy.")
    email: Optional[str] = Field(default=None, description="Contact e-mail (PHI/PII).")
    phone: Optional[str] = Field(default=None, description="Contact phone (PHI/PII).")

    @property
    def age_years(self) -> int:
        """Member age in completed years, computed from ``date_of_birth``."""
        today = datetime.now(timezone.utc).date()
        had_birthday = (today.month, today.day) >= (self.date_of_birth.month, self.date_of_birth.day)
        return today.year - self.date_of_birth.year - (0 if had_birthday else 1)


class ClaimLineItem(BaseModel):
    """A single billable line on a claim (one procedure, drug, or service)."""

    procedure_code: str = Field(..., description="CPT / ICD-10-PCS / service code.")
    description: str = Field(..., description="Human-readable description of the service.")
    quantity: int = Field(default=1, ge=1)
    unit_amount: float = Field(..., ge=0, description="Billed amount per unit (INR).")

    @property
    def line_total(self) -> float:
        """Total billed amount for this line (``quantity * unit_amount``)."""
        return round(self.quantity * self.unit_amount, 2)


class Claim(BaseModel):
    """A complete claim submission - the primary input to the engine.

    This is the *raw* claim as received from the intake channel. The engine
    enriches it with retrieved context, an LLM assessment and rule findings to
    produce a :class:`~claim_engine.models.decision.ClaimDecision`.
    """

    claim_id: str = Field(..., description="Globally-unique claim identifier.")
    member: Member
    claim_type: ClaimType
    diagnosis_codes: list[str] = Field(
        default_factory=list, description="ICD-10 diagnosis codes supporting the claim."
    )
    line_items: list[ClaimLineItem] = Field(default_factory=list)
    provider_id: str = Field(..., description="Identifier of the billing hospital/clinic.")
    provider_in_network: bool = Field(
        default=True, description="Whether the provider is in the insurer's network."
    )
    service_date: date = Field(..., description="Date the service was rendered.")
    submission_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the claim entered the system.",
    )
    clinical_notes: str = Field(
        default="",
        description="Free-text clinical narrative; primary unstructured signal for the LLM.",
    )
    prior_authorization_number: Optional[str] = Field(
        default=None, description="Pre-authorisation reference, when applicable."
    )

    @field_validator("diagnosis_codes")
    @classmethod
    def _strip_blank_codes(cls, codes: list[str]) -> list[str]:
        """Normalise diagnosis codes: uppercased, trimmed, no empties."""
        return [code.strip().upper() for code in codes if code and code.strip()]

    @property
    def total_billed_amount(self) -> float:
        """Sum of all line-item totals (the amount under adjudication, INR)."""
        return round(sum(item.line_total for item in self.line_items), 2)

    def to_context_summary(self) -> str:
        """Render a compact, PHI-free textual summary for retrieval queries.

        Deliberately excludes member identity; only clinical/coding signal is
        included so the resulting embedding query stays HIPAA-safe.
        """
        codes = ", ".join(self.diagnosis_codes) or "none"
        services = "; ".join(item.description for item in self.line_items) or "none"
        return (
            f"Claim type: {self.claim_type.value}. "
            f"Diagnoses: {codes}. "
            f"Services: {services}. "
            f"Provider in-network: {self.provider_in_network}."
        )
