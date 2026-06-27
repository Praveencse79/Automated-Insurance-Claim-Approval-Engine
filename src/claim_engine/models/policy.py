"""Model describing an insurance policy and its coverage terms."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from claim_engine.models.enums import ClaimType


class Policy(BaseModel):
    """The coverage contract a claim is adjudicated against.

    Holds the structured, deterministic coverage terms used by the rules
    engine (limits, waiting periods, exclusions). The *narrative* clauses of
    the policy live in the vector knowledge base and are retrieved separately
    for the LLM.
    """

    policy_id: str = Field(..., description="Unique policy identifier.")
    product_name: str = Field(..., description="Marketed product name.")
    effective_date: date = Field(..., description="Date coverage began.")
    expiry_date: date = Field(..., description="Date coverage ends.")
    annual_sum_insured: float = Field(..., ge=0, description="Maximum annual payout (INR).")
    amount_consumed_ytd: float = Field(
        default=0.0, ge=0, description="Sum-insured already used this policy year (INR)."
    )
    covered_claim_types: list[ClaimType] = Field(default_factory=list)
    excluded_procedure_codes: list[str] = Field(
        default_factory=list, description="Procedure codes explicitly not covered."
    )
    waiting_period_days: int = Field(
        default=0, ge=0, description="Days after effective date before claims are payable."
    )
    co_payment_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Member's cost-share fraction (0-1)."
    )
    requires_prior_auth_above: Optional[float] = Field(
        default=None, description="Amount (INR) above which prior authorisation is mandatory."
    )

    @property
    def remaining_sum_insured(self) -> float:
        """Sum insured still available this policy year (INR)."""
        return round(max(self.annual_sum_insured - self.amount_consumed_ytd, 0.0), 2)

    def is_active_on(self, service_date: date) -> bool:
        """Return True if the policy was in force on ``service_date``."""
        return self.effective_date <= service_date <= self.expiry_date

    def covers_claim_type(self, claim_type: ClaimType) -> bool:
        """Return True if ``claim_type`` is within this policy's coverage."""
        return claim_type in self.covered_claim_types
