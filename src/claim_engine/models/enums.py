"""Enumerations used across the domain.

Using explicit ``str``-backed enums (rather than free-form strings) gives us
validation, IDE autocompletion and stable serialised values for audit logs.
"""

from __future__ import annotations

from enum import Enum


class ClaimType(str, Enum):
    """High-level category of a health-insurance claim."""

    INPATIENT = "INPATIENT"            # Hospitalisation / admission
    OUTPATIENT = "OUTPATIENT"          # Clinic visit, no admission
    PHARMACY = "PHARMACY"              # Prescription drugs
    DIAGNOSTIC = "DIAGNOSTIC"          # Lab tests, imaging
    DENTAL = "DENTAL"
    MATERNITY = "MATERNITY"
    EMERGENCY = "EMERGENCY"


class Gender(str, Enum):
    """Member gender (used by some clinical-eligibility rules)."""

    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class DecisionOutcome(str, Enum):
    """Terminal outcome assigned to a processed claim.

    * ``AUTO_APPROVED``  - approved by the engine with no human in the loop.
    * ``AUTO_DENIED``    - denied by a hard policy/eligibility rule.
    * ``MANUAL_REVIEW``  - routed to a human adjudicator (low confidence,
      high value, or a flagged edge case).
    """

    AUTO_APPROVED = "AUTO_APPROVED"
    AUTO_DENIED = "AUTO_DENIED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class RuleSeverity(str, Enum):
    """Severity of a deterministic rule finding.

    * ``BLOCKER``  - immediately disqualifies auto-approval (hard denial).
    * ``WARNING``  - lowers confidence and may trigger manual review.
    * ``INFO``     - informational; recorded for the audit trail only.
    """

    BLOCKER = "BLOCKER"
    WARNING = "WARNING"
    INFO = "INFO"
