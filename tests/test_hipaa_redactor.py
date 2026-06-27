"""Tests for the HIPAA de-identification layer."""

from __future__ import annotations

from claim_engine.compliance.hipaa_redactor import HipaaRedactor
from claim_engine.models.claim import Claim


def test_redacts_email_phone_and_name() -> None:
    """Emails, phone numbers and honorific names must be scrubbed."""
    redactor = HipaaRedactor()
    report = redactor.redact_text(
        "Contact Mr. John Smith at john.smith@example.com or +91 98765 43210."
    )
    assert "john.smith@example.com" not in report.redacted_text
    assert "John Smith" not in report.redacted_text
    assert "[REDACTED_EMAIL]" in report.redacted_text
    assert report.total_redactions >= 2


def test_safe_clinical_context_excludes_member_identity(clean_claim: Claim) -> None:
    """The clinical context block must never contain the member's name."""
    redactor = HipaaRedactor()
    report = redactor.build_safe_clinical_context(clean_claim)
    assert clean_claim.member.full_name not in report.redacted_text
    assert "Claim type:" in report.redacted_text


def test_empty_text_is_handled() -> None:
    """Empty input yields empty output with no redactions."""
    report = HipaaRedactor().redact_text("")
    assert report.redacted_text == ""
    assert report.total_redactions == 0
