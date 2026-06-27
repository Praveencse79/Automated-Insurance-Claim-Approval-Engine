"""HIPAA Safe-Harbor de-identification for text sent to the LLM.

Why this exists
---------------
Under the HIPAA Privacy Rule (45 CFR §164.514(b)(2)), Protected Health
Information (PHI) must be de-identified before it leaves the trust boundary of
the covered entity. Anthropic's Claude API is an external processor, so *every*
free-text payload (clinical notes, member context) is routed through this
redactor first.

The redactor implements a **defence-in-depth** strategy:

1. **Structured stripping** - identity fields from typed models are never
   placed in the prompt to begin with (see :meth:`build_safe_clinical_context`).
2. **Pattern-based scrubbing** - a regex pass removes the common Safe-Harbor
   identifiers (names following honorifics, emails, phones, MRNs, SSNs,
   Aadhaar, dates, URLs, IPs, etc.) from any residual free text.

This is intentionally conservative: false positives (over-redaction) are
acceptable, false negatives (leaked PHI) are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from claim_engine.models.claim import Claim


@dataclass
class RedactionReport:
    """Summary of a redaction pass, attached to the audit trail.

    Attributes:
        redacted_text: The de-identified text safe to send downstream.
        redaction_counts: Per-identifier-type count of substitutions made.
    """

    redacted_text: str
    redaction_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_redactions(self) -> int:
        """Total number of identifiers scrubbed in this pass."""
        return sum(self.redaction_counts.values())


# Ordered (label, compiled-pattern, placeholder) triples. Order matters: more
# specific patterns (e.g. email) run before greedier ones (e.g. names).
_REDACTION_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    ("URL", re.compile(r"\bhttps?://\S+\b"), "[REDACTED_URL]"),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
    # US SSN (###-##-####)
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # Indian Aadhaar (#### #### ####)
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[REDACTED_AADHAAR]"),
    # Phone numbers (international / Indian / US formats)
    (
        "PHONE",
        re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}\b"),
        "[REDACTED_PHONE]",
    ),
    # Medical record numbers / member ids (MRN12345, ID: 99887)
    ("MRN", re.compile(r"\b(?:MRN|ID|MEMBER)[:#\s-]*\w+\b", re.IGNORECASE), "[REDACTED_ID]"),
    # Explicit calendar dates (1990-05-12, 12/05/1990)
    (
        "DATE",
        re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
        "[REDACTED_DATE]",
    ),
    # Person names that follow an honorific (Mr./Mrs./Dr. John Smith)
    (
        "NAME",
        re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Miss|Mx)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"),
        "[REDACTED_NAME]",
    ),
]


class HipaaRedactor:
    """Stateless utility that de-identifies free text and builds safe prompts.

    The class is stateless and therefore thread-safe; a single shared instance
    can be reused across the whole process.
    """

    def redact_text(self, text: str) -> RedactionReport:
        """Scrub Safe-Harbor identifiers from an arbitrary free-text string.

        Args:
            text: Raw text that may contain PHI/PII.

        Returns:
            A :class:`RedactionReport` containing the cleaned text and a
            per-type count of how many identifiers were removed.
        """
        if not text:
            return RedactionReport(redacted_text="", redaction_counts={})

        counts: dict[str, int] = {}
        cleaned = text
        for label, pattern, placeholder in _REDACTION_RULES:
            cleaned, num_subs = pattern.subn(placeholder, cleaned)
            if num_subs:
                counts[label] = counts.get(label, 0) + num_subs
        return RedactionReport(redacted_text=cleaned, redaction_counts=counts)

    def build_safe_clinical_context(self, claim: Claim) -> RedactionReport:
        """Produce a HIPAA-safe textual context block for a claim.

        Only clinical and coding signal is included - never member identity.
        Any residual identifiers in the free-text clinical notes are then
        scrubbed by :meth:`redact_text` as a second line of defence.

        Args:
            claim: The claim under adjudication.

        Returns:
            A :class:`RedactionReport` whose ``redacted_text`` is safe to embed
            in the LLM prompt.
        """
        # Structured fields are assembled WITHOUT any identity attributes.
        structured_block = (
            f"Claim type: {claim.claim_type.value}\n"
            f"Member age: {claim.member.age_years} years\n"
            f"Member gender: {claim.member.gender.value}\n"
            f"Diagnosis codes: {', '.join(claim.diagnosis_codes) or 'none'}\n"
            f"Billed services: "
            f"{'; '.join(item.description for item in claim.line_items) or 'none'}\n"
            f"Total billed amount (INR): {claim.total_billed_amount}\n"
            f"Provider in-network: {claim.provider_in_network}\n"
        )

        notes_report = self.redact_text(claim.clinical_notes)
        combined = f"{structured_block}\nClinical notes: {notes_report.redacted_text}"
        return RedactionReport(
            redacted_text=combined,
            redaction_counts=notes_report.redaction_counts,
        )
