"""Default knowledge-base corpus for the RAG retrieval layer.

In production this corpus is curated by clinical and policy teams and loaded
into Pinecone once. For local/mock runs we ship a small but representative set
of policy clauses and clinical-coverage guidelines so retrieval returns
meaningful grounding context out of the box.

Each document is tagged with a ``claim_type`` in its metadata so the retriever
can apply a coarse pre-filter before semantic ranking.
"""

from __future__ import annotations

from claim_engine.models.enums import ClaimType
from claim_engine.models.knowledge import KnowledgeDocument


def get_default_knowledge_documents() -> list[KnowledgeDocument]:
    """Return the built-in policy / clinical-guideline corpus.

    Returns:
        A list of :class:`KnowledgeDocument` ready to be embedded and indexed.
    """
    return [
        KnowledgeDocument(
            document_id="KB-INPATIENT-PNEUMONIA",
            title="Inpatient coverage: lower respiratory tract infections",
            content=(
                "Hospitalisation for bacterial pneumonia (ICD-10 J18.x) is covered for "
                "in-network providers when supported by imaging or laboratory confirmation. "
                "Initial hospital care codes (99221-99223) and supportive IV therapy are "
                "payable subject to the policy co-payment and remaining sum insured."
            ),
            source="guideline:CGL-RESP-01",
            metadata={"claim_type": ClaimType.INPATIENT.value, "version": "2025.1"},
        ),
        KnowledgeDocument(
            document_id="KB-EXCLUSION-COSMETIC",
            title="General exclusions: cosmetic and elective procedures",
            content=(
                "Purely cosmetic or aesthetic procedures, including elective rhinoplasty "
                "(procedure code COSMETIC-001), are explicitly excluded from coverage unless "
                "medically necessary following accidental trauma documented in the record."
            ),
            source="policy:GOLD-2024-EXCL",
            metadata={"claim_type": ClaimType.OUTPATIENT.value, "version": "2024.4"},
        ),
        KnowledgeDocument(
            document_id="KB-CRITICAL-CARE-REVIEW",
            title="Critical care and unspecified-diagnosis review criteria",
            content=(
                "Critical-care services (99291) and claims carrying unspecified diagnoses "
                "(e.g. ICD-10 R69) require documented medical necessity. Where the primary "
                "diagnosis is unclear or documentation is incomplete, the claim must be "
                "routed for manual clinical review rather than auto-adjudicated."
            ),
            source="guideline:CGL-CRIT-07",
            metadata={"claim_type": ClaimType.INPATIENT.value, "version": "2025.1"},
        ),
        KnowledgeDocument(
            document_id="KB-PRIOR-AUTH",
            title="Prior authorisation requirements for high-value services",
            content=(
                "Any single claim exceeding INR 200,000 requires a valid prior-authorisation "
                "reference number. Claims above this threshold submitted without prior "
                "authorisation should not be auto-approved and require human verification."
            ),
            source="policy:GOLD-2024-AUTH",
            metadata={"claim_type": ClaimType.INPATIENT.value, "version": "2024.4"},
        ),
        KnowledgeDocument(
            document_id="KB-NETWORK-COSTSHARE",
            title="Network status and member cost-sharing",
            content=(
                "In-network providers are reimbursed at the contracted rate with the standard "
                "co-payment. Out-of-network providers attract higher member cost-share and "
                "additional documentation scrutiny before payment."
            ),
            source="policy:GOLD-2024-NET",
            metadata={"claim_type": ClaimType.OUTPATIENT.value, "version": "2024.4"},
        ),
        KnowledgeDocument(
            document_id="KB-DIAGNOSTIC-COVERAGE",
            title="Diagnostic and imaging coverage",
            content=(
                "Medically indicated diagnostic tests and imaging ordered to investigate a "
                "documented symptom or diagnosis are covered. Screening tests without a "
                "supporting indication may be subject to sub-limits."
            ),
            source="guideline:CGL-DIAG-03",
            metadata={"claim_type": ClaimType.DIAGNOSTIC.value, "version": "2025.1"},
        ),
    ]
