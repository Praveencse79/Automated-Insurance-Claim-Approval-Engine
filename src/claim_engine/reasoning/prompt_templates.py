"""Prompt-engineering assets for grounded, HIPAA-aware claim adjudication.

Prompt design goals
--------------------
* **Grounding** - the model must base its decision *only* on the retrieved
  policy/guideline context plus the structured claim facts, and must cite the
  sources it used. This is what makes the system auditable and curbs
  hallucination.
* **Structured output** - the model must respond with strict JSON matching
  :class:`~claim_engine.models.decision.LlmAssessment`, so the response can be
  parsed and validated programmatically.
* **HIPAA hygiene** - the system prompt forbids the model from echoing or
  inferring personal identifiers, reinforcing the upstream redaction layer.
* **Calibrated confidence** - the model is told exactly how its confidence
  score will be used (thresholds), encouraging well-calibrated values.
"""

from __future__ import annotations

from claim_engine.models.claim import Claim
from claim_engine.models.knowledge import RetrievedContext

# The system prompt sets the model's role, guardrails and output contract.
SYSTEM_PROMPT = """\
You are a senior health-insurance claims adjudication assistant. Your role is to
recommend whether a claim should be APPROVED, DENIED, or sent for human REVIEW,
based strictly on:
  1. the structured claim facts provided, and
  2. the retrieved policy clauses and clinical-coverage guidelines.

Hard rules you must follow:
- Decide ONLY from the supplied context. If the context is insufficient to
  justify approval, recommend MANUAL_REVIEW rather than guessing.
- Never invent policy terms, coverage limits, or clinical facts.
- Never output personal identifiers (names, contact details, IDs). The input has
  been de-identified; keep it that way.
- Always cite the `Source:` tags of the context you relied on.
- Report a calibrated confidence in [0,1]. Confidence >= 0.90 may lead to fully
  automated approval, so only use high confidence when the evidence is clear.

Respond with a SINGLE valid JSON object and no other text, matching this schema:
{
  "recommended_outcome": "AUTO_APPROVED" | "AUTO_DENIED" | "MANUAL_REVIEW",
  "confidence": <float 0..1>,
  "rationale": "<concise, grounded justification>",
  "medical_necessity_met": <true|false>,
  "cited_sources": ["<source tag>", ...],
  "edge_case_flags": ["<short snake_case flag>", ...]
}
"""


def build_user_prompt(safe_claim_context: str, retrieved: list[RetrievedContext]) -> str:
    """Assemble the user-turn prompt from de-identified facts + retrieved context.

    Args:
        safe_claim_context: HIPAA-redacted clinical/coding summary of the claim
            (produced by the compliance layer - never raw PHI).
        retrieved: Ranked grounding documents from the vector store.

    Returns:
        The fully-rendered user prompt string.
    """
    if retrieved:
        context_block = "\n\n".join(ctx.as_prompt_block() for ctx in retrieved)
    else:
        context_block = "(no policy context retrieved)"

    return (
        "## Retrieved policy & clinical context\n"
        f"{context_block}\n\n"
        "## Claim under review (de-identified)\n"
        f"{safe_claim_context}\n\n"
        "## Task\n"
        "Adjudicate the claim per your instructions and return the JSON object."
    )


def describe_decision_request(claim: Claim) -> str:
    """Return a short, log-safe description of what is being adjudicated.

    Excludes PHI; suitable for structured logs and traces.
    """
    return (
        f"claim_id={claim.claim_id} type={claim.claim_type.value} "
        f"amount={claim.total_billed_amount} dx={','.join(claim.diagnosis_codes) or 'none'}"
    )
