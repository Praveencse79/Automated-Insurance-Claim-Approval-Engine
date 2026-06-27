"""Domain models: the typed vocabulary shared by every layer of the engine.

These Pydantic models form the *contract* between components. By centralising
them we ensure the ingestion, retrieval, reasoning, decisioning and monitoring
layers all speak the same language and that data is validated at every
boundary.
"""

from claim_engine.models.claim import Claim, ClaimLineItem, Member
from claim_engine.models.decision import (
    ClaimDecision,
    DecisionTrace,
    LlmAssessment,
    RuleFinding,
)
from claim_engine.models.enums import (
    ClaimType,
    DecisionOutcome,
    Gender,
    RuleSeverity,
)
from claim_engine.models.knowledge import KnowledgeDocument, RetrievedContext
from claim_engine.models.policy import Policy

__all__ = [
    "Claim",
    "ClaimLineItem",
    "Member",
    "Policy",
    "KnowledgeDocument",
    "RetrievedContext",
    "LlmAssessment",
    "RuleFinding",
    "DecisionTrace",
    "ClaimDecision",
    "ClaimType",
    "DecisionOutcome",
    "Gender",
    "RuleSeverity",
]
