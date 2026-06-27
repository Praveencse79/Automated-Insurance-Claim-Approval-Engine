"""The end-to-end claim adjudication pipeline - the engine's public entry point.

Orchestration sequence for a single claim:

    1. Load claim + policy (data store)
    2. De-identify clinical context (HIPAA compliance layer)
    3. Retrieve grounding policy/guideline context (RAG retrieval)
    4. Reason with the LLM over the grounded context (Claude / mock)
    5. Evaluate deterministic policy & eligibility rules
    6. Detect edge cases
    7. Arbitrate the final decision (approval engine)
    8. Persist the decision + record metrics (audit & monitoring)

All collaborators are injected, so the pipeline is fully unit-testable and the
same code path runs in mock mode and in production.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from claim_engine.compliance.hipaa_redactor import HipaaRedactor
from claim_engine.config.settings import Settings, get_settings
from claim_engine.decisioning.approval_engine import ApprovalEngine
from claim_engine.decisioning.rules_engine import DeterministicRulesEngine
from claim_engine.ingestion.data_store import ClaimDataStore, build_data_store
from claim_engine.knowledge_base import get_default_knowledge_documents
from claim_engine.logging_utils import get_logger, log_with_context
from claim_engine.models.claim import Claim
from claim_engine.models.decision import ClaimDecision, DecisionTrace
from claim_engine.monitoring.edge_case_detector import EdgeCaseDetector
from claim_engine.monitoring.metrics import MetricsCollector
from claim_engine.reasoning.claude_client import build_llm_client
from claim_engine.reasoning.rag_reasoner import RagClaimReasoner
from claim_engine.retrieval.embeddings import build_embedder
from claim_engine.retrieval.retriever import PolicyKnowledgeRetriever
from claim_engine.retrieval.vector_store import InMemoryVectorStore, build_vector_store

logger = get_logger(__name__)


class ClaimNotFoundError(LookupError):
    """Raised when a referenced claim or its policy cannot be located."""


class ClaimApprovalPipeline:
    """Coordinates every component to turn a raw claim into a final decision."""

    def __init__(
        self,
        settings: Settings,
        data_store: ClaimDataStore,
        retriever: PolicyKnowledgeRetriever,
        redactor: HipaaRedactor,
        reasoner: RagClaimReasoner,
        rules_engine: DeterministicRulesEngine,
        approval_engine: ApprovalEngine,
        edge_case_detector: EdgeCaseDetector,
        metrics: MetricsCollector,
    ) -> None:
        self._settings = settings
        self._data_store = data_store
        self._retriever = retriever
        self._redactor = redactor
        self._reasoner = reasoner
        self._rules_engine = rules_engine
        self._approval_engine = approval_engine
        self._edge_case_detector = edge_case_detector
        self._metrics = metrics

    # ------------------------------------------------------------- public API
    def process_claim_by_id(self, claim_id: str) -> ClaimDecision:
        """Adjudicate a claim referenced by id (fetched from the data store).

        Args:
            claim_id: Identifier of the claim to process.

        Returns:
            The finalised :class:`ClaimDecision`.

        Raises:
            ClaimNotFoundError: If the claim or its policy is missing.
        """
        claim = self._data_store.get_claim(claim_id)
        if claim is None:
            raise ClaimNotFoundError(f"Claim '{claim_id}' not found.")
        return self.process_claim(claim)

    def process_claim(self, claim: Claim) -> ClaimDecision:
        """Adjudicate an in-memory claim object end-to-end.

        This is the single, authoritative code path. Each stage contributes to
        a :class:`DecisionTrace` that is attached to the result for full
        auditability.

        Args:
            claim: The claim to adjudicate.

        Returns:
            The finalised :class:`ClaimDecision`.

        Raises:
            ClaimNotFoundError: If the claim's policy cannot be located.
        """
        started_at = time.perf_counter()
        trace = DecisionTrace(used_mock_components=self._active_mock_components())

        policy = self._data_store.get_policy(claim.member.policy_id)
        if policy is None:
            raise ClaimNotFoundError(f"Policy '{claim.member.policy_id}' not found.")

        # Step 2 - de-identify before any external/LLM exposure.
        redaction = self._redactor.build_safe_clinical_context(claim)

        # Step 3 - retrieve grounding context.
        retrieved = self._retriever.retrieve_for_claim(claim)
        trace.retrieved_sources = [ctx.document.source for ctx in retrieved]

        # Step 4 - grounded LLM assessment.
        assessment = self._reasoner.assess(redaction.redacted_text, retrieved)

        # Step 5/6 - deterministic rules + edge-case detection. Edge-case flags
        # are merged into the assessment so the approval engine sees both.
        rule_findings = self._rules_engine.evaluate(claim, policy)
        edge_flags = self._edge_case_detector.detect(claim, policy)
        merged_flags = sorted(set(assessment.edge_case_flags) | set(edge_flags))
        assessment = assessment.model_copy(update={"edge_case_flags": merged_flags})

        trace.rule_findings = rule_findings
        trace.llm_assessment = assessment

        # Step 7 - arbitrate the final decision.
        decision = self._approval_engine.decide(claim, policy, rule_findings, assessment, trace)

        # Finalise trace timing and step 8 - persist + record metrics.
        trace.latency_ms = (time.perf_counter() - started_at) * 1000.0
        decision.trace = trace
        self._data_store.save_decision(decision)
        self._metrics.record_decision(decision, edge_flags)

        log_with_context(
            logger,
            logging.INFO,
            "claim_adjudicated",
            claim_id=claim.claim_id,
            outcome=decision.outcome.value,
            confidence=decision.confidence,
            latency_ms=round(trace.latency_ms, 2),
            edge_cases=merged_flags,
            redactions=redaction.redaction_counts,
        )
        return decision

    @property
    def metrics(self) -> MetricsCollector:
        """Expose the shared metrics collector (for dashboards / the demo)."""
        return self._metrics

    # ------------------------------------------------------------- internals
    def _active_mock_components(self) -> list[str]:
        """List which integrations are currently running in mock mode."""
        s = self._settings
        components = []
        if s.resolve_use_mock(s.claude_credentials_present):
            components.append("claude")
        if s.resolve_use_mock(s.pinecone_credentials_present):
            components.append("pinecone")
        if s.resolve_use_mock(s.snowflake_credentials_present):
            components.append("snowflake")
        return components


def build_pipeline(settings: Optional[Settings] = None) -> ClaimApprovalPipeline:
    """Compose a fully-wired pipeline from configuration.

    This is the composition root: it instantiates every concrete component
    (choosing mock vs real implementations from settings), wires them together
    and seeds the in-memory knowledge base when running offline.

    Args:
        settings: Optional explicit settings; defaults to the cached singleton.

    Returns:
        A ready-to-use :class:`ClaimApprovalPipeline`.
    """
    settings = settings or get_settings()

    data_store = build_data_store(settings)
    embedder = build_embedder(settings)
    vector_store = build_vector_store(settings)
    retriever = PolicyKnowledgeRetriever(embedder=embedder, vector_store=vector_store)

    # Seed the default corpus when using the ephemeral in-memory index so that
    # retrieval works on a cold start without a separate ingestion job.
    if isinstance(vector_store, InMemoryVectorStore):
        retriever.index_documents(get_default_knowledge_documents())

    pipeline = ClaimApprovalPipeline(
        settings=settings,
        data_store=data_store,
        retriever=retriever,
        redactor=HipaaRedactor(),
        reasoner=RagClaimReasoner(build_llm_client(settings)),
        rules_engine=DeterministicRulesEngine(),
        approval_engine=ApprovalEngine(settings),
        edge_case_detector=EdgeCaseDetector(),
        metrics=MetricsCollector(),
    )
    logger.info("Pipeline composed (mock_components=%s).", pipeline._active_mock_components())
    return pipeline
