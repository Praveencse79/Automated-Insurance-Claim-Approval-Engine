"""High-level retriever tying the embedder and vector store together.

This is the 'R' in RAG: given a claim, it builds a HIPAA-safe query, embeds it,
and fetches the most relevant policy clauses / clinical guidelines to ground the
LLM's reasoning.
"""

from __future__ import annotations

from typing import Optional

from claim_engine.logging_utils import get_logger
from claim_engine.models.claim import Claim
from claim_engine.models.knowledge import KnowledgeDocument, RetrievedContext
from claim_engine.retrieval.embeddings import EmbeddingProvider
from claim_engine.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class PolicyKnowledgeRetriever:
    """Retrieves grounding context for a claim from the knowledge base.

    Collaborators are injected (embedder + vector store), keeping this class
    free of any provider-specific details and trivially unit-testable.
    """

    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def index_documents(self, documents: list[KnowledgeDocument]) -> None:
        """Embed and upsert a batch of knowledge documents into the store.

        Args:
            documents: Policy clauses / guidelines to make searchable.
        """
        if not documents:
            return
        vectors = self._embedder.embed_batch([doc.content for doc in documents])
        self._vector_store.upsert_documents(documents, vectors)
        logger.info("Indexed %d knowledge documents.", len(documents))

    def retrieve_for_claim(
        self, claim: Claim, top_k: int = 4, filter_by_claim_type: bool = True
    ) -> list[RetrievedContext]:
        """Fetch the most relevant grounding documents for a claim.

        Args:
            claim: The claim under adjudication.
            top_k: Number of documents to retrieve.
            filter_by_claim_type: When True, restricts retrieval to documents
                tagged with the claim's type (plus untagged/global docs are
                matched via a separate pass and merged).

        Returns:
            A ranked list of :class:`RetrievedContext`.
        """
        query_text = claim.to_context_summary()
        query_vector = self._embedder.embed_text(query_text)

        metadata_filter: Optional[dict] = None
        if filter_by_claim_type:
            metadata_filter = {"claim_type": claim.claim_type.value}

        results = self._vector_store.query(
            query_vector=query_vector, top_k=top_k, metadata_filter=metadata_filter
        )

        # If a strict type filter yields too little context, fall back to an
        # unfiltered search so the LLM is never starved of grounding material.
        if filter_by_claim_type and len(results) < top_k:
            unfiltered = self._vector_store.query(query_vector=query_vector, top_k=top_k)
            results = self._merge_unique(results, unfiltered, top_k)

        logger.debug("Retrieved %d context documents for claim %s", len(results), claim.claim_id)
        return results

    @staticmethod
    def _merge_unique(
        primary: list[RetrievedContext], secondary: list[RetrievedContext], limit: int
    ) -> list[RetrievedContext]:
        """Merge two result lists, de-duplicating by document id, keeping order."""
        seen = {ctx.document.document_id for ctx in primary}
        merged = list(primary)
        for ctx in secondary:
            if ctx.document.document_id not in seen:
                merged.append(ctx)
                seen.add(ctx.document.document_id)
            if len(merged) >= limit:
                break
        return merged[:limit]
