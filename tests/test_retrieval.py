"""Tests for the embedding + vector-store retrieval layer."""

from __future__ import annotations

from claim_engine.knowledge_base import get_default_knowledge_documents
from claim_engine.retrieval.embeddings import DeterministicEmbedder
from claim_engine.retrieval.retriever import PolicyKnowledgeRetriever
from claim_engine.retrieval.vector_store import InMemoryVectorStore, cosine_similarity


def test_deterministic_embedder_is_stable_and_normalised() -> None:
    """Same input yields the same unit-norm vector across calls."""
    embedder = DeterministicEmbedder(dimension=256)
    v1 = embedder.embed_text("acute bacterial pneumonia")
    v2 = embedder.embed_text("acute bacterial pneumonia")
    assert v1 == v2
    assert abs(cosine_similarity(v1, v1) - 1.0) < 1e-9


def test_similar_text_scores_higher_than_unrelated() -> None:
    """Shared vocabulary should raise cosine similarity."""
    embedder = DeterministicEmbedder(dimension=512)
    base = embedder.embed_text("pneumonia hospital inpatient antibiotics")
    similar = embedder.embed_text("inpatient hospital care for pneumonia")
    unrelated = embedder.embed_text("dental crown replacement procedure")
    assert cosine_similarity(base, similar) > cosine_similarity(base, unrelated)


def test_retriever_returns_relevant_documents(clean_claim) -> None:
    """Retrieval for a pneumonia inpatient claim surfaces the respiratory guideline."""
    embedder = DeterministicEmbedder(dimension=1536)
    store = InMemoryVectorStore()
    retriever = PolicyKnowledgeRetriever(embedder=embedder, vector_store=store)
    retriever.index_documents(get_default_knowledge_documents())

    results = retriever.retrieve_for_claim(clean_claim, top_k=3)
    assert results, "expected at least one retrieved document"
    sources = {ctx.document.document_id for ctx in results}
    assert "KB-INPATIENT-PNEUMONIA" in sources
