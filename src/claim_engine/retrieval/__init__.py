"""Retrieval layer: embeddings + vector store powering the RAG step."""

from claim_engine.retrieval.embeddings import (
    DeterministicEmbedder,
    EmbeddingProvider,
    build_embedder,
)
from claim_engine.retrieval.retriever import PolicyKnowledgeRetriever
from claim_engine.retrieval.vector_store import (
    InMemoryVectorStore,
    VectorStore,
    build_vector_store,
)

__all__ = [
    "EmbeddingProvider",
    "DeterministicEmbedder",
    "build_embedder",
    "VectorStore",
    "InMemoryVectorStore",
    "build_vector_store",
    "PolicyKnowledgeRetriever",
]
