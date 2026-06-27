"""Vector store abstraction with in-memory and Pinecone implementations.

The retriever depends on the :class:`VectorStore` protocol. In mock mode an
exact, brute-force cosine-similarity store is used; in production the same
interface is served by Pinecone's approximate-nearest-neighbour index.
"""

from __future__ import annotations

import math
from typing import Optional, Protocol, runtime_checkable

from claim_engine.config.settings import Settings
from claim_engine.logging_utils import get_logger
from claim_engine.models.knowledge import KnowledgeDocument, RetrievedContext

logger = get_logger(__name__)


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Args:
        vector_a: First vector.
        vector_b: Second vector.

    Returns:
        Similarity in ``[-1, 1]``; ``0.0`` if either vector has zero magnitude.
    """
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@runtime_checkable
class VectorStore(Protocol):
    """Upsert/query contract for a vector index of knowledge documents."""

    def upsert_documents(
        self, documents: list[KnowledgeDocument], vectors: list[list[float]]
    ) -> None:
        """Insert or update documents and their pre-computed embeddings."""
        ...

    def query(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedContext]:
        """Return the ``top_k`` most similar documents to ``query_vector``."""
        ...


class InMemoryVectorStore:
    """Exact brute-force cosine-similarity store for offline/mock use.

    Holds every document vector in memory and scans linearly on query. Perfect
    for unit tests and small demo corpora; not intended for production scale
    (that is what Pinecone is for).
    """

    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeDocument] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert_documents(
        self, documents: list[KnowledgeDocument], vectors: list[list[float]]
    ) -> None:
        """Store/overwrite documents alongside their embeddings."""
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must be the same length")
        for document, vector in zip(documents, vectors):
            self._documents[document.document_id] = document
            self._vectors[document.document_id] = vector
        logger.info("Upserted %d documents into in-memory vector store.", len(documents))

    def query(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedContext]:
        """Linearly score all documents and return the best ``top_k`` matches.

        Args:
            query_vector: Embedding of the query text.
            top_k: Maximum number of results to return.
            metadata_filter: Optional exact-match filter on document metadata.

        Returns:
            Results sorted by descending similarity score.
        """
        scored: list[RetrievedContext] = []
        for document_id, document in self._documents.items():
            if metadata_filter and not self._matches_filter(document, metadata_filter):
                continue
            score = cosine_similarity(query_vector, self._vectors[document_id])
            scored.append(RetrievedContext(document=document, similarity_score=score))

        scored.sort(key=lambda ctx: ctx.similarity_score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _matches_filter(document: KnowledgeDocument, metadata_filter: dict) -> bool:
        """Return True if every key/value in the filter matches the document."""
        return all(document.metadata.get(key) == value for key, value in metadata_filter.items())

    @property
    def document_count(self) -> int:
        """Number of documents currently indexed (inspection helper)."""
        return len(self._documents)


class PineconeVectorStore:
    """Production vector store backed by Pinecone (serverless index).

    Lazily imports the Pinecone client and creates the index on first use if it
    does not already exist. The metadata payload stores the document text so a
    query can return full :class:`KnowledgeDocument` objects without a second
    round-trip to the warehouse.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._index = None  # connected lazily

    def _get_index(self):  # noqa: ANN202 - vendor index object
        """Connect to (creating if needed) the configured Pinecone index."""
        if self._index is not None:
            return self._index
        from pinecone import Pinecone, ServerlessSpec  # optional heavy dependency

        client = Pinecone(api_key=self._settings.pinecone_api_key)
        existing = {idx["name"] for idx in client.list_indexes()}
        if self._settings.pinecone_index not in existing:
            client.create_index(
                name=self._settings.pinecone_index,
                dimension=self._settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )
            logger.info("Created Pinecone index %s", self._settings.pinecone_index)
        self._index = client.Index(self._settings.pinecone_index)
        return self._index

    def upsert_documents(
        self, documents: list[KnowledgeDocument], vectors: list[list[float]]
    ) -> None:
        """Upsert documents (text carried in metadata) into Pinecone."""
        index = self._get_index()
        payload = [
            {
                "id": document.document_id,
                "values": vector,
                "metadata": {
                    "title": document.title,
                    "content": document.content,
                    "source": document.source,
                    **document.metadata,
                },
            }
            for document, vector in zip(documents, vectors)
        ]
        index.upsert(vectors=payload)

    def query(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedContext]:
        """Run an ANN similarity query against Pinecone and map the results."""
        index = self._get_index()
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter or None,
        )
        results: list[RetrievedContext] = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            document = KnowledgeDocument(
                document_id=match["id"],
                title=meta.get("title", ""),
                content=meta.get("content", ""),
                source=meta.get("source", ""),
                metadata={
                    k: v for k, v in meta.items() if k not in {"title", "content", "source"}
                },
            )
            results.append(
                RetrievedContext(document=document, similarity_score=float(match["score"]))
            )
        return results


def build_vector_store(settings: Settings) -> VectorStore:
    """Factory returning the appropriate vector store for the environment.

    Args:
        settings: Active application settings.

    Returns:
        An :class:`InMemoryVectorStore` in mock mode, else a Pinecone adapter.
    """
    if settings.resolve_use_mock(settings.pinecone_credentials_present):
        logger.info("Vector store: using in-memory cosine store.")
        return InMemoryVectorStore()
    logger.info("Vector store: using Pinecone index %s.", settings.pinecone_index)
    return PineconeVectorStore(settings)
