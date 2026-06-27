"""Models for the retrieval (RAG) knowledge base and its query results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    """A single chunk of authoritative reference text stored in the vector DB.

    The corpus is composed of policy clauses, clinical-coverage guidelines,
    medical-necessity criteria and regulatory notes. Each document is embedded
    once and retrieved by semantic similarity at decision time.
    """

    document_id: str = Field(..., description="Stable unique id (used as the vector id).")
    title: str = Field(..., description="Short human-readable title.")
    content: str = Field(..., description="The full text that gets embedded and retrieved.")
    source: str = Field(..., description="Provenance, e.g. 'policy:GOLD-2024' or 'guideline:CGL-12'.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Filterable attributes (claim_type, jurisdiction, version, ...).",
    )


class RetrievedContext(BaseModel):
    """A knowledge document returned by a similarity search, with its score."""

    document: KnowledgeDocument
    similarity_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Cosine similarity between query and document (-1..1)."
    )

    def as_prompt_block(self) -> str:
        """Format the context for inclusion in the LLM prompt.

        Includes the source so the model can produce grounded, *citable*
        reasoning rather than hallucinating policy terms.
        """
        return (
            f"[Source: {self.document.source} | relevance={self.similarity_score:.2f}]\n"
            f"{self.document.title}\n{self.document.content}"
        )
