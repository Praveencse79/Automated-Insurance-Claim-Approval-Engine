#!/usr/bin/env python3
"""Seed the vector knowledge base with the default policy/guideline corpus.

Usage:
    python scripts/seed_knowledge_base.py

In production (real Pinecone credentials configured) this is a one-off / CI
ingestion job that embeds the curated corpus and upserts it into the index. In
mock mode it indexes into the ephemeral in-memory store and prints a summary so
the workflow can be verified locally.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claim_engine.config.settings import get_settings  # noqa: E402
from claim_engine.knowledge_base import get_default_knowledge_documents  # noqa: E402
from claim_engine.logging_utils import configure_logging  # noqa: E402
from claim_engine.retrieval.embeddings import build_embedder  # noqa: E402
from claim_engine.retrieval.retriever import PolicyKnowledgeRetriever  # noqa: E402
from claim_engine.retrieval.vector_store import build_vector_store  # noqa: E402


def main() -> None:
    """Embed and upsert the default knowledge corpus into the vector store."""
    settings = get_settings()
    configure_logging(settings.log_level)

    embedder = build_embedder(settings)
    vector_store = build_vector_store(settings)
    retriever = PolicyKnowledgeRetriever(embedder=embedder, vector_store=vector_store)

    documents = get_default_knowledge_documents()
    retriever.index_documents(documents)

    print(f"Indexed {len(documents)} knowledge documents into the vector store.")
    for document in documents:
        print(f"  - {document.document_id} ({document.source})")


if __name__ == "__main__":
    main()
