"""Embedding providers that turn text into dense vectors for retrieval.

The retrieval layer depends on the :class:`EmbeddingProvider` protocol only.
Two implementations are supplied:

* :class:`DeterministicEmbedder` - a dependency-free, hash-seeded embedder used
  in mock mode. It is *deterministic* (same text -> same vector) and roughly
  semantically sensitive (shared tokens raise cosine similarity), which is
  enough to exercise and test the full RAG path offline.
* :class:`ManagedApiEmbedder` - a thin adapter sketch for a hosted embedding
  model (e.g. Voyage/OpenAI/Bedrock). Swapped in automatically when real
  credentials are present.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from claim_engine.config.settings import Settings
from claim_engine.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract for any component that maps text to a fixed-length vector."""

    @property
    def dimension(self) -> int:
        """Dimensionality of the vectors this provider emits."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string into a unit-normalised vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings at once (override for true batched APIs)."""
        ...


class DeterministicEmbedder:
    """Hash-based bag-of-words embedder for offline/mock use.

    Each token is hashed into the vector space and accumulated; the result is
    L2-normalised so cosine similarity reduces to a dot product. Documents that
    share vocabulary therefore score higher - good enough to validate ranking
    logic without any network calls or model downloads.
    """

    def __init__(self, dimension: int = 1536) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Vector dimensionality (matches the configured Pinecone index)."""
        return self._dimension

    def _hash_token_to_index(self, token: str) -> int:
        """Map a token to a stable bucket in ``[0, dimension)`` via SHA-256."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Embed ``text`` into a unit-normalised dense vector.

        Args:
            text: Input string.

        Returns:
            A list of floats of length :pyattr:`dimension`. Empty/whitespace
            input yields a zero vector.
        """
        vector = [0.0] * self._dimension
        tokens = _TOKEN_PATTERN.findall(text.lower())
        for token in tokens:
            index = self._hash_token_to_index(token)
            # Sign derived from the token hash keeps vectors expressive.
            sign = 1.0 if (hash(token) & 1) == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings (sequentially for the mock implementation)."""
        return [self.embed_text(text) for text in texts]


class ManagedApiEmbedder:
    """Adapter for a hosted embedding model (production path).

    Kept as a thin sketch: wire in the concrete SDK call (Voyage, OpenAI,
    Bedrock Titan, ...) where indicated. The rest of the system is agnostic to
    which provider sits behind this interface.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._dimension = settings.embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Wire a hosted embedding API here; use mock mode for local runs."
        )


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Factory returning the appropriate embedder for the environment.

    Args:
        settings: Active application settings.

    Returns:
        A :class:`DeterministicEmbedder` in mock mode, else a managed adapter.
    """
    if settings.resolve_use_mock(settings.pinecone_credentials_present):
        logger.info("Embedder: using deterministic in-memory embedder.")
        return DeterministicEmbedder(dimension=settings.embedding_dimension)
    logger.info("Embedder: using managed API embedder.")
    return ManagedApiEmbedder(settings)
