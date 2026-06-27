"""Data-access layer: the system-of-record for claims, policies and decisions.

Exposes a single :class:`ClaimDataStore` protocol with two interchangeable
implementations selected at runtime by :func:`build_data_store`:

* :class:`InMemoryClaimDataStore` - deterministic fixtures for local/CI runs.
* :class:`SnowflakeClaimDataStore` - production adapter over Snowflake.
"""

from claim_engine.ingestion.data_store import (
    ClaimDataStore,
    InMemoryClaimDataStore,
    build_data_store,
)

__all__ = ["ClaimDataStore", "InMemoryClaimDataStore", "build_data_store"]
