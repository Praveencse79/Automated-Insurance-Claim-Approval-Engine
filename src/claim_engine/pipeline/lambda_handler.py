"""AWS Lambda entry point for serverless, event-driven claim adjudication.

Deployment model
----------------
The function is invoked per-claim (or per small batch) from an event source -
typically an SQS queue or API Gateway fed by the claims intake system. Snowflake
streams new claims, a small dispatcher enqueues claim ids, and Lambda scales out
horizontally to hit the 50K-claims/day throughput target while keeping
per-claim latency sub-second.

Cold-start optimisation
-----------------------
The pipeline is built **once per container** and cached in a module-level
global, so warm invocations skip all client/Pinecone/Snowflake initialisation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from claim_engine.config.settings import get_settings
from claim_engine.logging_utils import configure_logging, get_logger
from claim_engine.pipeline.claim_pipeline import ClaimApprovalPipeline, build_pipeline

# Cached across warm invocations of the same Lambda container.
_PIPELINE: Optional[ClaimApprovalPipeline] = None

logger = get_logger(__name__)


def _get_pipeline() -> ClaimApprovalPipeline:
    """Lazily build and cache the pipeline for reuse across warm invocations."""
    global _PIPELINE
    if _PIPELINE is None:
        settings = get_settings()
        configure_logging(settings.log_level)
        _PIPELINE = build_pipeline(settings)
    return _PIPELINE


def _extract_claim_ids(event: dict[str, Any]) -> list[str]:
    """Normalise the supported event shapes into a flat list of claim ids.

    Supports:
        * direct invoke: ``{"claim_id": "CLM-1"}`` or ``{"claim_ids": [...]}``
        * SQS batch: ``{"Records": [{"body": "{\\"claim_id\\": \\"CLM-1\\"}"}]}``

    Args:
        event: The raw Lambda event payload.

    Returns:
        A list of claim ids to process.
    """
    if "claim_id" in event:
        return [event["claim_id"]]
    if "claim_ids" in event:
        return list(event["claim_ids"])
    if "Records" in event:  # SQS-style batch
        claim_ids: list[str] = []
        for record in event["Records"]:
            body = record.get("body", "{}")
            payload = json.loads(body) if isinstance(body, str) else body
            if "claim_id" in payload:
                claim_ids.append(payload["claim_id"])
        return claim_ids
    return []


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """AWS Lambda handler: adjudicate one or more claims from the event.

    Args:
        event: Lambda event (direct invoke or SQS batch).
        context: Lambda context object (unused; present for the AWS contract).

    Returns:
        An API-Gateway-compatible response with each claim's decision summary
        and a live metrics snapshot.
    """
    pipeline = _get_pipeline()
    claim_ids = _extract_claim_ids(event)

    results: list[dict[str, Any]] = []
    for claim_id in claim_ids:
        try:
            decision = pipeline.process_claim_by_id(claim_id)
            results.append(
                {
                    "claim_id": decision.claim_id,
                    "outcome": decision.outcome.value,
                    "confidence": decision.confidence,
                    "approved_amount": decision.approved_amount,
                    "requires_human_review": decision.requires_human_review,
                    "summary": decision.summary,
                }
            )
        except Exception as error:  # one bad claim must not fail the batch
            logger.log(logging.ERROR, "Failed to process claim %s: %s", claim_id, error)
            results.append({"claim_id": claim_id, "error": str(error)})

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "processed": len(results),
                "decisions": results,
                "metrics": pipeline.metrics.snapshot().as_dict(),
            }
        ),
    }
