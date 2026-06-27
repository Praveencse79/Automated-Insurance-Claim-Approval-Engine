"""Automated Insurance Claim Approval Engine.

A Retrieval-Augmented Generation (RAG) system that adjudicates health-insurance
claims. It combines:

* **LangChain** for LLM orchestration,
* **Anthropic Claude** for grounded clinical/policy reasoning,
* **Pinecone** for semantic retrieval of policy and clinical-guideline context,
* **Snowflake** as the system-of-record data warehouse, and
* **AWS Lambda** as the serverless execution surface.

The public entry point is :class:`claim_engine.pipeline.claim_pipeline.ClaimApprovalPipeline`.
"""

from claim_engine.__version__ import __version__

__all__ = ["__version__"]
