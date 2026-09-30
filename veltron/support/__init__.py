"""Customer-support domain layer: knowledge base, intent classification, entity
extraction, escalation policy and response grounding."""

from .knowledge_base import (
    ALL_DOCS,
    DEMO_DISCLAIMER,
    kb_documents,
    load_knowledge_base,
)

__all__ = ["ALL_DOCS", "DEMO_DISCLAIMER", "kb_documents", "load_knowledge_base"]
