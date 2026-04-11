"""
Orchestration Layer - Composes RAG + Evaluation with optional advanced features.
"""

from .pipeline import RAGEvaluationPipeline, create_pipeline

__all__ = ["RAGEvaluationPipeline", "create_pipeline"]
