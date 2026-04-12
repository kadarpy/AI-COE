"""
Cross-Encoder Reranker for RAG
==============================

Uses lightweight cross-encoder to rerank retrieved documents.

Flow:
1. Retrieve initial documents (k=20)
2. Rerank using cross-encoder
3. Return top_k best documents

Cross-encoder is more accurate than bi-encoder retrieval.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Rerank retrieved documents using cross-encoder.
    
    Model: sentence-transformers/ms-marco-MiniLM-L-12-v2
    Task: Estimate relevance score(query, document)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model to use for reranking
        """
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self.model is not None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float, str]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: The query/question
            documents: List of documents to rerank
            top_k: Return top k documents (None = return all)
            
        Returns:
            List of tuples: (original_index, score, document)
            Sorted by score (descending)
        """
        if not self.model:
            logger.warning("Cross-encoder not available, returning docs unchanged")
            return [(i, 0.0, doc) for i, doc in enumerate(documents)]

        if not documents:
            return []

        try:
            # Compute cross-encoder scores
            logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")
            
            # Create (query, document) pairs
            pairs = [[query, doc] for doc in documents]
            
            # Score all pairs
            scores = self.model.predict(pairs)
            
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            # Sort by score (descending)
            ranked = sorted(
                enumerate(zip(scores, documents)),
                key=lambda x: x[1][0],
                reverse=True
            )
            
            # Format output: (original_index, score, document)
            result = [(idx, float(score), doc) for idx, (score, doc) in ranked]
            
            # Limit to top_k if specified
            if top_k is not None:
                result = result[:top_k]
            
            logger.debug(f"Reranking complete: top_1 score={result[0][1]:.3f}" if result else "No results")
            
            return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [(i, 0.0, doc) for i, doc in enumerate(documents)]

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[str],
        threshold: float = 0.5,
        top_k: int = None
    ) -> List[Tuple[int, float, str]]:
        """
        Rerank documents and filter by threshold.
        
        Args:
            query: The query/question
            documents: List of documents to rerank
            threshold: Minimum score to include [0, 1]
            top_k: Return top k documents (after threshold)
            
        Returns:
            List of tuples: (original_index, score, document)
        """
        ranked = self.rerank(query, documents)
        
        # Filter by threshold
        filtered = [r for r in ranked if r[1] >= threshold]
        
        if not filtered:
            logger.warning(f"No documents meet threshold {threshold}")
            # Return top 1 if nothing passes threshold
            return ranked[:1] if ranked else []
        
        # Limit to top_k
        if top_k is not None:
            filtered = filtered[:top_k]
        
        return filtered


class RerankerFactory:
    """Factory for creating reranker instances."""

    _instance = None
    _enabled = True
    _model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    @classmethod
    def get_reranker(cls) -> CrossEncoderReranker:
        """Get or create global reranker instance."""
        if not cls._enabled:
            logger.debug("Reranker disabled")
            return None
        
        if cls._instance is None:
            cls._instance = CrossEncoderReranker(model_name=cls._model_name)
        
        return cls._instance

    @classmethod
    def disable_reranker(cls):
        """Disable reranker."""
        cls._enabled = False
        logger.info("Reranker disabled")

    @classmethod
    def enable_reranker(cls):
        """Enable reranker."""
        cls._enabled = True
        logger.info("Reranker enabled")

    @classmethod
    def set_model(cls, model_name: str):
        """Set model to use for reranking."""
        cls._model_name = model_name
        cls._instance = None  # Reset to reinitialize
        logger.info(f"Reranker model set to {model_name}")


def get_reranker() -> CrossEncoderReranker:
    """Get global reranker instance."""
    return RerankerFactory.get_reranker()


def apply_reranking(
    query: str,
    retrieved_docs: List[Any],
    top_k: int = 5,
    extract_content=None
) -> List[Any]:
    """
    Apply reranking to retrieved documents.
    
    Args:
        query: The query
        retrieved_docs: List of document objects
        top_k: Return top k documents
        extract_content: Function to extract content from doc objects (default: .page_content)
        
    Returns:
        Reranked document list
    """
    reranker = get_reranker()
    
    if not reranker or not reranker.is_available():
        logger.debug("Reranker not available, returning original results")
        return retrieved_docs[:top_k] if top_k else retrieved_docs
    
    # Extract content from documents
    if extract_content is None:
        def extract_content(doc):
            return doc.page_content if hasattr(doc, 'page_content') else str(doc)
    
    texts = [extract_content(doc) for doc in retrieved_docs]
    
    # Rerank
    ranked = reranker.rerank(query, texts, top_k=top_k)
    
    # Match back to original documents
    result = []
    for orig_idx, score, text in ranked:
        result.append({
            "document": retrieved_docs[orig_idx],
            "rerank_score": score,
            "original_index": orig_idx
        })
    
    logger.info(f"Reranked {len(retrieved_docs)} docs → top_{len(result)}")
    return result
