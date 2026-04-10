"""
RERANKER MODULE
===============
Dynamically reranks retrieved documents based on relevance to the query.

Uses:
- CrossEncoder for semantic relevance scoring
- Same model as ML Evaluator for consistency

Function:
- rerank(query: str, docs: List[Document]) -> List[Document]

Returns documents sorted by relevance score (highest first).
"""

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ===========================
# MODULE-LEVEL SINGLETON
# ===========================

_reranker_instance = None
_last_model_name = None


class Reranker:
    """
    Document reranker using CrossEncoder.
    Reorders documents by relevance to query.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize Reranker with CrossEncoder model.
        
        Args:
            model_name: HuggingFace model identifier for CrossEncoder.
                       If not provided, uses config.ML_EVALUATOR_MODEL
                       
        Raises:
            ValueError: If model initialization fails
        """
        # Use provided model or fallback to config
        if model_name is None:
            from config import config
            model_name = config.ML_EVALUATOR_MODEL or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        logger.info(f"Initializing Reranker with model: {model_name}")
        try:
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info(f"Reranker initialized successfully with '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoder for reranking with '{model_name}': {e}")
            raise ValueError(
                f"Failed to load reranker model '{model_name}'. "
                f"Ensure it's a valid HuggingFace CrossEncoder model: {e}"
            )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Input query string
            documents: List of Document objects to rerank
            top_k: Optional limit on number of documents to return
                  If None, returns all documents sorted by score
        
        Returns:
            List of Document objects sorted by relevance (highest first)
            
        Raises:
            ValueError: If documents list is empty or invalid
        """
        if not documents:
            logger.warning("Empty document list provided to reranker")
            return []
        
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            logger.warning("Empty or invalid query provided to reranker")
            return documents
        
        try:
            # Extract document content
            doc_contents = [doc.page_content for doc in documents]
            
            # Create query-document pairs for CrossEncoder
            pairs = [[query, doc_content] for doc_content in doc_contents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Create scored documents list
            scored_docs = [
                {
                    "document": doc,
                    "score": float(score)
                }
                for doc, score in zip(documents, scores)
            ]
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Log reranking results
            top_scores = [f"{d['score']:.3f}" for d in scored_docs[:3]]
            logger.info(
                f"Reranked {len(documents)} documents. "
                f"Top scores: {top_scores}"
            )
            
            # Apply top_k limit if specified
            if top_k:
                scored_docs = scored_docs[:top_k]
                logger.info(f"Limited to top {top_k} documents after reranking")
            
            # Return reranked documents (without scores - just the Document objects)
            reranked = [doc_dict["document"] for doc_dict in scored_docs]
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            logger.warning("Returning original document order due to reranking failure")
            return documents

    def score_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """
        Score documents without reordering.
        
        Useful for analysis or custom sorting logic.
        
        Args:
            query: Input query string
            documents: List of Document objects to score
        
        Returns:
            List of dicts with document and score: [{"document": doc, "score": 0.95}, ...]
        """
        if not documents:
            return []
        
        try:
            doc_contents = [doc.page_content for doc in documents]
            pairs = [[query, doc_content] for doc_content in doc_contents]
            scores = self.model.predict(pairs)
            
            scored = [
                {
                    "document": doc,
                    "score": float(score)
                }
                for doc, score in zip(documents, scores)
            ]
            
            return scored
            
        except Exception as e:
            logger.error(f"Error scoring documents: {e}")
            return []


# ===========================
# SINGLETON GETTER WITH RELOAD SUPPORT
# ===========================

def get_reranker(force_reload: bool = False) -> Reranker:
    """
    Get or create Reranker singleton.
    
    Automatically detects if configuration has changed and reloads.
    Can be forced to reload via force_reload parameter.
    
    Args:
        force_reload: Force reload even if model hasn't changed
    
    Returns:
        Reranker instance
    """
    global _reranker_instance, _last_model_name
    
    from config import config
    current_model = config.ML_EVALUATOR_MODEL or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Check if model configuration has changed
    model_changed = (_last_model_name is not None and _last_model_name != current_model)
    
    if force_reload or _reranker_instance is None or model_changed:
        logger.info("Initializing or reloading Reranker singleton")
        _reranker_instance = Reranker(model_name=current_model)
        _last_model_name = current_model
    
    return _reranker_instance


def reset_reranker():
    """
    Reset Reranker singleton.
    
    Call this when configuration changes and you want to force reload
    on next get_reranker() call.
    """
    global _reranker_instance, _last_model_name
    logger.info("Resetting Reranker singleton")
    _reranker_instance = None
    _last_model_name = None
