"""
Retrieval Quality Metrics
==========================
Measures the effectiveness of the retrieval component.

Metrics:
1. Recall@K - Fraction of relevant documents retrieved in top K
2. Precision@K - Fraction of retrieved documents that are relevant
3. Coverage - Fraction of ground truth context covered by retrieved context
"""

import logging
from typing import Dict, List, Any, Set, Optional
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Computes retrieval quality metrics for RAG evaluation.
    """

    def __init__(self, k: int = 3):
        """
        Initialize retrieval metrics calculator.
        
        Args:
            k: Number of top documents to evaluate (recall@k, precision@k)
        """
        self.k = k
        logger.info(f"Retrieval metrics initialized (k={k})")

    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text into normalized tokens.
        
        Args:
            text: Text to tokenize
        
        Returns:
            Set of tokens
        """
        # Convert to lowercase, remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = set(text.split())
        return tokens

    def _similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score [0, 1]
        """
        matcher = SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()

    def recall_at_k(
        self,
        retrieved_docs: List[str],
        ground_truth_docs: List[str],
    ) -> float:
        """
        Calculate Recall@K: fraction of relevant docs retrieved in top K.
        
        Recall@K = (# relevant docs in top K) / (total # relevant docs)
        
        Args:
            retrieved_docs: List of retrieved document texts (top K)
            ground_truth_docs: List of ground truth (relevant) document texts
        
        Returns:
            Recall@K score [0, 1]
        """
        if not ground_truth_docs:
            logger.warning("No ground truth documents for recall calculation")
            return 1.0  # No ground truth = perfect recall
        
        retrieved_set = len(retrieved_docs)
        if retrieved_set == 0:
            return 0.0
        
        # Count how many ground truth docs are in retrieved set
        matched = 0
        for gt_doc in ground_truth_docs:
            for ret_doc in retrieved_docs:
                # Use similarity threshold
                if self._similarity(ret_doc, gt_doc) > 0.5:
                    matched += 1
                    break
        
        recall = matched / len(ground_truth_docs)
        logger.debug(f"Recall@{self.k}: {recall:.4f} ({matched}/{len(ground_truth_docs)})")
        
        return min(1.0, max(0.0, recall))

    def precision_at_k(
        self,
        retrieved_docs: List[str],
        ground_truth_docs: List[str],
    ) -> float:
        """
        Calculate Precision@K: fraction of retrieved docs that are relevant.
        
        Precision@K = (# relevant docs in top K) / (# retrieved docs)
        
        Args:
            retrieved_docs: List of retrieved document texts (top K)
            ground_truth_docs: List of ground truth document texts
        
        Returns:
            Precision@K score [0, 1]
        """
        if not retrieved_docs:
            logger.warning("No retrieved documents for precision calculation")
            return 1.0  # No retrieved = perfect precision
        
        # Count how many retrieved docs match ground truth
        matched = 0
        for ret_doc in retrieved_docs:
            for gt_doc in ground_truth_docs:
                if self._similarity(ret_doc, gt_doc) > 0.5:
                    matched += 1
                    break
        
        precision = matched / len(retrieved_docs)
        logger.debug(f"Precision@{self.k}: {precision:.4f} ({matched}/{len(retrieved_docs)})")
        
        return min(1.0, max(0.0, precision))

    def coverage(
        self,
        retrieved_context: str,
        ground_truth_context: List[str],
    ) -> float:
        """
        Calculate Coverage: fraction of ground truth context covered by retrieved context.
        
        Uses token-level overlap.
        
        Args:
            retrieved_context: Combined text of retrieved documents
            ground_truth_context: List of ground truth context documents
        
        Returns:
            Coverage score [0, 1]
        """
        if not ground_truth_context:
            logger.warning("No ground truth context for coverage calculation")
            return 1.0  # No GT = perfect coverage
        
        # Tokenize
        retrieved_tokens = self._tokenize(retrieved_context)
        
        # Count ground truth tokens
        all_gt_tokens = set()
        for gt_doc in ground_truth_context:
            all_gt_tokens.update(self._tokenize(gt_doc))
        
        if not all_gt_tokens:
            return 1.0  # No tokens to cover = perfect coverage
        
        # Calculate overlap
        overlap = len(retrieved_tokens & all_gt_tokens)
        coverage = overlap / len(all_gt_tokens)
        
        logger.debug(f"Coverage: {coverage:.4f} ({overlap}/{len(all_gt_tokens)} tokens)")
        
        return min(1.0, max(0.0, coverage))

    def compute_all(
        self,
        retrieved_docs: List[str],
        retrieved_context: str,
        ground_truth_docs: List[str],
    ) -> Dict[str, float]:
        """
        Compute all retrieval metrics.
        
        Args:
            retrieved_docs: List of retrieved document texts
            retrieved_context: Combined retrieved context text
            ground_truth_docs: List of ground truth documents
        
        Returns:
            Dictionary with all metrics:
            {
                "recall_at_k": float,
                "precision_at_k": float,
                "coverage": float,
                "avg_score": float
            }
        """
        recall = self.recall_at_k(retrieved_docs, ground_truth_docs)
        precision = self.precision_at_k(retrieved_docs, ground_truth_docs)
        cov = self.coverage(retrieved_context, ground_truth_docs)
        
        avg_score = (recall + precision + cov) / 3.0
        
        metrics = {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "coverage": cov,
            "avg_score": avg_score
        }
        
        logger.debug(f"Retrieval metrics: {metrics}")
        
        return metrics


# Global instance
_retrieval_metrics: Optional[RetrievalMetrics] = None


def get_retrieval_metrics(k: int = 3) -> RetrievalMetrics:
    """
    Get or create retrieval metrics instance.
    
    Args:
        k: Number of top documents to evaluate
    
    Returns:
        RetrievalMetrics instance
    """
    global _retrieval_metrics
    if _retrieval_metrics is None:
        _retrieval_metrics = RetrievalMetrics(k=k)
    return _retrieval_metrics
