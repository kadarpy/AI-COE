"""
Retrieval Quality Evaluator
===========================

Computes retrieval metrics:
- Precision@k: What fraction of retrieved docs are relevant?
- Recall@k: What fraction of relevant docs were retrieved?
- Hit Rate@k: Was there at least one relevant doc?

Uses semantic similarity (embeddings) for relevance judgment.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticRelevanceJudge:
    """Judge relevance using semantic embeddings."""

    def __init__(self):
        """Initialize semantic model for relevance judgment."""
        try:
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            self.util = util
            logger.info("SemanticRelevanceJudge initialized with BAAI/bge-small-en-v1.5")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticRelevanceJudge: {e}")
            self.model = None
            self.util = None

    def compute_relevance(self, text1: str, text2: str) -> float:
        """
        Compute semantic relevance between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Relevance score [0, 1]
        """
        if not self.model:
            logger.warning("Model not initialized, returning 0.0")
            return 0.0

        try:
            # Clean texts
            t1 = str(text1).strip().lower()
            t2 = str(text2).strip().lower()
            
            if not t1 or not t2:
                return 0.0

            # Compute embeddings
            emb1 = self.model.encode([t1], convert_to_tensor=True)
            emb2 = self.model.encode([t2], convert_to_tensor=True)

            # Cosine similarity
            similarity = self.util.pytorch_cos_sim(emb1, emb2)[0][0].item()
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.error(f"Failed to compute relevance: {e}")
            return 0.0

    def judge_relevance(self, retrieved_text: str, ground_truth: str, threshold: float = 0.7) -> bool:
        """
        Judge if a retrieved text is relevant to ground truth.
        
        Uses semantic similarity with threshold.
        
        Args:
            retrieved_text: Text retrieved by system
            ground_truth: Ground truth reference text
            threshold: Similarity threshold for relevance [0, 1]
            
        Returns:
            True if relevant, False otherwise
        """
        similarity = self.compute_relevance(retrieved_text, ground_truth)
        return similarity >= threshold


class RetrievalEvaluator:
    """
    Evaluate retrieval quality using semantic similarity.
    """

    def __init__(self, relevance_threshold: float = 0.7):
        """
        Initialize retrieval evaluator.
        
        Args:
            relevance_threshold: Threshold for judging relevance [0, 1]
        """
        self.judge = SemanticRelevanceJudge()
        self.relevance_threshold = relevance_threshold
        logger.info(f"RetrievalEvaluator initialized (threshold={relevance_threshold})")

    def evaluate_retrieval(
        self,
        retrieved_chunks: List[str],
        ground_truth_context: List[str],
        k: int = None
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance against ground truth.
        
        Args:
            retrieved_chunks: List of chunks retrieved by system
            ground_truth_context: List of ground truth context chunks
            k: Evaluate at k (if None, use all retrieved)
            
        Returns:
            Dictionary with retrieval metrics
        """
        if not ground_truth_context:
            logger.warning("No ground truth context provided")
            return {
                "precision_at_k": None,
                "recall_at_k": None,
                "hit_rate_at_k": None,
                "mean_similarity": None,
                "error": "No ground truth context"
            }

        # Limit to k if specified
        if k is not None and len(retrieved_chunks) > k:
            retrieved_chunks = retrieved_chunks[:k]
        
        actual_k = len(retrieved_chunks)
        if actual_k == 0:
            logger.warning("No chunks retrieved")
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "hit_rate_at_k": 0.0,
                "mean_similarity": 0.0,
                "k": actual_k,
                "num_ground_truth": len(ground_truth_context)
            }

        # Compute relevance for each retrieved chunk vs each ground truth
        relevance_matrix = np.zeros((len(retrieved_chunks), len(ground_truth_context)))
        
        for i, retrieved in enumerate(retrieved_chunks):
            for j, truth in enumerate(ground_truth_context):
                relevance_matrix[i, j] = self.judge.compute_relevance(retrieved, truth)

        # Precision@k: fraction of retrieved that are relevant
        # A retrieved chunk is relevant if it has max similarity >= threshold with any ground truth
        retrieved_relevance = np.max(relevance_matrix, axis=1)  # best match for each retrieved
        num_relevant_retrieved = np.sum(retrieved_relevance >= self.relevance_threshold)
        precision = num_relevant_retrieved / actual_k if actual_k > 0 else 0.0

        # Recall@k: fraction of ground truth that was retrieved
        # Ground truth is found if it has max similarity >= threshold with any retrieved
        truth_relevance = np.max(relevance_matrix, axis=0)  # best match for each ground truth
        num_relevant_found = np.sum(truth_relevance >= self.relevance_threshold)
        recall = num_relevant_found / len(ground_truth_context) if len(ground_truth_context) > 0 else 0.0

        # Hit Rate@k: at least one relevant chunk retrieved?
        hit_rate = 1.0 if np.max(retrieved_relevance) >= self.relevance_threshold else 0.0

        # Mean similarity (diagnostic)
        mean_similarity = float(np.mean(relevance_matrix))

        result = {
            "precision_at_k": round(precision, 3),
            "recall_at_k": round(recall, 3),
            "hit_rate_at_k": round(hit_rate, 3),
            "mean_similarity": round(mean_similarity, 3),
            "k": actual_k,
            "num_ground_truth": len(ground_truth_context),
            "num_relevant_retrieved": int(num_relevant_retrieved),
            "num_relevant_found": int(num_relevant_found),
            "relevance_threshold": self.relevance_threshold
        }

        logger.info(
            f"Retrieval metrics: "
            f"precision={result['precision_at_k']}, "
            f"recall={result['recall_at_k']}, "
            f"hit_rate={result['hit_rate_at_k']}"
        )

        return result

    def compute_stability_metrics(
        self,
        runs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute stability metrics across multiple runs.
        
        Args:
            runs: List of retrieval evaluation results from multiple runs
            
        Returns:
            Stability statistics (mean, std, min, max)
        """
        if not runs:
            return {}

        metrics_to_track = ["precision_at_k", "recall_at_k", "hit_rate_at_k", "mean_similarity"]
        stability = {}

        for metric in metrics_to_track:
            values = [r.get(metric) for r in runs if r.get(metric) is not None]
            
            if values:
                stability[metric] = {
                    "mean": round(np.mean(values), 3),
                    "std": round(np.std(values), 3),
                    "min": round(np.min(values), 3),
                    "max": round(np.max(values), 3),
                    "variance": round(np.var(values), 3)
                }

        logger.info(f"Stability metrics computed across {len(runs)} runs")
        return stability


# Global evaluator instance
_evaluator = None


def get_retrieval_evaluator(threshold: float = 0.7) -> RetrievalEvaluator:
    """
    Get or create global retrieval evaluator.
    
    Args:
        threshold: Relevance threshold
        
    Returns:
        RetrievalEvaluator instance
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = RetrievalEvaluator(threshold)
    return _evaluator
