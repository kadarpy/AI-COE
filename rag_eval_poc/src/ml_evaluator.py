"""
ML-BASED EVALUATION LAYER
==========================
Provides machine learning-based evaluation using Sentence Transformers CrossEncoder.

This module complements DeepEval by providing:
1. Semantic Relevance: How relevant is the answer to the question (using CrossEncoder)
2. Context Overlap: How much does the answer overlap with retrieved context
3. Confidence Score: Combined score for overall reliability

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Fast, lightweight, deterministic
- No API calls required (runs locally)
- Good for relevance scoring
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from sentence_transformers import CrossEncoder
import re

logger = logging.getLogger(__name__)


class MLEvaluator:
    """
    Machine Learning-based evaluator using CrossEncoder.
    Provides deterministic ML scoring to complement DeepEval's LLM-based scoring.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize ML evaluator with CrossEncoder model.
        
        Args:
            model_name: HuggingFace model identifier for CrossEncoder
        """
        logger.info(f"Initializing MLEvaluator with model: {model_name}")
        try:
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info("ML Evaluator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoder: {e}")
            raise

    def _tokenize(self, text: str) -> set:
        """
        Simple tokenization for overlap calculation.
        Converts text to lowercase and removes punctuation.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Set of tokens (words)
        """
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Split into tokens
        tokens = set(text.split())
        return tokens

    def _calculate_semantic_relevance(self, question: str, answer: str) -> float:
        """
        Calculate semantic relevance between question and answer using CrossEncoder.
        
        Args:
            question: Input question
            answer: Generated answer
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # CrossEncoder expects list of [query, document] pairs
            pairs = [[question, answer]]
            
            # Get scores from model (returns logits that need sigmoid for probabilities)
            scores = self.model.predict(pairs, convert_to_numpy=True)
            
            # Convert logits to probability (sigmoid function)
            sigmoid_scores = 1 / (1 + np.exp(-scores))
            relevance_score = float(sigmoid_scores[0])
            
            logger.debug(f"Semantic Relevance: {relevance_score:.4f}")
            return min(max(relevance_score, 0.0), 1.0)  # Ensure in [0, 1]
        
        except Exception as e:
            logger.error(f"Failed to calculate semantic relevance: {e}")
            return 0.5  # Default neutral score on error

    def _calculate_context_overlap(self, answer: str, context: List[str]) -> float:
        """
        Calculate how much the answer overlaps with retrieved context.
        Uses hybrid approach: token overlap (Jaccard) + embedding similarity.
        
        Args:
            answer: Generated answer
            context: List of context strings from retrieval
            
        Returns:
            Overlap score between 0 and 1
        """
        try:
            if not context:
                logger.warning("Empty context for overlap calculation")
                return 0.5  # Neutral if no context
            
            # Calculate Jaccard token overlap
            answer_tokens = self._tokenize(answer)
            
            if not answer_tokens:
                logger.warning("Answer produced no tokens")
                return 0.0
            
            # Collect all context tokens
            context_tokens = set()
            for ctx in context:
                context_tokens.update(self._tokenize(ctx))
            
            if not context_tokens:
                logger.warning("Context produced no tokens")
                return 0.0
            
            # Calculate Jaccard similarity: intersection / union
            intersection = len(answer_tokens & context_tokens)
            union = len(answer_tokens | context_tokens)
            
            if union == 0:
                jaccard_score = 0.0
            else:
                jaccard_score = intersection / union
            
            # Calculate embedding-based similarity for better semantic alignment
            try:
                # Concatenate context for embedding
                context_text = "\n".join(context)
                
                # Use CrossEncoder to score answer-context alignment
                pairs = [[context_text, answer]]
                scores = self.model.predict(pairs, convert_to_numpy=True)
                
                # Convert logits to probability (sigmoid)
                embedding_similarity = float(1 / (1 + np.exp(-scores[0])))
            except Exception as e:
                logger.debug(f"Embedding similarity calculation failed: {e}, using Jaccard only")
                embedding_similarity = jaccard_score
            
            # Hybrid: combine token overlap (50%) with embedding similarity (50%)
            # This provides both lexical and semantic overlap detection
            overlap_score = (0.5 * jaccard_score) + (0.5 * embedding_similarity)
            
            logger.debug(f"Context Overlap - Jaccard: {jaccard_score:.4f}, Embedding: {embedding_similarity:.4f}, Hybrid: {overlap_score:.4f}")
            return min(max(overlap_score, 0.0), 1.0)  # Ensure in [0, 1]
        
        except Exception as e:
            logger.error(f"Failed to calculate context overlap: {e}")
            return 0.5  # Default neutral score on error

    def evaluate(
        self,
        question: str,
        answer: str,
        context: List[str],
        context_length: Optional[int] = None,
        num_docs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive ML-based evaluation with optional metadata.
        
        Args:
            question: Input question
            answer: Generated answer
            context: List of retrieved context strings
            context_length: Optional context length (characters) for confidence calibration
            num_docs: Optional number of documents retrieved for confidence calibration
            
        Returns:
            Dictionary with:
            {
                "semantic_relevance": float (0-1),
                "context_overlap": float (0-1),
                "confidence_score": float (0-1),
                "metrics_source": "ml_evaluator"
            }
        """
        try:
            # Calculate individual scores
            semantic_relevance = self._calculate_semantic_relevance(question, answer)
            context_overlap = self._calculate_context_overlap(answer, context)
            
            # Base confidence: 70% relevance, 30% overlap
            # This weighting prioritizes answer quality over context alignment
            confidence_score = (0.7 * semantic_relevance) + (0.3 * context_overlap)
            
            # Optional: Calibrate confidence based on retrieval metadata
            if context_length is not None and num_docs is not None:
                # Boost confidence if we have substantial context
                # Penalize if we have very little context
                min_context_threshold = 100  # Minimum useful context chars
                max_context_threshold = 5000  # Maximum useful context chars
                
                context_adequacy = 0.5  # Default neutral
                if context_length < min_context_threshold:
                    # Insufficient context reduces confidence
                    context_adequacy = 0.3
                elif context_length > max_context_threshold:
                    # Too much context may indicate poor retrieval
                    context_adequacy = 0.7
                else:
                    # Good amount of context
                    context_adequacy = 0.9
                
                # Lightly adjust confidence based on context adequacy
                # 95% base score + 5% metadata adjustment to avoid over-weighting
                confidence_score = (0.95 * confidence_score) + (0.05 * context_adequacy)
            
            # Ensure confidence is in valid range
            confidence_score = min(max(confidence_score, 0.0), 1.0)
            
            result = {
                "semantic_relevance": round(float(semantic_relevance), 4),
                "context_overlap": round(float(context_overlap), 4),
                "confidence_score": round(float(confidence_score), 4),
                "metrics_source": "ml_evaluator"
            }
            
            logger.info(
                f"ML Evaluation complete - Relevance: {semantic_relevance:.4f}, "
                f"Overlap: {context_overlap:.4f}, Confidence: {confidence_score:.4f}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Comprehensive ML evaluation failed: {e}")
            # Return neutral scores on critical error
            return {
                "semantic_relevance": 0.5,
                "context_overlap": 0.5,
                "confidence_score": 0.5,
                "metrics_source": "ml_evaluator",
                "error": str(e)
            }


# Singleton instance for lazy loading (optional optimization)
_ml_evaluator_instance: Optional[MLEvaluator] = None


def get_ml_evaluator() -> MLEvaluator:
    """
    Get or create singleton instance of MLEvaluator.
    This ensures the model is loaded only once.
    
    Returns:
        MLEvaluator instance
    """
    global _ml_evaluator_instance
    if _ml_evaluator_instance is None:
        _ml_evaluator_instance = MLEvaluator()
    return _ml_evaluator_instance
