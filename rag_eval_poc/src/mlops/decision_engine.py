"""
DECISION ENGINE MODULE
======================
Determines whether to accept/reject/retry answers based on validation metrics.

Rules:
- confidence < MIN_CONFIDENCE_THRESHOLD → Retry
- hallucination > HALLUCINATION_REJECT_THRESHOLD → Reject
- Otherwise → Accept
"""

import logging
from typing import Dict, Any, Optional
from config import config

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Decision engine for RAG answer validation.
    Makes accept/reject/retry decisions based on ML metrics.
    """

    def __init__(
        self,
        min_confidence_threshold: float = None,
        hallucination_reject_threshold: float = None
    ):
        """
        Initialize Decision Engine with thresholds.
        
        Args:
            min_confidence_threshold: Confidence score below which to retry
                                     Defaults to config.MIN_CONFIDENCE_THRESHOLD
            hallucination_reject_threshold: Hallucination score above which to reject
                                           Defaults to config.HALLUCINATION_REJECT_THRESHOLD
        """
        self.min_confidence_threshold = (
            min_confidence_threshold or config.MIN_CONFIDENCE_THRESHOLD
        )
        self.hallucination_reject_threshold = (
            hallucination_reject_threshold or config.HALLUCINATION_REJECT_THRESHOLD
        )
        
        logger.info(
            f"DecisionEngine initialized with "
            f"min_confidence={self.min_confidence_threshold:.2f}, "
            f"hallucination_threshold={self.hallucination_reject_threshold:.2f}"
        )

    def should_retry(self, confidence_score: float) -> bool:
        """
        Determine if answer should trigger a retry.
        
        Args:
            confidence_score: ML confidence score (0.0 to 1.0)
        
        Returns:
            True if confidence is below threshold (should retry)
        """
        should_retry = confidence_score < self.min_confidence_threshold
        
        if should_retry:
            logger.info(
                f"Retry decision: confidence {confidence_score:.3f} "
                f"below threshold {self.min_confidence_threshold:.3f}"
            )
        
        return should_retry

    def should_reject(
        self,
        confidence_score: float,
        hallucination_score: Optional[float] = None
    ) -> bool:
        """
        Determine if answer should be rejected.
        
        Args:
            confidence_score: ML confidence score (0.0 to 1.0)
            hallucination_score: Optional hallucination score (0.0 to 1.0)
                                If provided and above threshold, triggers rejection
        
        Returns:
            True if answer should be rejected
        """
        # Check hallucination first (higher priority)
        if hallucination_score is not None:
            if hallucination_score > self.hallucination_reject_threshold:
                logger.warning(
                    f"Rejection decision: hallucination {hallucination_score:.3f} "
                    f"above threshold {self.hallucination_reject_threshold:.3f}"
                )
                return True
        
        # No rejection criteria met
        return False

    def make_decision(
        self,
        ml_validation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Make accept/reject/retry decision based on ML validation results.
        
        Args:
            ml_validation: Dictionary with ML validation results containing:
                          - confidence_score: float
                          - hallucination_score: float (optional)
        
        Returns:
            Dictionary with decision:
            {
                "action": "accept" | "reject" | "retry",
                "confidence_score": float,
                "hallucination_score": float or None,
                "reason": str,
                "accepted": True/False (for backward compatibility)
            }
        """
        if ml_validation is None:
            logger.debug("No validation results - accepting answer")
            return {
                "action": "accept",
                "confidence_score": 1.0,
                "hallucination_score": None,
                "reason": "No validation performed",
                "accepted": True
            }
        
        confidence = ml_validation.get("confidence_score", 1.0)
        hallucination = ml_validation.get("hallucination_score", None)
        
        # Check rejection first (higher priority than retry)
        if self.should_reject(confidence, hallucination):
            logger.info("Decision: REJECT - hallucination score too high")
            return {
                "action": "reject",
                "confidence_score": confidence,
                "hallucination_score": hallucination,
                "reason": f"Hallucination score {hallucination:.3f} exceeds threshold",
                "accepted": False
            }
        
        # Check retry
        if self.should_retry(confidence):
            logger.info("Decision: RETRY - low confidence score")
            return {
                "action": "retry",
                "confidence_score": confidence,
                "hallucination_score": hallucination,
                "reason": f"Confidence score {confidence:.3f} below threshold",
                "accepted": False
            }
        
        # Otherwise accept
        logger.info("Decision: ACCEPT - all scores above thresholds")
        return {
            "action": "accept",
            "confidence_score": confidence,
            "hallucination_score": hallucination,
            "reason": "All validation scores acceptable",
            "accepted": True
        }


# ===========================
# SINGLETON GETTER
# ===========================

_decision_engine_instance = None


def get_decision_engine(force_reload: bool = False) -> DecisionEngine:
    """
    Get or create DecisionEngine singleton.
    
    Args:
        force_reload: Force reload even if instance exists
    
    Returns:
        DecisionEngine instance
    """
    global _decision_engine_instance
    
    if force_reload or _decision_engine_instance is None:
        logger.info("Initializing DecisionEngine singleton")
        _decision_engine_instance = DecisionEngine()
    
    return _decision_engine_instance


def reset_decision_engine():
    """
    Reset DecisionEngine singleton.
    Call when configuration changes.
    """
    global _decision_engine_instance
    logger.info("Resetting DecisionEngine singleton")
    _decision_engine_instance = None
