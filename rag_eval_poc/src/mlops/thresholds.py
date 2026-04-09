"""
Auto-Threshold Scoring Engine
==============================
Provides automated pass/fail evaluation based on configurable thresholds.

Default thresholds:
- Relevance: >= 0.75 (answer must be relevant)
- Faithfulness: >= 0.80 (answer must stick to context)
- Hallucination: <= 0.20 (answer should not hallucinate)
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for threshold evaluation."""
    relevance: float = 0.75
    faithfulness: float = 0.80
    hallucination: float = 0.20  # Higher means more hallucination (bad)
    require_all: bool = True  # If True, all thresholds must pass


class ThresholdEngine:
    """
    Evaluates metrics against thresholds for automated pass/fail scoring.
    """

    DEFAULT_THRESHOLDS = {
        "relevance": 0.75,
        "faithfulness": 0.80,
        "hallucination": 0.20
    }

    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize threshold engine.
        
        Args:
            config: Optional ThresholdConfig. Uses defaults if not provided.
        """
        if config is None:
            config = ThresholdConfig()
        
        self.config = config
        
        logger.info("Threshold Engine initialized")
        logger.info(f"  Relevance threshold: {config.relevance}")
        logger.info(f"  Faithfulness threshold: {config.faithfulness}")
        logger.info(f"  Hallucination threshold: {config.hallucination}")

    def evaluate_pass_fail(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate metrics against thresholds for automated pass/fail.
        
        Args:
            metrics: Dictionary of metric values, expected to contain:
                - relevance: float [0,1]
                - faithfulness: float [0,1]
                - hallucination: float [0,1] (where 1 = high hallucination)
        
        Returns:
            Dictionary with pass/fail evaluation:
            {
                "pass": bool,
                "reason": str,
                "details": {
                    "relevance": { "value": float, "pass": bool, "reason": str },
                    "faithfulness": { "value": float, "pass": bool, "reason": str },
                    "hallucination": { "value": float, "pass": bool, "reason": str }
                }
            }
        """
        details = {}
        
        # Check relevance
        relevance = metrics.get('relevance')
        if relevance is not None:
            relevance_pass = relevance >= self.config.relevance
            details['relevance'] = {
                'value': relevance,
                'threshold': self.config.relevance,
                'pass': relevance_pass,
                'reason': f"Relevance {relevance:.3f} {'≥' if relevance_pass else '<'} {self.config.relevance}"
            }
        else:
            details['relevance'] = {
                'value': None,
                'threshold': self.config.relevance,
                'pass': False,
                'reason': "Relevance score not available"
            }
        
        # Check faithfulness
        faithfulness = metrics.get('faithfulness')
        if faithfulness is not None:
            faithfulness_pass = faithfulness >= self.config.faithfulness
            details['faithfulness'] = {
                'value': faithfulness,
                'threshold': self.config.faithfulness,
                'pass': faithfulness_pass,
                'reason': f"Faithfulness {faithfulness:.3f} {'≥' if faithfulness_pass else '<'} {self.config.faithfulness}"
            }
        else:
            details['faithfulness'] = {
                'value': None,
                'threshold': self.config.faithfulness,
                'pass': False,
                'reason': "Faithfulness score not available"
            }
        
        # Check hallucination (lower is better)
        hallucination = metrics.get('hallucination')
        if hallucination is not None:
            hallucination_pass = hallucination <= self.config.hallucination
            details['hallucination'] = {
                'value': hallucination,
                'threshold': self.config.hallucination,
                'pass': hallucination_pass,
                'reason': f"Hallucination {hallucination:.3f} {'≤' if hallucination_pass else '>'} {self.config.hallucination}"
            }
        else:
            details['hallucination'] = {
                'value': None,
                'threshold': self.config.hallucination,
                'pass': False,
                'reason': "Hallucination score not available"
            }
        
        # Determine overall pass/fail
        if self.config.require_all:
            overall_pass = all(d['pass'] for d in details.values() if d['value'] is not None)
        else:
            overall_pass = any(d['pass'] for d in details.values())
        
        # Generate reason
        failed_checks = [k for k, v in details.items() if not v['pass']]
        if overall_pass:
            reason = "All thresholds passed ✓"
        else:
            reason = f"Failed checks: {', '.join(failed_checks)}"
        
        return {
            "pass": overall_pass,
            "reason": reason,
            "details": details
        }

    def batch_evaluate(self, metrics_list: list) -> list:
        """
        Evaluate multiple metric sets.
        
        Args:
            metrics_list: List of metric dictionaries
        
        Returns:
            List of pass/fail evaluations
        """
        results = []
        for i, metrics in enumerate(metrics_list):
            try:
                result = self.evaluate_pass_fail(metrics)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating metrics {i}: {e}")
                results.append({
                    "pass": False,
                    "reason": f"Evaluation error: {str(e)}",
                    "details": {}
                })
        
        return results

    def update_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Update threshold values.
        
        Args:
            thresholds: Dictionary with threshold updates
        """
        for key, value in thresholds.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)
                setattr(self.config, key, value)
                logger.info(f"Updated {key}: {old_value} → {value}")
            else:
                logger.warning(f"Unknown threshold: {key}")


# Global threshold engine instance
_threshold_engine: Optional[ThresholdEngine] = None


def get_threshold_engine(config: Optional[ThresholdConfig] = None) -> ThresholdEngine:
    """
    Get or create threshold engine instance.
    
    Args:
        config: Optional configuration
    
    Returns:
        ThresholdEngine instance
    """
    global _threshold_engine
    if _threshold_engine is None:
        _threshold_engine = ThresholdEngine(config)
    return _threshold_engine
