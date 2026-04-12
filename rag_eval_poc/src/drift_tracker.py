"""
Drift Tracking & History Comparison
===================================

Tracks evaluation results over time to detect:
- Performance degradation
- Score variance
- Regression detection
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class DriftTracker:
    """
    Track evaluation results over time to detect drift and regressions.
    """

    def __init__(self, history_dir: str = "eval_history"):
        """
        Initialize drift tracker.
        
        Args:
            history_dir: Directory to store evaluation history
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Drift tracker initialized at {self.history_dir}")

    def save_evaluation_run(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        run_id: str = None
    ) -> str:
        """
        Save evaluation run results to history.
        
        Args:
            results: List of test results
            summary: Summary statistics
            run_id: Unique run identifier (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file_path = self.history_dir / f"eval_{run_id}.json"
        
        history_entry = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "results": results
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(history_entry, f, indent=2, default=str)
            logger.info(f"Evaluation run saved: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save evaluation run: {e}")
            raise

    def load_run_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load evaluation run history.
        
        Args:
            limit: Maximum number of runs to load (most recent first)
            
        Returns:
            List of evaluation runs
        """
        runs = []
        
        for file_path in sorted(self.history_dir.glob("eval_*.json"), reverse=True):
            if limit and len(runs) >= limit:
                break
            
            try:
                with open(file_path, 'r') as f:
                    run = json.load(f)
                runs.append(run)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        return runs

    def compute_drift(
        self,
        current_summary: Dict[str, Any],
        baseline_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Compute drift compared to baseline.
        
        Args:
            current_summary: Current evaluation summary
            baseline_runs: Number of recent runs to use as baseline
            
        Returns:
            Drift analysis dictionary
        """
        history = self.load_run_history(limit=baseline_runs + 1)
        
        if len(history) < baseline_runs:
            logger.warning("Not enough history for drift analysis")
            return {"error": "Not enough historical data"}
        
        # Most recent is current, rest are baseline
        baseline_runs_list = history[1:baseline_runs + 1]
        
        drift = {
            "timestamp": datetime.now().isoformat(),
            "baseline_runs": len(baseline_runs_list),
            "metrics": {}
        }

        # Extract scores from current and baseline
        current_score = current_summary.get("pass_fail_decision", {}).get("pass_rate", 0)
        
        if baseline_runs_list:
            baseline_scores = [
                r.get("summary", {}).get("pass_fail_decision", {}).get("pass_rate", 0)
                for r in baseline_runs_list
            ]
            
            mean_baseline = np.mean(baseline_scores)
            std_baseline = np.std(baseline_scores)
            
            # Calculate drift
            if std_baseline > 0:
                z_score = (current_score - mean_baseline) / std_baseline
            else:
                z_score = 0 if current_score == mean_baseline else float('inf')
            
            drift["metrics"]["pass_rate"] = {
                "current": current_score,
                "baseline_mean": round(mean_baseline, 3),
                "baseline_std": round(std_baseline, 3),
                "change": round(current_score - mean_baseline, 3),
                "z_score": round(z_score, 2),
                "status": self._assess_drift(z_score)
            }

        return drift

    def detect_regression(
        self,
        current_summary: Dict[str, Any],
        threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect if current results show regression.
        
        Args:
            current_summary: Current evaluation summary
            threshold_std: Number of std devs for regression detection
            
        Returns:
            Regression detection results
        """
        history = self.load_run_history(limit=10)
        
        if len(history) < 3:
            return {"error": "Not enough history"}
        
        # Get trending scores
        scores = []
        for run in history:
            score = run.get("summary", {}).get("pass_fail_decision", {}).get("pass_rate", 0)
            scores.append(score)
        
        # Fit trend line (simple approach)
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array)
        
        current_score = scores[0]  # Most recent
        
        # Calculate z-score for regression detection
        if std > 0:
            z_score = (current_score - mean) / std
        else:
            z_score = 0
        
        is_regression = z_score < -threshold_std
        
        return {
            "is_regression": is_regression,
            "current_score": current_score,
            "historical_mean": round(mean, 3),
            "historical_std": round(std, 3),
            "z_score": round(z_score, 2),
            "threshold_std": threshold_std,
            "severity": "CRITICAL" if is_regression else "OK"
        }

    def compare_runs(
        self,
        run_id_1: str,
        run_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two evaluation runs.
        
        Args:
            run_id_1: First run ID
            run_id_2: Second run ID
            
        Returns:
            Comparison results
        """
        file1 = self.history_dir / f"eval_{run_id_1}.json"
        file2 = self.history_dir / f"eval_{run_id_2}.json"
        
        if not file1.exists() or not file2.exists():
            return {"error": "One or both run files not found"}
        
        try:
            with open(file1, 'r') as f:
                run1 = json.load(f)
            with open(file2, 'r') as f:
                run2 = json.load(f)
            
            return {
                "run_id_1": run_id_1,
                "run_id_2": run_id_2,
                "comparison": self._compare_summaries(
                    run1.get("summary", {}),
                    run2.get("summary", {})
                )
            }
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {"error": str(e)}

    def _compare_summaries(
        self,
        summary1: Dict[str, Any],
        summary2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two evaluation summaries."""
        pass_fail_1 = summary1.get("pass_fail_decision", {})
        pass_fail_2 = summary2.get("pass_fail_decision", {})
        
        return {
            "passed": {
                "run1": pass_fail_1.get("passed", 0),
                "run2": pass_fail_2.get("passed", 0),
                "delta": pass_fail_2.get("passed", 0) - pass_fail_1.get("passed", 0)
            },
            "pass_rate": {
                "run1": pass_fail_1.get("pass_rate", 0),
                "run2": pass_fail_2.get("pass_rate", 0),
                "delta": pass_fail_2.get("pass_rate", 0) - pass_fail_1.get("pass_rate", 0)
            }
        }

    def _assess_drift(self, z_score: float) -> str:
        """Assess drift severity based on z-score."""
        if abs(z_score) < 1.0:
            return "STABLE"
        elif abs(z_score) < 2.0:
            return "MINOR_DRIFT"
        elif abs(z_score) < 3.0:
            return "MODERATE_DRIFT"
        else:
            return "CRITICAL_DRIFT"

    def get_trend_analysis(self, limit: int = 10) -> Dict[str, Any]:
        """
        Analyze evaluation trends.
        
        Args:
            limit: Number of recent runs to analyze
            
        Returns:
            Trend analysis
        """
        history = self.load_run_history(limit=limit)
        
        if not history:
            return {"error": "No history available"}
        
        scores = []
        timestamps = []
        
        for run in reversed(history):  # Chronological order
            score = run.get("summary", {}).get("pass_fail_decision", {}).get("pass_rate", 0)
            timestamp = run.get("timestamp")
            scores.append(score)
            timestamps.append(timestamp)
        
        if not scores:
            return {"error": "No scores to analyze"}
        
        # Trend analysis
        trend = {
            "num_runs": len(scores),
            "scores": scores,
            "mean": round(np.mean(scores), 3),
            "std": round(np.std(scores), 3),
            "min": round(np.min(scores), 3),
            "max": round(np.max(scores), 3),
            "direction": self._detect_trend_direction(scores)
        }
        
        return trend

    def _detect_trend_direction(self, scores: List[float]) -> str:
        """Detect if trend is improving, degrading, or stable."""
        if len(scores) < 2:
            return "UNKNOWN"
        
        # Simple linear regression
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Fit line
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        if slope > 0.02:
            return "IMPROVING"
        elif slope < -0.02:
            return "DEGRADING"
        else:
            return "STABLE"


# Global drift tracker instance
_drift_tracker = None


def get_drift_tracker(history_dir: str = "eval_history") -> DriftTracker:
    """
    Get or create global drift tracker instance.
    
    Args:
        history_dir: History directory
        
    Returns:
        DriftTracker instance
    """
    global _drift_tracker
    if _drift_tracker is None:
        _drift_tracker = DriftTracker(history_dir)
    return _drift_tracker
