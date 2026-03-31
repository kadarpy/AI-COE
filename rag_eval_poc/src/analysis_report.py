"""
Analysis Report Generator - Evaluate vs Manual Testing Comparison
Demonstrates cases where metrics catch issues manual testing would miss
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalysisReport:
    """Generate comparison between manual judgment and metric-based evaluation"""

    def __init__(self):
        self.comparison_rows = []
        self.findings = {
            "metrics_caught_correctly": [],
            "metrics_missed": [],
            "false_positives": [],
            "manual_would_miss": []
        }

    def analyze_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single evaluation result
        
        Returns:
        {
            question: str
            expected_answer: str
            actual_answer: str
            manual_judgment: str
            metric_scores: dict
            verdict: str
            analysis: str
        }
        """
        
        question = result.get("question", "")
        expected_answer = result.get("expected_answer", "")
        actual_answer = result.get("actual_answer", "")
        metrics = result.get("metrics", {})
        failure_analysis = result.get("failure_analysis", {})

        # Manual judgment: would human tester notice the issue?
        manual_judgment = self._get_manual_judgment(
            actual_answer,
            expected_answer,
            failure_analysis
        )

        # Get verdict from metrics
        metric_scores = {m: metrics[m].get("score") for m in metrics}
        overall_passed = result.get("overall_passed", False)

        # Determine if metrics and manual agree or disagree
        verdict = self._compare_judgments(
            manual_judgment,
            overall_passed,
            failure_analysis
        )

        analysis = self._generate_analysis(
            manual_judgment,
            overall_passed,
            failure_analysis,
            metric_scores
        )

        row = {
            "test_id": result.get("test_id"),
            "category": result.get("category"),
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "manual_judgment": manual_judgment,
            "metric_verdict": "PASS" if overall_passed else "FAIL",
            "metric_scores": metric_scores,
            "failure_type": failure_analysis.get("failure_type", "unknown"),
            "verdict": verdict,
            "analysis": analysis
        }

        self.comparison_rows.append(row)
        return row

    def _get_manual_judgment(self, actual_answer: str, expected_answer: str, failure_analysis: Dict) -> str:
        """
        Simulate human judgment:
        - "Looks correct" if answer appears reasonable
        - "Looks incorrect" if answer is obviously wrong
        """
        
        # Check if answer looks reasonable (passes basic sanity check)
        if not actual_answer or len(actual_answer.strip()) < 5:
            return "Looks incorrect"

        # Check if it appears to match expected answer
        actual_lower = actual_answer.lower()
        expected_lower = expected_answer.lower()

        # If keywords overlap, human might think it's correct
        actual_words = set(actual_lower.split())
        expected_words = set(expected_lower.split())

        if not expected_words:
            return "Looks reasonable"

        overlap_ratio = len(actual_words & expected_words) / len(expected_words)

        if overlap_ratio > 0.6:
            return "Looks correct"
        elif overlap_ratio > 0.3:
            return "Looks partially correct"
        else:
            return "Looks incorrect"

    def _compare_judgments(self, manual: str, metrics_pass: bool, failure_analysis: Dict) -> str:
        """
        Compare manual judgment vs metrics
        
        Returns:
        - "AGREES": Both manual and metrics agree
        - "METRICS_WINS": Metrics correctly identified issue human missed
        - "METRICS_WRONG": Metrics failed but answer actually wrong
        - "HUMAN_MISLED": Answer looks good but metrics caught real issue
        """
        
        looks_correct = "correct" in manual.lower()
        failure_type = failure_analysis.get("failure_type", "correct")

        if failure_type == "correct":
            if looks_correct and metrics_pass:
                return "AGREES"
            elif looks_correct and not metrics_pass:
                return "METRICS_WRONG"
            elif not looks_correct and metrics_pass:
                return "METRICS_WRONG"
            else:
                return "AGREES"
        else:
            # Failure detected by analysis
            if looks_correct and not metrics_pass:
                return "METRICS_WINS"  # Metrics caught what human missed!
            elif looks_correct and metrics_pass:
                return "HUMAN_MISLED"  # Human fooled but metrics passed (false positive)
            else:
                return "AGREES"  # Both detected the issue

    def _generate_analysis(self, manual: str, metrics_pass: bool, failure_analysis: Dict, metric_scores: Dict) -> str:
        """Generate detailed analysis of the evaluation"""
        
        failure_type = failure_analysis.get("failure_type", "correct")
        reason = failure_analysis.get("reason", "")
        metric_alignment = failure_analysis.get("metric_alignment", "")

        if failure_type == "correct":
            return f"Answer appears acceptable. {reason}"

        elif failure_type == "hallucination":
            return f"HALLUCINATION DETECTED: {reason}. Model likely fabricated or inferred information beyond source documents."

        elif failure_type == "retrieval_miss":
            return f"RETRIEVAL FAILURE: {reason}. Insufficient or irrelevant context retrieved."

        elif failure_type == "partial_answer":
            return f"PARTIAL ANSWER: {reason}. Model provided incomplete or tangentially related response."

        else:
            return f"Analysis: {reason}"

    def generate_comparison_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate full comparison report from evaluation results
        
        Returns comprehensive analysis showing:
        1. Which metrics caught what issues
        2. Which issues metrics missed
        3. False positives
        4. Cases where metrics won against human judgment
        """
        
        self.comparison_rows = []
        
        for result in results:
            self.analyze_result(result)

        # Categorize findings
        for row in self.comparison_rows:
            verdict = row["verdict"]
            failure_type = row["failure_type"]
            manual = row["manual_judgment"]

            if verdict == "METRICS_WINS":
                self.findings["manual_would_miss"].append({
                    "test_id": row["test_id"],
                    "issue": failure_type,
                    "question": row["question"],
                    "analys": row["analysis"]
                })

            elif verdict == "HUMAN_MISLED":
                self.findings["false_positives"].append({
                    "test_id": row["test_id"],
                    "question": row["question"],
                    "reason": "Answer looks correct but metrics flagged"
                })

            elif failure_type != "correct" and row["metric_verdict"] == "PASS":
                self.findings["metrics_missed"].append({
                    "test_id": row["test_id"],
                    "issue": failure_type,
                    "question": row["question"]
                })

            elif failure_type != "correct" and row["metric_verdict"] == "FAIL":
                self.findings["metrics_caught_correctly"].append({
                    "test_id": row["test_id"],
                    "issue": failure_type,
                    "question": row["question"]
                })

        return {
            "total_comparisons": len(self.comparison_rows),
            "comparison_rows": self.comparison_rows,
            "findings": self.findings,
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        total = len(self.comparison_rows)
        if total == 0:
            return {}

        metrics_advantages = len(self.findings["manual_would_miss"])
        metrics_failures = len(self.findings["metrics_missed"])
        false_positives = len(self.findings["false_positives"])

        return {
            "total_cases_analyzed": total,
            "cases_metrics_caught_human_missed": metrics_advantages,
            "cases_metrics_failed_to_catch": metrics_failures,
            "false_positives": false_positives,
            "metrics_precision": (
                metrics_advantages / total * 100 if total > 0 else 0
            ),
            "key_insight": (
                f"Structured evaluation caught {metrics_advantages} issues "
                f"that manual testing would likely miss ({metrics_advantages/total*100:.1f}% "
                f"of cases). With {metrics_failures} missed issues and {false_positives} false positives."
            )
        }

    def export_comparison_csv(self, output_path: Optional[Path] = None) -> str:
        """Export comparison results as CSV"""
        
        import csv

        if not output_path:
            output_path = Path("comparison_analysis.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "test_id",
                    "category",
                    "question",
                    "manual_judgment",
                    "metric_verdict",
                    "failure_type",
                    "verdict",
                    "analysis"
                ]
            )
            writer.writeheader()
            for row in self.comparison_rows:
                writer.writerow({
                    "test_id": row["test_id"],
                    "category": row["category"],
                    "question": row["question"],
                    "manual_judgment": row["manual_judgment"],
                    "metric_verdict": row["metric_verdict"],
                    "failure_type": row["failure_type"],
                    "verdict": row["verdict"],
                    "analysis": row["analysis"]
                })

        return str(output_path)

    def export_comparison_json(self, output_path: Optional[Path] = None) -> str:
        """Export comparison results as JSON"""
        
        if not output_path:
            output_path = Path("comparison_analysis.json")

        report = self.generate_comparison_report(
            [{"test_id": i} for i in range(len(self.comparison_rows))]
        )

        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "comparison_results": self.comparison_rows,
                    "findings": self.findings
                },
                f,
                indent=2
            )

        return str(output_path)
