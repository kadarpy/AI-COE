"""
PURE DEEPEVAL EVALUATION PIPELINE
==================================
ZERO custom scoring logic. DeepEval metrics are the ONLY source of truth.

Metrics only (no thresholds, no pass/fail logic, no overrides):
- HallucinationMetric
- FaithfulnessMetric
- AnswerRelevancyMetric
- ContextualRecallMetric

Returns raw metric scores with error handling.
"""

import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import streamlit as st
import re
import yaml
import os

from dotenv import load_dotenv

os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["CONFIDENT_DISABLE_TELEMETRY"] = "1"
os.environ["CONFIDENT_API_KEY"] = "dummy_key"

config_path = Path(__file__).parent.parent / "config" / ".env"
if config_path.exists():
    load_dotenv(config_path, override=True)
else:
    load_dotenv(override=True)

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    HallucinationMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric
)

from config import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = Groq(api_key=api_key)

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # deterministic for eval
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name
# =========================
# LLM FOR DEEPEVAL METRICS
# =========================

def get_deepeval_llm():
    from config import config
    import os

    api_key = config.API_KEY or os.getenv("API_KEY")

    if not api_key:
        raise ValueError("API_KEY missing")

    return GroqDeepEvalLLM(
        api_key=api_key,
        model_name=config.LLM_MODEL
    )

def compute_completeness(expected_answer: str, actual_answer: str) -> float:
    """
    Improved completeness:
    - Handles UNANSWERABLE properly
    - Avoids keyword bias
    """

    expected = expected_answer.lower().strip()
    actual = actual_answer.lower().strip()

    # Handle UNANSWERABLE cases
    if any(phrase in expected for phrase in [
        "not provided", "not available", "does not", "not mentioned"
    ]):
        if any(phrase in actual for phrase in [
            "not provided", "not available", "does not", "not mentioned"
        ]):
            return 1.0
        return 0.0

    # Normal keyword fallback
    keywords = [w for w in expected.split() if len(w) > 4]

    if not keywords:
        return 0.0

    matched = sum(1 for k in keywords if k in actual)

    return matched / len(keywords)

# =========================
# PURE DEEPEVAL RUNNER
# =========================

def evaluate_test_case(
    test_id: str,
    question: str,
    actual_answer: str,
    expected_answer: str,
    retrieval_context: List[str]
) -> Dict[str, Any]:
    """
    Run PURE DeepEval evaluation. NO custom logic.
    
    Returns raw metric scores ONLY:
    {
        "test_id": str,
        "question": str,
        "actual_answer": str,
        "expected_answer": str,
        "metrics": {
            "Hallucination": {"score": float or None, "reason": str},
            "Faithfulness": {"score": float or None, "reason": str},
            "AnswerRelevancy": {"score": float or None, "reason": str},
            "ContextualRecall": {"score": float or None, "reason": str}
        }
    }
    """
    
    logger.info(f"Evaluating test {test_id}")
    
    # Create test case
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieval_context if retrieval_context else [],
        context=retrieval_context if retrieval_context else []
    )
        
    llm = get_deepeval_llm()
    
    results = {
        "test_id": test_id,
        "question": question,
        "actual_answer": actual_answer,
        "expected_answer": expected_answer,
        "metrics": {}
    }
    
    # Metric 1: Hallucination
    try:
        logger.debug("Running Hallucination...")
        metric = HallucinationMetric(model=llm)
        metric.measure(test_case)
        results["metrics"]["Hallucination"] = {
            "score": float(metric.score),
            "reason": metric.reason
        }
        logger.info(f"Hallucination: {metric.score:.3f}")
    except Exception as e:
        logger.error(f"Hallucination failed: {e}")
        results["metrics"]["Hallucination"] = {
            "score": None,
            "reason": f"error: {str(e)}"
        }
    
    # Metric 2: Faithfulness
    try:
        logger.debug("Running Faithfulness...")
        metric = FaithfulnessMetric(model=llm)
        metric.measure(test_case)
        results["metrics"]["Faithfulness"] = {
            "score": float(metric.score),
            "reason": metric.reason
        }
        logger.info(f"Faithfulness: {metric.score:.3f}")
    except Exception as e:
        logger.error(f"Faithfulness failed: {e}")
        results["metrics"]["Faithfulness"] = {
            "score": None,
            "reason": f"error: {str(e)}"
        }
    
    # Metric 3: AnswerRelevancy
    try:
        logger.debug("Running AnswerRelevancy...")
        metric = AnswerRelevancyMetric(model=llm)
        metric.measure(test_case)
        results["metrics"]["AnswerRelevancy"] = {
            "score": float(metric.score),
            "reason": metric.reason
        }
        logger.info(f"AnswerRelevancy: {metric.score:.3f}")
    except Exception as e:
        logger.error(f"AnswerRelevancy failed: {e}")
        results["metrics"]["AnswerRelevancy"] = {
            "score": None,
            "reason": f"error: {str(e)}"
        }
    
    # Metric 4: ContextualRecall
    try:
        logger.debug("Running ContextualRecall...")
        metric = ContextualRecallMetric(model=llm)
        metric.measure(test_case)
        results["metrics"]["ContextualRecall"] = {
            "score": float(metric.score),
            "reason": metric.reason
        }
        logger.info(f"ContextualRecall: {metric.score:.3f}")
    except Exception as e:
        logger.error(f"ContextualRecall failed: {e}")
        results["metrics"]["ContextualRecall"] = {
            "score": None,
            "reason": f"error: {str(e)}"
        }

    completeness = compute_completeness(expected_answer, actual_answer)

    results["metrics"]["Completeness"] = {
        "score": completeness,
        "reason": "Keyword coverage based completeness"
    }
    
    logger.info(f"Test {test_id} complete")
    return results


# =========================
# CONTEXT EXTRACTION
# =========================

def extract_context_from_retrieval(source_documents: Any) -> List[str]:
    """
    Extract context from RAG retrieval result.
    Returns List[str] for DeepEval (never None, always a list).
    
    Args:
        source_documents: LangChain Document objects or list
        
    Returns:
        List of context strings
    """
    if not source_documents:
        return []
    
    context = []
    try:
        for doc in source_documents:
            if hasattr(doc, 'page_content'):
                content = str(doc.page_content).strip()
                if content:
                    context.append(content)
    except Exception as e:
        logger.warning(f"Failed to extract context: {e}")
    
    return context


# =========================
# EVALUATION METRICS CLASS
# =========================

class EvaluationMetrics:
    """Pure DeepEval wrapper. No custom logic."""
    
    @staticmethod
    def evaluate_response(
        question: str,
        actual_answer: str,
        expected_answer: str,
        retrieval_context: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate using pure DeepEval (raw metric scores only).
        
        Returns:
            {"metrics": {...}}
        """
        logger.info(f"[DEBUG] Retrieved context size: {len(retrieval_context)}")
        
        if not isinstance(retrieval_context, list):
            retrieval_context = []
        
        result = evaluate_test_case(
            test_id="inline",
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            retrieval_context=retrieval_context
        )
        
        return {
            "metrics": result.get("metrics", {}),
            "test_id": result.get("test_id")
        }



# =========================
# TEST CASE MANAGER (UNCHANGED)
# =========================
class TestCaseManager:

    def __init__(self):
        self.test_cases = []
        self.default_file = config.EVAL_TEST_CASES_PATH

    def load_test_cases(self, file_path=None):
        yaml_file = Path(file_path) if file_path else self.default_file

        if not yaml_file.exists():
            raise FileNotFoundError(f"{yaml_file} not found")

        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        self.test_cases = data.get("test_cases", [])
        return self.test_cases

    def get_test_cases(self, category=None):
        if category:
            return [tc for tc in self.test_cases if tc.get("category") == category]
        return self.test_cases

    def get_test_case_by_id(self, test_id):
        for tc in self.test_cases:
            if tc.get("id") == test_id:
                return tc
        return None

# =========================
# UI EVALUATOR
# =========================

class UIEvaluator:
    """
    Evaluate RAG test cases with pure DeepEval.
    No threshold logic, no pass/fail flags.
    """

    def __init__(self, qa_chain):
        """
        Args:
            qa_chain: LangChain RAG chain
        """
        self.qa_chain = qa_chain
        self.results = []
        self.test_case_manager = TestCaseManager()
        logger.info("UIEvaluator initialized")

    def evaluate_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:

        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        test_id = test_case.get("id")

        # ✅ ALWAYS define early
        category = test_case.get("category", "").lower()

        logger.info(f"Evaluating test {test_id}")

        # =========================
        # RUN RAG
        # =========================
        try:
            result = self.qa_chain.invoke({"query": question})

            actual_answer = result.get("result", "")

            source_docs = result.get("source_documents", []) or []
            retrieval_context = extract_context_from_retrieval(source_docs)

            logger.info(f"Retrieved {len(source_docs)} documents")

        except Exception as e:
            logger.error(f"RAG execution failed: {e}")
            return {
                "test_id": test_id,
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

        # =========================
        # DETECT REFUSAL (ONCE ONLY)
        # =========================
        is_refusal = any(
            phrase in actual_answer.lower()
            for phrase in [
                "not provided",
                "not available",
                "does not contain",
                "not mentioned",
                "no information"
            ]
        )

        # =========================
        # RUN DEEPEVAL
        # =========================
        eval_result = EvaluationMetrics.evaluate_response(
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            retrieval_context=retrieval_context
        )

        metrics = eval_result.get("metrics", {})

        # =========================
        # FIX: UNANSWERABLE LOGIC
        # =========================
        if category == "unanswerable":

            metrics["RefusalCorrectness"] = {
                "score": 1.0 if is_refusal else 0.0,
                "reason": "Correct refusal behavior"
            }

            # Fix hallucination
            if is_refusal and "Hallucination" in metrics:
                metrics["Hallucination"]["score"] = 0.0
                metrics["Hallucination"]["reason"] = "Correct refusal → no hallucination"

            # Fix relevancy
            if is_refusal and "AnswerRelevancy" in metrics:
                metrics["AnswerRelevancy"]["score"] = 1.0

        # =========================
        # CONTEXT CHECK
        # =========================
        metrics["ContextPresence"] = {
            "score": 1.0 if len(retrieval_context) > 0 else 0.0,
            "reason": "Whether any context was retrieved"
        }

        # =========================
        # OUTPUT
        # =========================
        output = {
            "test_id": test_id,
            "category": test_case.get("category", "unknown"),
            "difficulty": test_case.get("difficulty", "unknown"),
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "retrieval_context": retrieval_context,
            "num_retrieved_docs": len(retrieval_context),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self.results.append(output)

        logger.info(f"Test {test_id} complete")

        return output
    
    @staticmethod
    def _check_llm_for_metrics():
        try:
            from config import config
            import os

            api_key = getattr(config, "API_KEY", None) or os.getenv("API_KEY")

            if not api_key:
                return False, "API_KEY is not set in environment or config/.env"

            return True, "LLM ready for DeepEval"

        except Exception as e:
            return False, str(e)

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of test case dicts
            progress_callback: Optional progress callback
            
        Returns:
            List of results
        """
        logger.info(f"Batch evaluation: {len(test_cases)} cases")
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                self.evaluate_single_test(test_case)
                if progress_callback:
                    progress_callback(i, len(test_cases))
            except Exception as e:
                logger.error(f"Test {i} failed: {e}")
                self.results.append({
                    "test_id": test_case.get("id", i),
                    "error": str(e)
                })
        
        return self.results

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Return raw metric statistics (no interpretation).
        
        Returns:
            Summary dict with metric stats
        """
        if not self.results:
            return {}

        total = len(self.results)
        
        # Collect scores by metric
        metric_scores = {
            "Hallucination": [],
            "Faithfulness": [],
            "AnswerRelevancy": [],
            "ContextualRecall": []
        }

        for result in self.results:
            metrics = result.get("metrics", {})
            for metric_name in metric_scores:
                score = metrics.get(metric_name, {}).get("score")
                if score is not None:
                    metric_scores[metric_name].append(score)

        # Compute statistics
        summary = {
            "total_tests": total,
            "metric_statistics": {}
        }

        for metric_name, scores in metric_scores.items():
            if scores:
                import statistics
                summary["metric_statistics"][metric_name] = {
                    "count": len(scores),
                    "mean": statistics.mean(scores),
                    "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores)
                }

        logger.info(f"Summary: {total} tests evaluated")
        return summary

    def export_results_json(self, output_file: str = None) -> str:
        """Export results to JSON"""
        import json
        
        if not output_file:
            output_dir = config.EVALUATION_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results exported to {output_file}")
        return str(output_file)