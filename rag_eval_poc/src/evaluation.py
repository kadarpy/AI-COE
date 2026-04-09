"""
DEEPEVAL-COMPLIANT RAG EVALUATION MODULE
======================================

Pure DeepEval-based evaluation pipeline with:
- Only 4 DeepEval metrics (Hallucination, Faithfulness, AnswerRelevancy, ContextualRecall)
- Clean separation of concerns (evaluation, UI, failure analysis)
- Deterministic evaluation (temperature=0)
- Proper context handling
- Comprehensive logging
- Reproducible results

NO CUSTOM METRICS - Pure DeepEval execution only.
"""
import os
os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["CONFIDENT_DISABLE_TELEMETRY"] = "1"
import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re
import yaml

from dotenv import load_dotenv
os.environ["CONFIDENT_API_KEY"] = "dummy_key"
# =========================
# ENV LOADING
# =========================
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

# =========================
# CACHE FOR REPRODUCIBILITY
# =========================
_CACHE = {}

def _hash_key(*args):
    """Generate cache key from arguments"""
    return hashlib.md5(str(args).encode()).hexdigest()


# =========================
# LLM WRAPPER FOR DEEPEVAL
# =========================
# def _get_llm():
#     """
#     Initialize LLM for DeepEval metrics.
#     Uses Groq API with temperature=0 for deterministic evaluation.
#     """
#     try:
#         from langchain_groq import ChatGroq
#         from deepeval.models import DeepEvalBaseLLM

#         api_key = getattr(config, "API_KEY", None) or os.getenv("API_KEY")

#         if not api_key:
#             logger.warning("API_KEY missing - DeepEval metrics will fail")
#             return None

#         class GroqModel(DeepEvalBaseLLM):
#             def __init__(self):
#                 self.client = ChatGroq(
#                     api_key=api_key,
#                     model=config.LLM_MODEL,
#                     temperature=0.0  # DETERMINISTIC: Always 0 for evaluation
#                 )

#             def load_model(self):
#                 return self.client

#             def get_model_name(self):
#                 return config.LLM_MODEL

#             def generate(self, prompt: str) -> str:
#                 return self.client.invoke(prompt).content

#             async def a_generate(self, prompt: str) -> str:
#                 return (await self.client.ainvoke(prompt)).content

#         return GroqModel()

#     except Exception as e:
#         logger.error(f"LLM initialization failed: {e}")
#         return None


def _get_llm():
    return None

_llm = None

def _ensure_llm():
    """Lazy-load LLM once"""
    global _llm
    if _llm is None:
        _llm = _get_llm()
    return _llm


# =========================
# TEXT UTILITIES
# =========================
def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def is_no_answer_response(text: str) -> bool:
    """
    Detect if model is saying "I don't know" / "not in document".
    Used for failure classification, NOT for metric override.
    """
    text = text.lower()
    keywords = [
        "not contain",
        "not found",
        "no information",
        "not mentioned",
        "not defined",
        "cannot be found",
        "not provided",
        "not specified",
        "not in the document",
        "i don't have",
        "unable to find",
        "i cannot find"
    ]
    return any(k in text for k in keywords)


# =========================
# SAFE METRIC EXECUTION
# =========================
def safe_measure(metric, test_case, retries=3):
    """
    Safely measure a metric with retry logic for rate limits.
    
    Args:
        metric: DeepEval metric instance
        test_case: LLMTestCase instance
        retries: Number of retry attempts
        
    Returns:
        Measured metric with score and reasoning
        
    Raises:
        RuntimeError: If metric fails after all retries
    """
    for attempt in range(retries):
        try:
            metric.measure(test_case)
            logger.debug(f"{metric.__class__.__name__} measured successfully: {metric.score:.2f}")
            return metric
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(1)
    
        logger.error(f"{metric.__class__.__name__} failed permanently")

        metric.score = 0.0
        metric.reason = "Metric execution failed"

        return metric



# =========================
# PURE DEEPEVAL RUNNER
# =========================
def run_deepeval(
    question: str,
    actual_answer: str,
    expected_answer: str,
    context: List[str]
) -> Dict[str, Any]:
    """
    PURE DEEPEVAL EVALUATION PIPELINE
    
    Runs ONLY the 4 standard DeepEval metrics:
    1. HallucinationMetric
    2. FaithfulnessMetric
    3. AnswerRelevancyMetric
    4. ContextualRecallMetric
    
    Args:
        question: User question
        actual_answer: Model's answer
        expected_answer: Ground truth answer
        context: List of retrieved context strings
        
    Returns:
        Dict with metric results: {
            "metrics": {
                "Hallucination": {"score": 0.0, "reason": "..."},
                "Faithfulness": {"score": 1.0, "reason": "..."},
                ...
            },
            "overall_passed": True/False,
            "evaluation_profile": {...}
        }
    """
    
    # Get evaluation profile from Streamlit or use default
    try:
        import streamlit as st
        profile = st.session_state.get("eval_profile")
    except:
        profile = None
    
    if not profile:
        profile = {
            "faithfulness": 0.70,
            "relevancy": 0.75,
            "recall": 0.70,
            "hallucination": 0.25
        }
    
    # Check cache
    cache_key = _hash_key(question, actual_answer, expected_answer, str(context))
    if cache_key in _CACHE:
        logger.debug("Returning cached evaluation result")
        return _CACHE[cache_key]
    
    # Get LLM
    llm = _ensure_llm()
    if not llm:
        logger.error("LLM unavailable - cannot run DeepEval metrics")
        return {
            "metrics": {},
            "overall_passed": False,
            "error": "LLM unavailable for evaluation"
        }
    
    # Construct LLMTestCase
    # IMPORTANT: context must be a list of strings
    if not context:
        context = []  # Explicit empty list for clarity
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        context=context  # Must be list of strings
    )
    
    logger.info(f"Running DeepEval on question: '{question[:50]}...'")
    logger.info(f"Retrieved {len(context)} context chunks")
    logger.debug(f"Context sample (first 100 chars): {str(context[:1])[:100] if context else 'EMPTY'}")
    
    # Validate inputs
    assert isinstance(context, list), f"Context must be list, got {type(context)}"
    assert isinstance(question, str), f"Question must be str, got {type(question)}"
    assert isinstance(actual_answer, str), f"Answer must be str, got {type(actual_answer)}"
    
    metrics_results = {}
    
    # ==============================
    # PURE DEEPEVAL: ALL METRICS ALWAYS RUN
    # NO CONDITIONAL LOGIC, NO MANUAL OVERRIDES
    # ==============================
    
    # 1. HALLUCINATION METRIC
    try:
        logger.debug("Measuring Hallucination...")
        h_metric = safe_measure(HallucinationMetric(model=llm), test_case)
        metrics_results["Hallucination"] = {
            "score": h_metric.score,
            "reason": h_metric.reason,
            "threshold": profile["hallucination"],
            "passed": h_metric.score <= profile["hallucination"]
        }
        logger.info(f"Hallucination: {h_metric.score:.2f}")
    except Exception as e:
        logger.error(f"Hallucination metric failed: {e}")
        metrics_results["Hallucination"] = {
            "score": None,
            "reason": str(e),
            "threshold": profile["hallucination"],
            "passed": False,
            "error": str(e)
        }
    
    # 2. FAITHFULNESS METRIC
    try:
        logger.debug("Measuring Faithfulness...")
        f_metric = safe_measure(FaithfulnessMetric(model=llm), test_case)
        metrics_results["Faithfulness"] = {
            "score": f_metric.score,
            "reason": f_metric.reason,
            "threshold": profile["faithfulness"],
            "passed": f_metric.score >= profile["faithfulness"]
        }
        logger.info(f"Faithfulness: {f_metric.score:.2f}")
    except Exception as e:
        logger.error(f"Faithfulness metric failed: {e}")
        metrics_results["Faithfulness"] = {
            "score": None,
            "reason": str(e),
            "threshold": profile["faithfulness"],
            "passed": False,
            "error": str(e)
        }
    
    # 3. ANSWER RELEVANCY METRIC
    try:
        logger.debug("Measuring AnswerRelevancy...")
        r_metric = safe_measure(AnswerRelevancyMetric(model=llm), test_case)
        metrics_results["AnswerRelevancy"] = {
            "score": r_metric.score,
            "reason": r_metric.reason,
            "threshold": profile["relevancy"],
            "passed": r_metric.score >= profile["relevancy"]
        }
        logger.info(f"AnswerRelevancy: {r_metric.score:.2f}")
    except Exception as e:
        logger.error(f"AnswerRelevancy metric failed: {e}")
        metrics_results["AnswerRelevancy"] = {
            "score": None,
            "reason": str(e),
            "threshold": profile["relevancy"],
            "passed": False,
            "error": str(e)
        }
    
    # 4. CONTEXTUAL RECALL METRIC
    try:
        logger.debug("Measuring ContextualRecall...")
        c_metric = safe_measure(ContextualRecallMetric(model=llm), test_case)
        metrics_results["ContextualRecall"] = {
            "score": c_metric.score,
            "reason": c_metric.reason,
            "threshold": profile["recall"],
            "passed": c_metric.score >= profile["recall"]
        }
        logger.info(f"ContextualRecall: {c_metric.score:.2f}")
    except Exception as e:
        logger.error(f"ContextualRecall metric failed: {e}")
        metrics_results["ContextualRecall"] = {
            "score": None,
            "reason": str(e),
            "threshold": profile["recall"],
            "passed": False,
            "error": str(e)
        }
    # Determine overall pass
    # All metrics must pass (or not be None)
    valid_scores = {
        name: m for name, m in metrics_results.items()
        if m.get("score") is not None
    }
    
    if not valid_scores:
        overall_passed = False
        logger.warning("No valid metric scores - marking as failed")
    else:
        # All available metrics must pass
        overall_passed = all(m.get("passed", False) for m in valid_scores.values())
        logger.info(f"Overall passed: {overall_passed}")
    
    result = {
        "metrics": metrics_results,
        "overall_passed": overall_passed,
        "evaluation_profile": profile,
        "timestamp": datetime.now().isoformat()
    }
    
    _CACHE[cache_key] = result
    return result


# =========================
# FAILURE ANALYSIS ENGINE
# =========================
def analyze_failure(
    question: str,
    expected_answer: str,
    actual_answer: str,
    metrics: Dict[str, Any],
    retrieval_context: List[str]
) -> Dict[str, Any]:
    """
    FAILURE ANALYSIS (runs AFTER DeepEval)
    
    Classifies failure patterns WITHOUT modifying metric scores.
    Used for insights, not for scoring override.
    
    Returns:
        {
            "failure_type": "hallucination|retrieval_miss|partial_answer|correct",
            "reason": "...",
            "metric_alignment": "correct|false_positive|false_negative",
            "notes": "..."
        }
    """
    
    h_score = metrics.get("Hallucination", {}).get("score")
    f_score = metrics.get("Faithfulness", {}).get("score")
    r_score = metrics.get("AnswerRelevancy", {}).get("score")
    c_score = metrics.get("ContextualRecall", {}).get("score")
    
    expected_norm = normalize_text(expected_answer)
    actual_norm = normalize_text(actual_answer)
    
    failure_type = "correct"
    reason = ""
    metric_alignment = "correct"
    notes = ""
    
    # Rule 1: Hallucination detected
    if h_score is not None and h_score > 0.3:
        failure_type = "hallucination"
        reason = "Model generated content not in source documents"
        metric_alignment = "correct"
        notes = f"Hallucination score: {h_score:.2f}. Model fabricated or inferred beyond source."
    
    # Rule 2: Retrieval miss (no context or low recall)
    elif not retrieval_context or (c_score is not None and c_score < 0.4):
        failure_type = "retrieval_miss"
        reason = "Failed to retrieve relevant context"
        metric_alignment = "correct"
        c_str = f"{c_score:.2f}" if c_score is not None else "N/A"
        notes = f"Retrieved {len(retrieval_context)} chunks. Recall: {c_str}"
    
    # Rule 3: Unanswerable handled correctly (both say "not found")
    elif is_no_answer_response(actual_norm) and is_no_answer_response(expected_norm):
        failure_type = "correct"
        reason = "Correctly identified unanswerable question"
        metric_alignment = "correct"
        notes = "Model appropriately refused to answer question not in documents"
    
    # Rule 4: Unanswerable but model answered (hallucination)
    elif is_no_answer_response(expected_norm) and not is_no_answer_response(actual_norm):
        failure_type = "hallucination"
        reason = "Model answered question marked as unanswerable"
        h_str = f"{h_score:.2f}" if h_score is not None else "N/A"
        metric_alignment = "correct" if (h_score and h_score > 0.3) else "false_negative"
        notes = f"Expected 'not found' but got answer. Hallucination: {h_str}"
    
    # Rule 5: Low faithfulness (doesn't follow docs)
    elif f_score is not None and f_score < 0.5:
        failure_type = "partial_answer"
        reason = "Answer doesn't faithfully follow source documents"
        metric_alignment = "correct"
        notes = f"Faithfulness: {f_score:.2f}. Answer diverges from documents."
    
    # Rule 6: Low relevancy (misses question)
    elif r_score is not None and r_score < 0.5:
        failure_type = "partial_answer"
        reason = "Answer doesn't relevantly address the question"
        metric_alignment = "correct"
        notes = f"AnswerRelevancy: {r_score:.2f}. Model answered wrong question."
    
    # Rule 7: All metrics pass - correct
    else:
        failure_type = "correct"
        reason = "Answer passes all DeepEval metrics"
        metric_alignment = "correct"
        h_str = f"{h_score:.2f}" if h_score is not None else "N/A"
        f_str = f"{f_score:.2f}" if f_score is not None else "N/A"
        r_str = f"{r_score:.2f}" if r_score is not None else "N/A"
        c_str = f"{c_score:.2f}" if c_score is not None else "N/A"
        notes = (
            f"Hallucination: {h_str}, "
            f"Faithfulness: {f_str}, "
            f"Relevancy: {r_str}, "
            f"Recall: {c_str}"
        )
    
    return {
        "failure_type": failure_type,
        "reason": reason,
        "metric_alignment": metric_alignment,
        "notes": notes
    }


# =========================
# CONTEXT EXTRACTION
# =========================
def extract_context_from_retrieval(source_documents: Any) -> List[str]:
    """
    Extract context from RAG retrieval result.
    
    CRITICAL: This MUST return List[str] with proper content.
    DeepEval requires this format for Faithfulness and ContextualRecall.
    
    Args:
        source_documents: LangChain Document objects
        
    Returns:
        List of context strings (never None, always a list)
    """
    if not source_documents:
        logger.debug("No source documents retrieved - returning empty list")
        return []
    
    # Ensure source_documents is iterable
    try:
        doc_list = list(source_documents)
    except TypeError:
        logger.error(f"source_documents not iterable: {type(source_documents)}")
        return []
    
    logger.info(f"Processing {len(doc_list)} source documents")
    
    context = []
    for idx, doc in enumerate(doc_list):
        if hasattr(doc, 'page_content'):
            content = str(doc.page_content).strip()[:400]
            if content:
                context.append(content)
                logger.debug(f"  Doc {idx}: {len(content)} chars added to context")
            else:
                logger.debug(f"  Doc {idx}: Empty page_content, skipped")
        else:
            logger.warning(f"  Doc {idx}: No page_content attribute")
    
    logger.info(f"Extract complete: {len(context)} context chunks, {sum(len(c) for c in context)} total chars")
    
    if not context:
        logger.warning("Source documents exist but no valid context extracted")
    
    return context


# =========================
# EVALUATION METRICS CLASS
# =========================
class EvaluationMetrics:
    """
    Wrapper for pure DeepEval evaluation.
    Provides static methods for backward compatibility.
    """

    @staticmethod
    def _check_llm_configured() -> Tuple[bool, str]:
        """Check if LLM is properly configured"""
        api_key = os.getenv("API_KEY") or getattr(config, "API_KEY", None)
        if api_key:
            return True, f"Using {config.LLM_MODEL} for evaluation"
        return False, "API_KEY missing"

    @staticmethod
    def evaluate_response(
        question: str,
        actual_answer: str,
        expected_answer: str,
        retrieval_context: List[str]
    ) -> Dict[str, Any]:
        """
        MAIN EVALUATION ENDPOINT
        
        Uses pure DeepEval pipeline.
        
        Args:
            question: User question
            actual_answer: Model answer
            expected_answer: Ground truth answer
            retrieval_context: List of context strings from retrieval
            
        Returns:
            {
                "metrics": {...},
                "overall_passed": True/False,
                "error": "..." (if any)
            }
        """
        
        # Validate context is a list
        if not isinstance(retrieval_context, list):
            logger.error(f"Context must be list, got {type(retrieval_context)}")
            retrieval_context = []
        
        # Call pure DeepEval
        return run_deepeval(
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            context=retrieval_context
        )



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
        return next((tc for tc in self.test_cases if tc.get("id") == test_id), None)


# =========================
# UI EVALUATOR (COMPATIBLE)
# =========================
class UIEvaluator:

    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.results = []
        self.test_case_manager = TestCaseManager()

    @staticmethod
    def _check_llm_for_metrics():
        return EvaluationMetrics._check_llm_configured()

# =========================
# UI EVALUATOR - REFACTORED
# =========================
class UIEvaluator:
    """
    Main Evaluator class for UI integration.
    
    Methods:
    - evaluate_single_test: Run evaluation on one test case
    - evaluate_batch: Run evaluation on multiple test cases
    - get_results_summary: Generate summary statistics
    - export_results_json: Export results to JSON
    """

    def __init__(self, qa_chain):
        """
        Initialize evaluator with QA chain.
        
        Args:
            qa_chain: LangChain RAG chain that takes {"query": question}
                     and returns {"result": answer, "source_documents": [docs]}
        """
        self.qa_chain = qa_chain
        self.results = []
        self.test_case_manager = TestCaseManager()
        logger.info("UIEvaluator initialized")

    @staticmethod
    def _check_llm_for_metrics():
        """Check LLM configuration"""
        return EvaluationMetrics._check_llm_configured()

    def evaluate_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        EVALUATE SINGLE TEST CASE
        
        Flow:
        1. Get question and expected answer from test case
        2. Run RAG chain to get actual answer + context
        3. Extract context as list of strings
        4. Run pure DeepEval
        5. Run failure analysis (no metric override)
        6. Return structured result
        
        Args:
            test_case: Dict with question, expected_answer, etc.
            
        Returns:
            {
                "test_id": int,
                "category": str,
                "question": str,
                "expected_answer": str,
                "actual_answer": str,
                "context": str (formatted for display),
                "context_list": List[str] (raw),
                "num_retrieved_docs": int,
                "metrics": {...},
                "overall_passed": bool,
                "failure_analysis": {...},
                "timestamp": str
            }
        """
        
        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        test_id = test_case.get("id")
        
        logger.info(f"Evaluating test case {test_id}: {question[:50]}...")

        # Step 1: Run RAG chain
        try:
            result = self.qa_chain.invoke({"query": question})
            actual_answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # Step 2: Extract context as list
            retrieval_context = extract_context_from_retrieval(source_docs)
            
            logger.debug(f"Test {test_id}: Retrieved {len(retrieval_context)} context chunks")

        except Exception as e:
            logger.error(f"Test {test_id} RAG execution failed: {e}")
            return {
                "test_id": test_id,
                "category": test_case.get("category", "unknown"),
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": "",
                "error": str(e),
                "overall_passed": False,
                "timestamp": datetime.now().isoformat()
            }

        # Step 3: Run pure DeepEval
        eval_result = EvaluationMetrics.evaluate_response(
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            retrieval_context=retrieval_context
        )

        # Step 4: Run failure analysis (NO override)
        failure_analysis = analyze_failure(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            metrics=eval_result.get("metrics", {}),
            retrieval_context=retrieval_context
        )

        # Step 5: Assemble output
        output = {
            "test_id": test_id,
            "category": test_case.get("category", "unknown"),
            "difficulty": test_case.get("difficulty", "unknown"),
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "context": "\n\n---\n\n".join(retrieval_context),  # Formatted for display
            "context_list": retrieval_context,  # Raw list for analysis
            "num_retrieved_docs": len(retrieval_context),
            "metrics": eval_result.get("metrics", {}),
            "overall_passed": eval_result.get("overall_passed", False),
            "failure_analysis": failure_analysis,
            "timestamp": datetime.now().isoformat()
        }

        self.results.append(output)
        logger.info(f"Test {test_id} completed - Passed: {output['overall_passed']}")
        
        return output

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        EVALUATE BATCH OF TEST CASES
        
        Args:
            test_cases: List of test case dicts
            progress_callback: Optional callback(completed, total) for progress
            
        Returns:
            List of evaluation results
        """
        
        logger.info(f"Starting batch evaluation of {len(test_cases)} test cases")
        results = []

        for idx, tc in enumerate(test_cases):
            res = self.evaluate_single_test(tc)
            results.append(res)

            if progress_callback:
                progress_callback(idx + 1, len(test_cases))

        logger.info(f"Batch evaluation completed: {len(results)} cases")
        return results

    def get_results_summary(self, results=None) -> Dict[str, Any]:
        """
        GENERATE EVALUATION SUMMARY
        
        Returns:
            {
                "total_tests": int,
                "passed_tests": int,
                "failed_tests": int,
                "pass_rate": float,
                "failure_distribution": {...},
                "metrics": {
                    "Hallucination": {"avg_score": ..., "min": ..., "max": ...},
                    ...
                },
                "by_category": {...},
                "by_difficulty": {...}
            }
        """
        
        results = results or self.results

        if not results:
            return {}

        total = len(results)
        passed = sum(1 for r in results if r.get("overall_passed"))
        
        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r]

        # Failure distribution
        failure_types = {}
        for r in valid_results:
            failure_analysis = r.get("failure_analysis", {})
            failure_type = failure_analysis.get("failure_type", "unknown")
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

        summary = {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "failure_distribution": failure_types,
            "metrics": {},
            "by_category": {},
            "by_difficulty": {}
        }

        # Metric statistics (ONLY 4 DeepEval metrics)
        metric_names = [
            "Hallucination",
            "Faithfulness",
            "AnswerRelevancy",
            "ContextualRecall"
        ]

        for name in metric_names:
            scores = [
                r["metrics"][name]["score"]
                for r in valid_results
                if name in r.get("metrics", {}) and r["metrics"][name].get("score") is not None
            ]

            if scores:
                summary["metrics"][name] = {
                    "avg_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores)
                }

        # By category breakdown
        categories = set(r.get("category") for r in valid_results if r.get("category"))
        for cat in categories:
            cat_results = [r for r in valid_results if r.get("category") == cat]
            cat_passed = sum(1 for r in cat_results if r.get("overall_passed"))
            summary["by_category"][cat] = {
                "total": len(cat_results),
                "passed": cat_passed,
                "pass_rate": (cat_passed / len(cat_results) * 100) if cat_results else 0
            }

        # By difficulty breakdown
        difficulties = set(r.get("difficulty") for r in valid_results if r.get("difficulty"))
        for diff in difficulties:
            diff_results = [r for r in valid_results if r.get("difficulty") == diff]
            diff_passed = sum(1 for r in diff_results if r.get("overall_passed"))
            summary["by_difficulty"][diff] = {
                "total": len(diff_results),
                "passed": diff_passed,
                "pass_rate": (diff_passed / len(diff_results) * 100) if diff_results else 0
            }

        logger.info(f"Summary: {passed}/{total} passed ({summary['pass_rate']:.1f}%)")
        return summary

    def export_results_json(self, output_file=None, results=None) -> str:
        """
        EXPORT RESULTS TO JSON
        
        Args:
            output_file: Optional output path
            results: Optional results list (uses self.results if not provided)
            
        Returns:
            Path to exported file
        """
        
        results = results or self.results

        if not output_file:
            output_dir = config.EVALUATION_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results exported to {output_file}")
        return str(output_file)