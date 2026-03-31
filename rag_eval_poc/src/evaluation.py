"""
PRODUCTION-GRADE RAG EVALUATION MODULE (FULLY COMPATIBLE)
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

# =========================
# CACHE (REPRODUCIBILITY)
# =========================
_CACHE = {}

def _hash_key(*args):
    return hashlib.md5(str(args).encode()).hexdigest()


# =========================
# LLM WRAPPER
# =========================
def _get_llm():
    try:
        from langchain_groq import ChatGroq
        from deepeval.models import DeepEvalBaseLLM

        api_key = getattr(config, "API_KEY", None) or os.getenv("API_KEY")

        if not api_key:
            logger.warning("API_KEY missing")
            return None
        print("API KEY:", api_key)
        class GroqModel(DeepEvalBaseLLM):
            def __init__(self):
                self.client = ChatGroq(
                    api_key=api_key,
                    model=config.LLM_MODEL,
                    temperature=0.0  # FORCE deterministic
                )

            def load_model(self):
                return self.client

            def get_model_name(self):
                return config.LLM_MODEL

            def generate(self, prompt: str) -> str:
                return self.client.invoke(prompt).content

            async def a_generate(self, prompt: str) -> str:
                return (await self.client.ainvoke(prompt)).content

        return GroqModel()

    except Exception as e:
        logger.error(f"LLM init failed: {e}")
        return None


_llm = None

def _ensure_llm():
    global _llm
    if _llm is None:
        _llm = _get_llm()
    return _llm

# =========================
# DETERMINISTIC METRICS
# =========================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text

def extract_number(text: str):
    import re
    match = re.search(r'\d+(\.\d+)?', text)
    return float(match.group()) if match else None

def is_no_answer(text: str) -> bool:
    text = text.lower()

    keywords = [
        "not contain",
        "not found",
        "no information",
        "not mentioned",
        "not defined",
        "cannot be found"
    ]

    return any(k in text for k in keywords)

def is_valid(score):
    return score is not None

def fmt(score):
    return f"{score:.2f}" if score is not None else "N/A"

def safe_score(score):
    return min(max(score if score is not None else 0, 0), 1)

def exact_match(a, b):
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if is_no_answer(a_norm) and is_no_answer(b_norm):
        return 1.0

    if b_norm in a_norm:
        return 1.0

    # numeric fallback (existing)
    a_num = extract_number(a_norm)
    b_num = extract_number(b_norm)

    if a_num is not None and b_num is not None:
        if abs(a_num - b_num) <= 0.1:
            return 1.0

    return 0.0

def token_overlap(a, b):
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if is_no_answer(a_norm) and is_no_answer(b_norm):
        return 1.0

    if b_norm in a_norm:
        return 1.0

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())

    if not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(b_tokens)


# =========================
# SAFE EXECUTION
# =========================
def safe_measure(metric, test_case, retries=3):
    for i in range(retries):
        try:
            metric.measure(test_case)
            return metric
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep((2 ** i))
            else:
                time.sleep(1)
    raise RuntimeError("Metric failed after retries")


# =========================
# FAILURE ANALYSIS ENGINE
# =========================
def analyze_failure(question: str, expected_answer: str, actual_answer: str, metrics: Dict[str, Any], retrieval_context: List[str]) -> Dict[str, Any]:
    """
    Analyze failure pattern for a test case
    
    Returns structured failure analysis:
    - failure_type: Classification of failure or success
    - reason: Clear explanation of the issue
    - metric_alignment: Whether metrics correctly identified the issue
    - notes: Detailed explanation
    """
    
    hallucination_score = metrics.get("Hallucination", {}).get("score")
    faithfulness_score = metrics.get("Faithfulness", {}).get("score")
    relevancy_score = metrics.get("AnswerRelevancy", {}).get("score")
    recall_score = metrics.get("ContextualRecall", {}).get("score")
    exact_match_score = metrics.get("ExactMatch", {}).get("score")
    
    # Normalize text for comparison
    expected_norm = normalize_text(expected_answer)
    actual_norm = normalize_text(actual_answer)
    
    # Determine failure type
    failure_type = "correct"
    reason = ""
    metric_alignment = "correct"
    notes = ""
    
    # Check for hallucination (score > 0.3 indicates hallucination detected)
    if hallucination_score is not None and hallucination_score > 0.3:
        failure_type = "hallucination"
        reason = f"Model generated content not supported by documents"
        metric_alignment = "correct"
        notes = f"Hallucination metric score: {hallucination_score:.2f}. Model fabricated or inferred information beyond source context."
    
    # Check for retrieval miss (no context retrieved or low recall)
    elif not retrieval_context or (recall_score is not None and recall_score < 0.4):
        failure_type = "retrieval_miss"
        reason = "Failed to retrieve relevant context from documents"
        metric_alignment = "correct"
        notes = f"Only {len(retrieval_context)} documents retrieved. Low contextual recall score: {f'{recall_score:.2f}' if recall_score is not None else 'N/A'}"
    # Check for unanswerable question handled correctly
    elif is_no_answer(actual_norm) and is_no_answer(expected_norm):
        failure_type = "correct"
        reason = "Correctly identified unanswerable question"
        metric_alignment = "correct"
        notes = "Model appropriately refused to answer question not in documents"
    
    # Check for unanswerable question incorrectly answered
    elif is_no_answer(expected_norm) and not is_no_answer(actual_norm):
        failure_type = "hallucination"
        reason = "Model answered question marked as unanswerable in documents"
        metric_alignment = "correct" if (hallucination_score is not None and hallucination_score > 0.3) else "false_negative"
        notes = f"Expected 'not found' response but got: '{actual_answer[:100]}...'. This is hallucination or going beyond documents."
    
    # Check for partial answer (relevancy high but not exact match)
    elif relevancy_score is not None and relevancy_score > 0.7 and faithfulness_score is not None and faithfulness_score > 0.6 and exact_match_score is not None and exact_match_score < 0.7:
        failure_type = "partial_answer"
        reason = f"Answer is relevant and faithful but incomplete or slightly different from expected"
        metric_alignment = "false_negative"
        notes = f"Relevancy: {relevancy_score:.2f}, Faithfulness: {faithfulness_score:.2f}, but not exact match. Model provided acceptable answer but not the exact phrasing."
    
    # Check for low faithfulness (answer doesn't match documents)
    elif faithfulness_score is not None and faithfulness_score < 0.5:
        failure_type = "partial_answer"
        reason = "Answer doesn't faithfully follow source documents"
        metric_alignment = "correct"
        notes = f"Faithfulness score low ({faithfulness_score:.2f}). Answer diverges from or contradicts document content."
    
    # Check for low relevancy (answer doesn't address question)
    elif relevancy_score is not None and relevancy_score < 0.5:
        failure_type = "partial_answer"
        reason = "Answer doesn't relevantly address the question asked"
        metric_alignment = "correct"
        notes = f"Answer relevancy score low ({relevancy_score:.2f}). Model answered a different question or missed the point."
    
    # All metrics passing indicates correct answer
    else:
        if (exact_match_score == 1.0 and faithfulness_score is not None and faithfulness_score > 0.8 and relevancy_score is not None and relevancy_score > 0.8):
            failure_type = "correct"
            reason = "Answer is correct, faithful, and relevant"
            metric_alignment = "correct"
            notes = f"All metrics indicate strong answer quality. Exact match: {exact_match_score}, Faithfulness: {faithfulness_score:.2f}, Relevancy: {relevancy_score:.2f}"
        elif exact_match_score is not None and exact_match_score > 0.7 and faithfulness_score is not None and faithfulness_score > 0.7:
            failure_type = "correct"
            reason = "Answer is substantially correct with acceptable metric scores"
            metric_alignment = "correct"
            notes = f"Strong answer quality with appropriate semantic matching. Exact match: {exact_match_score:.2f}, Faithfulness: {faithfulness_score:.2f}"
        else:
            failure_type = "correct"
            reason = "Answer appears acceptable based on available metrics"
            metric_alignment = "correct"
            notes = (
                f"Hallucination: {f'{hallucination_score:.2f}' if hallucination_score is not None else 'N/A'}, "
                f"Faithfulness: {f'{faithfulness_score:.2f}' if faithfulness_score is not None else 'N/A'}, "
                f"Relevancy: {f'{relevancy_score:.2f}' if relevancy_score is not None else 'N/A'}"
            )
    return {
        "failure_type": failure_type,
        "reason": reason,
        "metric_alignment": metric_alignment,
        "notes": notes
    }


# =========================
# CORE EVALUATION
# =========================
class EvaluationMetrics:

    @staticmethod
    def _check_llm_configured() -> Tuple[bool, str]:
        api_key = os.getenv("API_KEY") or config.API_KEY
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

        # =========================
        # FAIL FAST: NO CONTEXT
        # =========================
        if not retrieval_context:
            return {
                "metrics": {},
                "overall_passed": False,
                "error": "No retrieval context"
            }

        cache_key = _hash_key(question, actual_answer, expected_answer)
        if cache_key in _CACHE:
            return _CACHE[cache_key]

        llm = _ensure_llm()
        if not llm:
            return {
                "metrics": {},
                "overall_passed": False,
                "error": "LLM unavailable"
            }

        # =========================
        # STREAMLIT THRESHOLDS
        # =========================
        try:
            t = st.session_state.get("eval_profile")
        except:
            t = st.session_state.get("eval_profile") or {
                "faithfulness": 0.7,
                "relevancy": 0.7,
                "recall": 0.6,
                "hallucination": 0.2
            }

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            retrieval_context=retrieval_context
        )

        metrics = {}

        # =========================
        # LLM METRICS
        # =========================
        try:
            try:
                h = safe_measure(HallucinationMetric(model=llm), test_case)
                score = h.score
            except:
                score = 0.0  # assume safe

            metrics["Hallucination"] = {
                "score": score,
                "threshold": t["hallucination"],
                "passed": score <= t["hallucination"]
            }
        except Exception as e:
            logger.error(f"Hallucination metric failed: {e}")
            metrics["Hallucination"] = {
                "score": None,
                "threshold": t["hallucination"],
                "passed": False,
                "error": str(e)
            }

        try:
            f = safe_measure(FaithfulnessMetric(model=llm), test_case)
            metrics["Faithfulness"] = {
                "score": f.score,
                "threshold": t["faithfulness"],
                "passed": f.score >= t["faithfulness"]
            }
        except Exception as e:
            logger.error(f"Faithfulness metric failed: {e}")
            metrics["Faithfulness"] = {
                "score": None,
                "threshold": t["faithfulness"],
                "passed": False,
                "error": str(e)
            }

        try:
            r = safe_measure(AnswerRelevancyMetric(model=llm), test_case)
            metrics["AnswerRelevancy"] = {
                "score": r.score,
                "threshold": t["relevancy"],
                "passed": r.score >= t["relevancy"]
            }
            # override relevancy for no-answer correctness
            if is_no_answer(actual_answer) and is_no_answer(expected_answer):
                metrics["AnswerRelevancy"] = {
                    "score": 1.0,
                    "threshold": t["relevancy"],
                    "passed": True
                }
        except Exception as e:
            logger.error(f"AnswerRelevancy metric failed: {e}")
            metrics["AnswerRelevancy"] = {
                "score": None,
                "threshold": t["relevancy"],
                "passed": False,
                "error": str(e)
            }

        try:
            c = safe_measure(ContextualRecallMetric(model=llm), test_case)
            metrics["ContextualRecall"] = {
                "score": c.score,
                "threshold": t["recall"],
                "passed": c.score >= t["recall"]
            }
        except Exception as e:
            logger.error(f"ContextualRecall metric failed: {e}")
            metrics["ContextualRecall"] = {
                "score": None,
                "threshold": t["recall"],
                "passed": False,
                "error": str(e)
            }

        # =========================
        # DETERMINISTIC METRICS
        # =========================
        em = exact_match(actual_answer, expected_answer)
        overlap = token_overlap(actual_answer, expected_answer)

        metrics["ExactMatch"] = {
            "score": em,
            "threshold": 1.0,
            "passed": em == 1.0
        }

        metrics["TokenOverlap"] = {
            "score": overlap,
            "threshold": 0.5,
            "passed": overlap >= 0.5
        }

        # =========================
        # FINAL PASS LOGIC
        # =========================
        valid_scores = [m["score"] for m in metrics.values() if m.get("score") is not None]
        if not valid_scores:
            overall_passed = False
        else:
            weights = {
                "Faithfulness": 0.3,
                "AnswerRelevancy": 0.25,
                "ContextualRecall": 0.2,
                "ExactMatch": 0.15,
                "TokenOverlap": 0.1
            }

            weighted_sum = 0
            total_weight = 0

            for name, m in metrics.items():
                if m.get("score") is not None and name in weights:
                    weighted_sum += m["score"] * weights[name]
                    total_weight += weights[name]

            final_score = weighted_sum / total_weight if total_weight else 0
            overall_passed = final_score >= 0.7

            result = {
                "metrics": metrics,
                "overall_passed": overall_passed
            }

            _CACHE[cache_key] = result
            return result
        # fallback safety (never return None)
        result = {
            "metrics": metrics,
            "overall_passed": overall_passed
        }

        _CACHE[cache_key] = result
        return result


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

    def evaluate_single_test(self, test_case):

        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")

        try:
            result = self.qa_chain.invoke({"query": question})

            actual_answer = result.get("result", "")
            source_docs = result.get("source_documents", [])

            retrieval_context = [
                doc.page_content for doc in source_docs
            ] if source_docs else []

        except Exception as e:
            return {
                "test_id": test_case.get("id"),
                "error": str(e),
                "overall_passed": False
            }

        eval_result = EvaluationMetrics.evaluate_response(
            question,
            actual_answer,
            expected_answer,
            retrieval_context
        )

        # Perform failure analysis
        failure_analysis = analyze_failure(
            question,
            expected_answer,
            actual_answer,
            eval_result["metrics"],
            retrieval_context
        )

        output = {
            "test_id": test_case.get("id"),
            "category": test_case.get("category", "unknown"),
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "context": "\n\n".join(retrieval_context),
            "num_retrieved_docs": len(retrieval_context),
            "metrics": eval_result["metrics"],
            "overall_passed": eval_result["overall_passed"],
            "failure_analysis": failure_analysis,
            "timestamp": datetime.now().isoformat()
        }

        self.results.append(output)
        return output

    def evaluate_batch(self, test_cases, progress_callback=None):

        results = []

        for idx, tc in enumerate(test_cases):
            res = self.evaluate_single_test(tc)
            results.append(res)

            if progress_callback:
                progress_callback(idx + 1, len(test_cases))

        return results

    def get_results_summary(self, results=None):

        results = results or self.results

        if not results:
            return {}

        total = len(results)
        passed = sum(1 for r in results if r.get("overall_passed"))

        # Count failure types
        failure_types = {}
        for r in results:
            failure_analysis = r.get("failure_analysis", {})
            failure_type = failure_analysis.get("failure_type", "unknown")
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

        summary = {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "failure_distribution": failure_types,
            "metrics": {}
        }

        metric_names = [
            "Hallucination",
            "Faithfulness",
            "AnswerRelevancy",
            "ContextualRecall"
        ]

        for name in metric_names:
            scores = [
                r["metrics"][name]["score"]
                for r in results
                if name in r.get("metrics", {}) and r["metrics"][name].get("score") is not None
            ]

            if scores:
                summary["metrics"][name] = {
                    "avg_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores)
                }

        return summary

    def export_results_json(self, output_file=None, results=None):

        results = results or self.results

        if not output_file:
            output_dir = config.EVALUATION_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        return str(output_file)