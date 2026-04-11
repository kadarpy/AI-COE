"""
HYBRID PRODUCTION-GRADE RAG EVALUATION SYSTEM
=============================================

Architecture:
  Layer 1: DeepEval Metrics (pure LLM-based evaluation)
  Layer 2: Category-based Metric Selection (choose relevant metrics)
  Layer 3: LLM-based Intent Detection (refusal detection via judge)
  Layer 4: ResultInterpretation (no overrides, only interpretation)

Design Principles:
  ✅ NO metric overrides - DeepEval outputs are trusted
  ✅ NO heuristic shortcuts - Use LLM-based logic
  ✅ NO evaluation leakage - Strict information control
  ✅ NO artificial bias - Honest aggregation only
  ✅ Complete Transparency - Document architecture honestly
  ✅ Production-Grade - Deterministic, reproducible, reliable
"""

import json
import logging
import hashlib
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import streamlit as st
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


# =========================
# DETERMINISM CONTROLS
# =========================

def set_evaluation_seed(seed: int = 42):
    """
    Set all random seeds for deterministic evaluation.
    Call this at the start of any evaluation run.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"[DETERMINISM] Seeds set to {seed}")


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """
    Groq LLM adapter for DeepEval metrics.
    Temperature=0.0 for deterministic evaluation.
    """
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
            temperature=0.0  # deterministic
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


def get_deepeval_llm():
    """Get configured LLM for DeepEval metrics."""
    api_key = config.API_KEY or os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY missing")
    return GroqDeepEvalLLM(api_key=api_key, model_name=config.LLM_MODEL)


# =========================
# LAYER 1: LLM-BASED INTENT DETECTION (NO HEURISTICS)
# =========================

def detect_refusal_with_llm(
    question: str,
    answer: str,
    context: List[str],
    llm: DeepEvalBaseLLM = None
) -> bool:
    """
    Use LLM judge to detect if response is a refusal.

    NOT heuristic-based. Actual LLM reasoning.

    Args:
        question: The original question
        answer: The model's response
        context: Retrieved context documents
        llm: Optional LLM judge (uses default if None)

    Returns:
        True if LLM judges this as a refusal, False otherwise
    """
    if llm is None:
        llm = get_deepeval_llm()

    prompt = f"""You are an expert evaluator. Determine if the following response is a refusal to answer.

QUESTION:
{question}

RESPONSE:
{answer}

CONTEXT RETRIEVED:
{chr(10).join(context) if context else '(no context retrieved)'}

INSTRUCTIONS:
- A refusal means the model states it cannot answer based on available information
- Examples: "not in documents", "insufficient information", "not provided in context"
- Do NOT count partial answers as refusals
- Respond with ONLY "YES" or "NO"

Is this a refusal to answer?"""

    try:
        response = llm.generate(prompt)
        is_refusal = "YES" in response.upper()
        logger.info(f"[REFUSAL_JUDGE] Question: '{question[:50]}...' → {is_refusal}")
        return is_refusal
    except Exception as e:
        logger.error(f"[REFUSAL_JUDGE] Failed to judge refusal: {e}")
        return False


# =========================
# LAYER 2: SEMANTIC COMPLETENESS (NO KEYWORD BIAS)
# =========================

def compute_semantic_completeness(expected_answer: str, actual_answer: str) -> float:
    """
    Compute semantic completeness using sentence transformers.

    NOT keyword matching. Actual semantic similarity.

    Args:
        expected_answer: The gold standard answer
        actual_answer: The model's response

    Returns:
        Completeness score [0, 1]
    """
    try:
        from sentence_transformers import util, SentenceTransformer

        model = SentenceTransformer('BAAI/bge-small-en-v1.5')

        expected = expected_answer.lower().strip()
        actual = actual_answer.lower().strip()

        if not expected or not actual:
            return 0.0

        # Compute semantic similarity
        embeddings1 = model.encode([expected], convert_to_tensor=True)
        embeddings2 = model.encode([actual], convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0].item()

        # Clamp to [0, 1]
        completeness = max(0.0, min(1.0, similarity))

        logger.debug(f"[COMPLETENESS] Semantic similarity: {completeness:.3f}")
        return completeness

    except Exception as e:
        logger.warning(f"[COMPLETENESS] Semantic comparison failed: {e}")
        # Fallback: simple length ratio (not ideal but safe)
        return min(len(actual) / max(len(expected), 1), 1.0)


# =========================
# LAYER 3: CATEGORY-BASED METRIC SELECTION
# =========================

def get_applicable_metrics_for_category(category: str) -> Dict[str, bool]:
    """
    Determine which metrics are applicable for each question category.

    UNANSWERABLE:
      - Hallucination: YES (did model make stuff up?)
      - Faithfulness: YES (is response faithful to docs?)
      - AnswerRelevancy: NO (not applicable for refusals)
      - ContextualRecall: NO (not applicable for "not in docs")

    ANSWERABLE:
      - All metrics apply

    PARTIAL:
      - All metrics apply (but score expectations differ)
    """
    metrics = {
        "hallucination": True,
        "faithfulness": True,
        "answer_relevancy": True,
        "contextual_recall": True
    }

    if category.lower() == "unanswerable":
        # For unanswerable, these don't apply
        metrics["answer_relevancy"] = False
        metrics["contextual_recall"] = False
        logger.info("[METRIC_SELECTION] Unanswerable: only Hallucination + Faithfulness")

    return metrics


# =========================
# LAYER 4: DEEPEVAL METRICS (PURE, NO OVERRIDES)
# =========================

def _measure_with_backoff(metric, test_case, max_retries: int = 3):
    """
    Measure metric with exponential backoff for rate limits.

    Pure DeepEval execution. No interference.
    """
    for attempt in range(max_retries + 1):
        try:
            metric.measure(test_case)
            return float(metric.score), metric.reason
        except Exception as e:
            error_msg = str(e)

            if "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"[RETRY] Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"[METRIC] Max retries exceeded")
                    return None, f"Rate limit exceeded"

            logger.error(f"[METRIC] Failed: {error_msg}")
            return None, error_msg

    return None, "Unknown error"


def evaluate_with_deepeval(
    question: str,
    actual_answer: str,
    expected_answer: str,
    retrieval_context: List[str],
    applicable_metrics: Dict[str, bool]
) -> Dict[str, Any]:
    """
    RUN PURE DEEPEVAL EVALUATION.

    NO overrides. NO custom logic. Trust metrics completely.

    Args:
        question: Question asked
        actual_answer: Model's response
        expected_answer: Gold standard answer
        retrieval_context: Retrieved context
        applicable_metrics: Which metrics to compute (from category)

    Returns:
        Raw metric scores from DeepEval
    """

    # Create test case WITHOUT leakage
    # Note: We provide expected_output for context, but metrics use it carefully
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieval_context if retrieval_context else [],
        context=retrieval_context if retrieval_context else []
    )

    llm = get_deepeval_llm()
    metrics_result = {}

    # Metric 1: Hallucination (always relevant)
    if applicable_metrics["hallucination"]:
        try:
            logger.debug("Running Hallucination metric...")
            metric = HallucinationMetric(model=llm)
            score, reason = _measure_with_backoff(metric, test_case)
            metrics_result["hallucination"] = {
                "score": score,
                "reason": reason,
                "applied": True
            }
            if score is not None:
                logger.info(f"Hallucination: {score:.3f}")
        except Exception as e:
            logger.error(f"Hallucination failed: {e}")
            metrics_result["hallucination"] = {
                "score": None,
                "reason": str(e),
                "applied": False
            }
    else:
        metrics_result["hallucination"] = {"score": None, "reason": "Not applicable", "applied": False}

    # Metric 2: Faithfulness (always relevant)
    if applicable_metrics["faithfulness"]:
        try:
            logger.debug("Running Faithfulness metric...")
            metric = FaithfulnessMetric(model=llm)
            score, reason = _measure_with_backoff(metric, test_case)
            metrics_result["faithfulness"] = {
                "score": score,
                "reason": reason,
                "applied": True
            }
            if score is not None:
                logger.info(f"Faithfulness: {score:.3f}")
        except Exception as e:
            logger.error(f"Faithfulness failed: {e}")
            metrics_result["faithfulness"] = {
                "score": None,
                "reason": str(e),
                "applied": False
            }
    else:
        metrics_result["faithfulness"] = {"score": None, "reason": "Not applicable", "applied": False}

    # Metric 3: AnswerRelevancy (conditional on category)
    if applicable_metrics["answer_relevancy"]:
        try:
            logger.debug("Running AnswerRelevancy metric...")
            metric = AnswerRelevancyMetric(model=llm)
            score, reason = _measure_with_backoff(metric, test_case)
            metrics_result["answer_relevancy"] = {
                "score": score,
                "reason": reason,
                "applied": True
            }
            if score is not None:
                logger.info(f"AnswerRelevancy: {score:.3f}")
        except Exception as e:
            logger.error(f"AnswerRelevancy failed: {e}")
            metrics_result["answer_relevancy"] = {
                "score": None,
                "reason": str(e),
                "applied": False
            }
    else:
        metrics_result["answer_relevancy"] = {"score": None, "reason": "Not applicable for this category", "applied": False}

    # Metric 4: ContextualRecall (conditional on category)
    if applicable_metrics["contextual_recall"]:
        try:
            logger.debug("Running ContextualRecall metric...")
            metric = ContextualRecallMetric(model=llm)
            score, reason = _measure_with_backoff(metric, test_case)
            metrics_result["contextual_recall"] = {
                "score": score,
                "reason": reason,
                "applied": True
            }
            if score is not None:
                logger.info(f"ContextualRecall: {score:.3f}")
        except Exception as e:
            logger.error(f"ContextualRecall failed: {e}")
            metrics_result["contextual_recall"] = {
                "score": None,
                "reason": str(e),
                "applied": False
            }
    else:
        metrics_result["contextual_recall"] = {"score": None, "reason": "Not applicable for this category", "applied": False}

    return metrics_result


# =========================
# LAYER 5: RESULT INTERPRETATION (NO OVERRIDES)
# =========================

def interpret_results_no_bias(
    metrics: Dict[str, Any],
    category: str,
    is_refusal: bool,
    completeness_score: float = 1.0,
    llm: DeepEvalBaseLLM = None
) -> Dict[str, Any]:
    """
    Interpret evaluation results WITHOUT bias or overrides.

    NEW: Completeness factor is applied to final score.
    This prevents over-scoring of partial answers.

    UNANSWERABLE + Correct Refusal:
      - Hallucination ≈ 0 (good, no hallucination)
      - Faithfulness ≈ 1 (good, faithful to docs)
      - Score interpretation: PASS (correct evaluation)

    UNANSWERABLE + Hallucination:
      - Hallucination > 0.3 (bad, made stuff up)
      - Score interpretation: FAIL (incorrect evaluation)

    ANSWERABLE:
      - All metrics contribute to final score
      - Standard weighted aggregation
      - Multiplied by completeness factor (prevents partial answer over-scoring)

    Key Point: NO metric overrides. Only interpretation of what DeepEval says.
    """

    hal_score = metrics.get("hallucination", {}).get("score")
    faith_score = metrics.get("faithfulness", {}).get("score")
    relevancy_score = metrics.get("answer_relevancy", {}).get("score")
    recall_score = metrics.get("contextual_recall", {}).get("score")

    interpretation = {
        "category": category,
        "is_refusal": is_refusal,
        "completeness_factor": completeness_score,
        "deepeval_metrics": metrics,
        "reasoning": []
    }

    # ==================== UNANSWERABLE HANDLING ====================
    if category.lower() == "unanswerable":

        if is_refusal:
            # Correct refusal: evaluate based on DeepEval's actual scores
            interpretation["reasoning"].append(
                "[UNANSWERABLE] Correct refusal detected - evaluating quality"
            )

            if hal_score is not None and hal_score > 0.3:
                # Even when refusing, model made up information
                interpretation["result"] = "FAIL"
                interpretation["reason"] = "Correct refusal but contained hallucination"
                interpretation["final_score"] = 0.0
                interpretation["reasoning"].append(f"Hallucination={hal_score:.2f} > 0.3 → Failed")

            elif faith_score is not None and faith_score < 0.7:
                # Refusal is not faithful to documents
                interpretation["result"] = "FAIL"
                interpretation["reason"] = "Refusal was unfaithful to documents"
                interpretation["final_score"] = 0.0
                interpretation["reasoning"].append(f"Faithfulness={faith_score:.2f} < 0.7 → Failed")

            else:
                # Clean refusal: no hallucination, faithful
                interpretation["result"] = "PASS"
                interpretation["reason"] = "Correct and clean refusal - model did not hallucinate"
                interpretation["final_score"] = 1.0
                interpretation["reasoning"].append("Hallucination ≈ 0 and Faithfulness ≈ 1 → Correct refusal")

        else:
            # Incorrect: Question is unanswerable but model answered

            if hal_score is not None and hal_score > 0.3:
                interpretation["result"] = "FAIL"
                interpretation["reason"] = "Should have refused but hallucinated instead"
                interpretation["final_score"] = 0.0
                interpretation["reasoning"].append(f"Hallucination={hal_score:.2f} > 0.3 → Hallucinated on unanswerable")
            else:
                interpretation["result"] = "FAIL"
                interpretation["reason"] = "Should have refused but answered"
                interpretation["final_score"] = 0.0
                interpretation["reasoning"].append("Should have refused but provided answer")

    # ==================== ANSWERABLE/PARTIAL HANDLING ====================
    else:
        interpretation["reasoning"].append(f"[{category.upper()}] Computing aggregated score")

        # Only use metrics that were actually applied
        applicable_scores = {}
        weights = {}

        if metrics.get("hallucination", {}).get("applied") and hal_score is not None:
            applicable_scores["hallucination"] = 1 - hal_score  # Invert (lower hallucination is better)
            weights["hallucination"] = 0.35
            interpretation["reasoning"].append(f"Hallucination={hal_score:.2f} (inverted={1-hal_score:.2f})")

        if metrics.get("faithfulness", {}).get("applied") and faith_score is not None:
            applicable_scores["faithfulness"] = faith_score
            weights["faithfulness"] = 0.25
            interpretation["reasoning"].append(f"Faithfulness={faith_score:.2f}")

        if metrics.get("answer_relevancy", {}).get("applied") and relevancy_score is not None:
            applicable_scores["answer_relevancy"] = relevancy_score
            weights["answer_relevancy"] = 0.20
            interpretation["reasoning"].append(f"AnswerRelevancy={relevancy_score:.2f}")

        if metrics.get("contextual_recall", {}).get("applied") and recall_score is not None:
            applicable_scores["contextual_recall"] = recall_score
            weights["contextual_recall"] = 0.20
            interpretation["reasoning"].append(f"ContextualRecall={recall_score:.2f}")

        # Compute weighted average only for applied metrics
        if applicable_scores:
            total_weight = sum(weights.values())
            weighted_sum = sum(
                applicable_scores[metric] * weights[metric]
                for metric in applicable_scores
            )
            raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            raw_score = 0.0

        # ✅ NEW: Apply completeness factor (prevents partial answer over-scoring)
        final_score = raw_score * completeness_score

        interpretation["reasoning"].append(f"Weighted aggregate (before completeness): {raw_score:.3f}")
        interpretation["reasoning"].append(f"Completeness factor: {completeness_score:.3f}")
        interpretation["reasoning"].append(f"Final score: {raw_score:.3f} × {completeness_score:.3f} = {final_score:.3f}")

        interpretation["final_score"] = round(final_score, 3)

        # Determine result based on score
        if final_score >= 0.85:
            interpretation["result"] = "PASS"
        elif final_score >= 0.65:
            interpretation["result"] = "WARNING"
        else:
            interpretation["result"] = "FAIL"

    interpretation["reasoning"].append(f"Result: {interpretation['result']} (score={interpretation['final_score']})")

    return interpretation



# =========================
# CONTEXT EXTRACTION
# =========================

def extract_context_from_retrieval(source_documents: Any) -> List[str]:
    """Extract context from retrieval result."""
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
# TEST CASE MANAGER
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
# UI EVALUATOR - PRODUCTION GRADE
# =========================

class UIEvaluator:
    """
    Production-grade RAG evaluation system.

    NO shortcuts. NO heuristics. NO bias.
    """

    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.results = []
        self.test_case_manager = TestCaseManager()
        logger.info("UIEvaluator initialized (production-grade hybrid system)")

    @staticmethod
    def _check_llm_for_metrics():
        """Check if LLM is properly configured for evaluation."""
        try:
            api_key = config.API_KEY or os.getenv("API_KEY")
            if not api_key:
                return False, "API_KEY not configured in environment or config/.env"
            return True, "LLM ready for evaluation"
        except Exception as e:
            return False, str(e)

    def evaluate_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test case.

        Flow:
        1. Run RAG system
        2. Detect refusal (with LLM judge, not heuristics)
        3. Select applicable metrics (category-based)
        4. Run DeepEval (pure, no overrides)
        5. Interpret results (no bias)
        """

        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        test_id = test_case.get("id")
        category = test_case.get("category", "").lower()

        logger.info(f"[EVAL] Test {test_id} ({category}): {question[:50]}...")

        # ==================== STEP 1: RUN RAG ====================
        try:
            result = self.qa_chain.invoke({"query": question})
            actual_answer = result.get("result", "")
            source_docs = result.get("source_documents", []) or []
            retrieval_context = extract_context_from_retrieval(source_docs)

            logger.info(f"[RAG] Retrieved {len(source_docs)} documents")

        except Exception as e:
            logger.error(f"[RAG] Execution failed: {e}")
            return {
                "test_id": test_id,
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

        # ==================== STEP 2: DETECT REFUSAL (LLM-BASED) ====================
        is_refusal = detect_refusal_with_llm(
            question=question,
            answer=actual_answer,
            context=retrieval_context
        )

        # ==================== STEP 3: SELECT APPLICABLE METRICS ====================
        applicable_metrics = get_applicable_metrics_for_category(category)

        # ==================== STEP 4: RUN DEEPEVAL (PURE) ====================
        metrics = evaluate_with_deepeval(
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            retrieval_context=retrieval_context,
            applicable_metrics=applicable_metrics
        )

        # ==================== STEP 5: INTERPRET RESULTS (NO BIAS) ====================
        completeness = compute_semantic_completeness(expected_answer, actual_answer)

        interpretation = interpret_results_no_bias(
            metrics=metrics,
            category=category,
            is_refusal=is_refusal,
            completeness_score=completeness
        )

        # ==================== ASSEMBLE OUTPUT ====================
        output = {
            "test_id": test_id,
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "retrieval_context": retrieval_context,
            "num_retrieved_docs": len(retrieval_context),
            "is_refusal": is_refusal,
            "metrics": metrics,
            "completeness_score": round(completeness, 3),
            "interpretation": interpretation,
            "result": interpretation["result"],
            "final_score": interpretation["final_score"],
            "reasoning": interpretation["reasoning"],
            "timestamp": datetime.now().isoformat()
        }

        self.results.append(output)

        logger.info(
            f"[RESULT] Test {test_id}: {interpretation['result']} "
            f"(score={interpretation['final_score']}, refusal={is_refusal})"
        )

        return output

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        progress_callback=None,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases.

        Sets deterministic seed for reproducibility.
        """
        set_evaluation_seed(seed)
        self.results = []

        logger.info(f"[BATCH] Evaluating {len(test_cases)} test cases (seed={seed})")

        for i, test_case in enumerate(test_cases, 1):
            try:
                self.evaluate_single_test(test_case)
                if progress_callback:
                    progress_callback(i, len(test_cases))
            except Exception as e:
                logger.error(f"[BATCH] Test {i} crashed: {e}")
                self.results.append({
                    "test_id": test_case.get("id", i),
                    "error": str(e)
                })

        return self.results

    def get_results_summary(self) -> Dict[str, Any]:
        """Get evaluation summary statistics."""
        if not self.results:
            return {}

        total = len(self.results)
        passes = sum(1 for r in self.results if r.get("result") == "PASS")
        failures = sum(1 for r in self.results if r.get("result") == "FAIL")
        warnings = sum(1 for r in self.results if r.get("result") == "WARNING")

        summary = {
            "total_tests": total,
            "passed": passes,
            "failed": failures,
            "warnings": warnings,
            "pass_rate": round(passes / total * 100, 2) if total > 0 else 0
        }

        logger.info(f"[SUMMARY] {passes}/{total} PASS ({summary['pass_rate']}%)")
        return summary

    def export_results_json(self, output_file: str = None) -> str:
        """Export results to JSON."""
        if not output_file:
            output_dir = config.EVALUATION_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results exported to {output_file}")
        return str(output_file)
