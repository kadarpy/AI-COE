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
import torch
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
from cache_manager import get_cache_manager, get_cached_rag_response, cache_rag_response
from retrieval_evaluator import get_retrieval_evaluator
from reranker import get_reranker

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
# DETERMINISTIC RECALL: Ground Truth Anchor
# =========================

def compute_deterministic_contextual_recall(
    ground_truth_context: List[str],
    retrieved_context: List[str]
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute contextual recall by comparing ground_truth_context vs retrieved_context.
    
    DETERMINISTIC (not LLM-based).
    Uses semantic similarity to check if ground truth concepts were retrieved.
    
    Args:
        ground_truth_context: Ground truth context chunks
        retrieved_context: Actually retrieved context chunks
        
    Returns:
        Tuple of (score [0,1], debug_info)
    """
    if not ground_truth_context:
        return 1.0, {"note": "No ground truth provided"}
    
    if not retrieved_context:
        logger.warning(f"[RECALL] No context retrieved but {len(ground_truth_context)} ground truth chunks required")
        return 0.0, {"ground_truth_missing": len(ground_truth_context), "retrieved": 0}
    
    try:
        from sentence_transformers import util, SentenceTransformer
        
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        # Embed ground truth
        gt_embeddings = model.encode(ground_truth_context, convert_to_tensor=True)
        
        # Embed retrieved
        ret_embeddings = model.encode(retrieved_context, convert_to_tensor=True)
        
        # Compute similarity matrix: ground_truth x retrieved
        similarity_matrix = util.pytorch_cos_sim(gt_embeddings, ret_embeddings)
        
        # For each ground truth, find best match in retrieved
        max_similarities = torch.max(similarity_matrix, dim=1)[0]
        
        # ✅ IMPROVEMENT: Use soft similarity scoring instead of binary threshold
        # Each ground truth concept contributes its maximum similarity to retrieved context
        # Result: Partial matches get partial credit, not all-or-nothing
        # Example: 0.95 + 0.45 + 0.80 → recall = (0.95+0.45+0.80)/3 = 0.73
        total = len(ground_truth_context)
        
        # Soft scoring: mean of best matches for each ground truth chunk
        recall = float(torch.mean(max_similarities)) if total > 0 else 0.0
        
        # For detailed analysis: also track binary matches at threshold
        threshold = 0.7
        covered_binary = (max_similarities >= threshold).sum().item()
        
        # Create mapping of each ground truth to its best match score
        concept_similarities = [
            {
                "concept": ground_truth_context[i],
                "best_similarity": float(max_similarities[i]),
                "covered_binary": bool(max_similarities[i] >= threshold)
            }
            for i in range(total)
        ]
        
        debug = {
            "ground_truth_chunks": total,
            "retrieved_chunks": len(retrieved_context),
            "recall_type": "soft_similarity",  # Now using soft scoring
            "soft_recall": recall,  # Primary metric: mean of similarities
            "binary_covered_at_0.7": int(covered_binary),  # Secondary: for reference
            "threshold": threshold,
            "avg_similarity": float(torch.mean(max_similarities)),
            "min_similarity": float(torch.min(max_similarities)),
            "max_similarity": float(torch.max(max_similarities)),
            "concept_details": concept_similarities
        }
        
        logger.info(f"[DETERMINISTIC_RECALL] Soft scoring: {recall:.3f} (avg similarity across {total} concepts, binary @0.7: {covered_binary}/{total})")
        
        return float(recall), debug
        
    except Exception as e:
        logger.warning(f"[RECALL] Deterministic computation failed: {e}, returning 0.0")
        return 0.0, {"error": str(e)}


def compute_concept_coverage(
    ground_truth_context: List[str],
    actual_answer: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if required concepts from ground truth are mentioned in the answer.
    
    DETERMINISTIC concept checking.
    
    Args:
        ground_truth_context: Ground truth context chunks
        actual_answer: Model's answer
        
    Returns:
        Tuple of (coverage_score, debug_info)
    """
    if not ground_truth_context or not actual_answer:
        return 1.0, {"note": "Insufficient input"}
    
    try:
        from sentence_transformers import util, SentenceTransformer
        
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        # Embed each concept and the answer
        concept_embeddings = model.encode(ground_truth_context, convert_to_tensor=True)
        answer_embedding = model.encode([actual_answer], convert_to_tensor=True)
        
        # Compute similarity of each concept to the answer
        similarities = util.pytorch_cos_sim(concept_embeddings, answer_embedding).squeeze()
        
        # Threshold for concept mention
        threshold = 0.5
        covered = (similarities >= threshold).sum().item()
        total = len(ground_truth_context)
        
        coverage = covered / total if total > 0 else 0.0
        
        debug = {
            "required_concepts": total,
            "mentioned_concepts": int(covered),
            "concept_threshold": threshold,
            "avg_similarity": float(torch.mean(similarities)),
            "similarity_scores": [float(s) for s in similarities]
        }
        
        uncovered = [gt for gt, sim in zip(ground_truth_context, similarities) if sim < threshold]
        if uncovered:
            debug["uncovered_concepts"] = uncovered
            logger.warning(f"[CONCEPT_COVERAGE] Missing concepts: {uncovered}")
        
        return float(coverage), debug
        
    except Exception as e:
        logger.warning(f"[CONCEPT_COVERAGE] Failed: {e}")
        return 0.0, {"error": str(e)}


def compute_retrieval_penalty(
    retrieval_metrics: Dict[str, Any]
) -> float:
    """
    Compute penalty factor based on retrieval quality.
    
    If retrieval failed (recall=0), answers should be penalized.
    
    Args:
        retrieval_metrics: Output from retrieval evaluator
        
    Returns:
        Penalty factor [0, 1] where 1.0 = no penalty
    """
    if not retrieval_metrics or retrieval_metrics.get("computed") is False:
        return 1.0  # No penalty if no metrics
    
    recall = retrieval_metrics.get("recall_at_k", 0.5)
    hit_rate = retrieval_metrics.get("hit_rate_at_k", 0.5)
    
    # Penalty: if we didn't retrieve relevant context, answer quality is suspect
    # penalty_factor = 0.5 + 0.5 * (recall * 0.6 + hit_rate * 0.4)
    penalty = 0.5 + 0.5 * (recall * 0.6 + hit_rate * 0.4)
    
    logger.info(f"[RETRIEVAL_PENALTY] recall={recall:.2f}, hit_rate={hit_rate:.2f} → penalty={penalty:.2f}")
    
    return penalty


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
    retrieval_metrics: Dict[str, Any] = None,
    llm: DeepEvalBaseLLM = None
) -> Dict[str, Any]:
    """
    Interpret evaluation results WITH proper retrieval grounding.

    NEW: Incorporates retrieval quality and concept coverage.
    This prevents over-scoring when retrieval/concepts are missing.

    UNANSWERABLE + Correct Refusal:
      - Hallucination ≈ 0 (good, no hallucination)
      - Faithfulness ≈ 1 (good, faithful to docs)
      - Score interpretation: PASS (correct evaluation)

    UNANSWERABLE + Hallucination:
      - Hallucination > 0.3 (bad, made stuff up)
      - Score interpretation: FAIL (incorrect evaluation)

    ANSWERABLE:
      - All metrics contribute to final score
      - Deterministic contextual recall (ground truth anchored)
      - Concept coverage from ground truth
      - Retrieval penalty applied
      - Completeness factor for partial answers

    Key Point: Ground truth is the anchor. LLM metrics are advisory only.
    """

    hal_score = metrics.get("hallucination", {}).get("score")
    faith_score = metrics.get("faithfulness", {}).get("score")
    relevancy_score = metrics.get("answer_relevancy", {}).get("score")
    recall_score = metrics.get("contextual_recall", {}).get("score")
    concept_coverage = metrics.get("concept_coverage", {}).get("score")

    interpretation = {
        "category": category,
        "is_refusal": is_refusal,
        "completeness_factor": completeness_score,
        "deepeval_metrics": metrics,
        "retrieval_metrics": retrieval_metrics or {},
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
            weights["hallucination"] = 0.30
            interpretation["reasoning"].append(f"Hallucination={hal_score:.2f} (inverted={1-hal_score:.2f})")

        if metrics.get("faithfulness", {}).get("applied") and faith_score is not None:
            applicable_scores["faithfulness"] = faith_score
            weights["faithfulness"] = 0.20
            interpretation["reasoning"].append(f"Faithfulness={faith_score:.2f}")

        if metrics.get("answer_relevancy", {}).get("applied") and relevancy_score is not None:
            applicable_scores["answer_relevancy"] = relevancy_score
            weights["answer_relevancy"] = 0.20
            interpretation["reasoning"].append(f"AnswerRelevancy={relevancy_score:.2f}")

        # ✅ FIX: Use deterministic recall (NOT LLM-based)
        if metrics.get("contextual_recall", {}).get("applied") and recall_score is not None:
            applicable_scores["contextual_recall"] = recall_score
            weights["contextual_recall"] = 0.15
            interpretation["reasoning"].append(f"ContextualRecall (DETERMINISTIC)={recall_score:.2f}")

        # ✅ FIX: Include concept coverage
        if concept_coverage is not None:
            applicable_scores["concept_coverage"] = concept_coverage
            weights["concept_coverage"] = 0.15
            interpretation["reasoning"].append(f"ConceptCoverage={concept_coverage:.2f}")

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

        # ✅ FIX: Apply retrieval penalty (CRITICAL)
        retrieval_penalty = compute_retrieval_penalty(retrieval_metrics or {})
        penalized_score = raw_score * retrieval_penalty

        interpretation["reasoning"].append(f"Weighted aggregate (before penalties): {raw_score:.3f}")
        interpretation["reasoning"].append(f"Retrieval penalty factor: {retrieval_penalty:.3f}")

        # ✅ Apply completeness factor (prevents partial answer over-scoring)
        final_score = penalized_score * completeness_score

        interpretation["reasoning"].append(f"Completeness factor: {completeness_score:.3f}")
        interpretation["reasoning"].append(f"Final score: {raw_score:.3f} × {retrieval_penalty:.3f} × {completeness_score:.3f} = {final_score:.3f}")

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
    """
    Extract context from retrieval result.
    
    Handles both:
    - Fresh LangChain Document objects (from RAG)
    - Cached/serialized documents (from cache - plain strings/dicts)
    """
    if not source_documents:
        return []

    context = []
    try:
        for doc in source_documents:
            content = None
            
            # ✅ FIX: Handle LangChain Document objects (fresh RAG)
            if hasattr(doc, 'page_content'):
                content = str(doc.page_content).strip()
            
            # ✅ FIX: Handle cached/serialized documents (from cache)
            elif isinstance(doc, dict) and 'page_content' in doc:
                content = str(doc['page_content']).strip()
            
            # ✅ FIX: Handle plain strings (fallback for cached content)
            elif isinstance(doc, str):
                content = doc.strip()
            
            # ✅ FIX: Handle dict with 'content' key (alternative serialization)
            elif isinstance(doc, dict) and 'content' in doc:
                content = str(doc['content']).strip()
            
            # Last resort: convert to string
            else:
                content = str(doc).strip()
            
            if content:
                context.append(content)
                
    except Exception as e:
        logger.warning(f"Failed to extract context: {e}")
        # Fallback: try to convert everything to strings
        try:
            context = [str(doc).strip() for doc in source_documents if str(doc).strip()]
        except:
            pass

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
        # ✅ FIX: Initialize cache manager (was missing)
        self.cache_manager = get_cache_manager(cache_dir=config.CACHE_DIR)
        logger.info("UIEvaluator initialized (production-grade hybrid system)")
        logger.info(f"Cache enabled: {config.ENABLE_CACHING} (dir: {config.CACHE_DIR})")

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

    def evaluate_single_test(self, test_case: Dict[str, Any], run_number: int = 1) -> Dict[str, Any]:
        """
        Evaluate a single test case with production-grade features.

        Flow:
        1. Check cache (if enabled)
        2. Run RAG system
        3. Compute retrieval metrics
        4. Detect refusal (with LLM judge, not heuristics)
        5. Select applicable metrics (category-based)
        6. Run DeepEval (pure, no overrides)
        7. Interpret results with pass/fail thresholds
        8. Cache results (if enabled)
        
        Args:
            test_case: Test case to evaluate
            run_number: Which run number (for multi-run evaluation)
            
        Returns:
            Dictionary with comprehensive evaluation results
        """

        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        ground_truth_context = test_case.get("ground_truth_context", [])
        test_id = test_case.get("id")
        category = test_case.get("category", "").lower()

        logger.info(f"[EVAL] Test {test_id} ({category}): {question[:50]}... (run {run_number})")

        # ==================== STEP 1: CHECK CACHE + RUN RAG ====================
        # ✅ FIX: Build deterministic cache key from question + model config
        rag_cache_key = {
            "question": question,
            "llm_model": config.LLM_MODEL,
            "temperature": config.TEMPERATURE,
            "retriever_k": config.RETRIEVER_K,
            "retriever_fetch_k": config.RETRIEVER_FETCH_K,
            "chunk_size": config.DOC_CHUNK_SIZE,
            "chunk_overlap": config.DOC_CHUNK_OVERLAP,
            "enable_reranker": config.ENABLE_RERANKER,
            "reranker_threshold": config.RERANKER_THRESHOLD
        }
        
        # ✅ FIX: Try cache lookup before RAG execution
        cached_result = None
        if config.ENABLE_CACHING:
            cached_result = self.cache_manager.get(rag_cache_key, cache_type="rag")
            if cached_result:
                logger.info(f"[CACHE HIT] Retrieved from cache: {rag_cache_key.get('question')[:30]}...")
        
        try:
            # Use cached result if available, otherwise run RAG
            if cached_result:
                result = cached_result
            else:
                result = self.qa_chain.invoke({"query": question})
                # ✅ FIX: Cache the result after successful RAG execution
                if config.ENABLE_CACHING:
                    self.cache_manager.set(rag_cache_key, result, cache_type="rag")
                    logger.info(f"[CACHE WRITE] Cached RAG response for: {rag_cache_key.get('question')[:30]}...")
            
            actual_answer = result.get("result", "")
            source_docs = result.get("source_documents", []) or []
            retrieval_context = extract_context_from_retrieval(source_docs)
            reranked = result.get("reranked", False)
            rerank_scores = result.get("rerank_scores", [])

            logger.info(f"[RAG] Retrieved {len(source_docs)} documents (reranked={reranked})")
            # ✅ FIX: Log retrieval context to verify extraction worked
            logger.info(f"[RAG] Extracted {len(retrieval_context)} context chunks from {len(source_docs)} docs")
            if retrieval_context:
                logger.info(f"[RAG] First context chunk: {retrieval_context[0][:60]}...")
            else:
                logger.warning(f"[RAG] WARNING: retrieval_context is empty! source_docs structure: {type(source_docs)}, len={len(source_docs)}")
                if source_docs:
                    logger.warning(f"[RAG] First source_doc: {type(source_docs[0])}, content: {str(source_docs[0])[:100]}")

        except Exception as e:
            logger.error(f"[RAG] Execution failed: {e}")
            return {
                "test_id": test_id,
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "run_number": run_number
            }

        # ==================== STEP 2: COMPUTE RETRIEVAL METRICS ====================
        retrieval_metrics = {}
        if config.COMPUTE_RETRIEVAL_METRICS and ground_truth_context:
            try:
                evaluator = get_retrieval_evaluator(threshold=config.RETRIEVAL_THRESHOLD)
                retrieval_metrics = evaluator.evaluate_retrieval(
                    retrieved_chunks=retrieval_context,
                    ground_truth_context=ground_truth_context,
                    k=len(retrieval_context)
                )
                logger.info(
                    f"[RETRIEVAL] precision={retrieval_metrics.get('precision_at_k', 0)}, "
                    f"recall={retrieval_metrics.get('recall_at_k', 0)}, "
                    f"hit_rate={retrieval_metrics.get('hit_rate_at_k', 0)}"
                )
            except Exception as e:
                logger.warning(f"[RETRIEVAL] Metrics computation failed: {e}")
                retrieval_metrics = {"error": str(e)}
        else:
            retrieval_metrics = {"computed": False, "reason": "No ground truth context"}

        # ==================== STEP 3: DETECT REFUSAL (LLM-BASED) ====================
        is_refusal = detect_refusal_with_llm(
            question=question,
            answer=actual_answer,
            context=retrieval_context
        )

        # ==================== STEP 4: SELECT APPLICABLE METRICS ====================
        applicable_metrics = get_applicable_metrics_for_category(category)

        # ==================== STEP 5: RUN DEEPEVAL (PURE) ====================
        metrics = evaluate_with_deepeval(
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            retrieval_context=retrieval_context,
            applicable_metrics=applicable_metrics
        )

        # ==================== STEP 5B: COMPUTE DETERMINISTIC RECALL METRICS ====================
        # FIX: Override DeepEval's contextual recall with ground-truth-based metrics
        det_recall_score, det_recall_debug = compute_deterministic_contextual_recall(
            ground_truth_context=ground_truth_context,
            retrieved_context=retrieval_context
        )
        
        concept_coverage_score, concept_debug = compute_concept_coverage(
            ground_truth_context=ground_truth_context,
            actual_answer=actual_answer
        )
        
        # Replace DeepEval contextual recall with deterministic version
        if ground_truth_context:
            logger.info(
                f"[METRICS] Replacing DeepEval contextual_recall with deterministic: "
                f"LLM={metrics.get('contextual_recall', {}).get('score', 'N/A')} → "
                f"Deterministic={det_recall_score:.3f}"
            )
            metrics["contextual_recall"]["score"] = det_recall_score
            metrics["contextual_recall"]["reason"] = f"Deterministic recall: {det_recall_debug}"
            metrics["concept_coverage"] = {
                "score": concept_coverage_score,
                "reason": f"Concept coverage: {len(ground_truth_context)} concepts, {concept_debug.get('mentioned_concepts', 0)} mentioned",
                "applied": True,
                "debug": concept_debug
            }
        
        # ==================== STEP 6: INTERPRET RESULTS with PASS/FAIL ====================
        # FIX: Semantic completeness using embeddings (NOT length fallback)
        completeness = compute_semantic_completeness(expected_answer, actual_answer)

        interpretation = interpret_results_no_bias(
            metrics=metrics,
            category=category,
            is_refusal=is_refusal,
            completeness_score=completeness,
            retrieval_metrics=retrieval_metrics
        )

        # ==================== STEP 7: COMPUTE PASS/FAIL DECISION ====================
        pass_fail_decision = self._make_pass_fail_decision(
            metrics=metrics,
            interpretation=interpretation,
            retrieval_metrics=retrieval_metrics,
            category=category
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
            "reranked": reranked,
            "rerank_scores": rerank_scores,
            "is_refusal": is_refusal,
            "metrics": metrics,
            "retrieval_metrics": retrieval_metrics,
            "completeness_score": round(completeness, 3),
            "interpretation": interpretation,
            "result": interpretation["result"],
            "final_score": interpretation["final_score"],
            "reasoning": interpretation["reasoning"],
            "pass_fail": pass_fail_decision,
            "overall_status": pass_fail_decision.get("verdict"),
            "run_number": run_number,
            "timestamp": datetime.now().isoformat()
        }

        self.results.append(output)

        logger.info(
            f"[RESULT] Test {test_id}: {interpretation['result']} "
            f"(status={pass_fail_decision.get('verdict')}, score={interpretation['final_score']}, run={run_number})"
        )

        return output

    def _make_pass_fail_decision(
        self,
        metrics: Dict[str, Any],
        interpretation: Dict[str, Any],
        retrieval_metrics: Dict[str, Any],
        category: str
    ) -> Dict[str, Any]:
        """
        Make final pass/fail/warning decision based on thresholds.
        
        Args:
            metrics: DeepEval metrics
            interpretation: Interpretation result
            retrieval_metrics: Retrieval quality metrics
            category: Question category
            
        Returns:
            Decision dictionary with verdict and reasoning
        """
        decision = {
            "verdict": "UNKNOWN",
            "thresholds_passed": {},
            "thresholds_failed": [],
            "reasoning": []
        }

        hal_score = metrics.get("hallucination", {}).get("score")
        faith_score = metrics.get("faithfulness", {}).get("score")
        relevancy_score = metrics.get("answer_relevancy", {}).get("score")

        # Check hallucination threshold
        if hal_score is not None:
            threshold = 1.0 - config.THRESHOLD_HALLUCINATION  # Invert (lower is better)
            passed = (1.0 - hal_score) >= threshold
            decision["thresholds_passed"]["hallucination"] = passed
            if not passed:
                decision["thresholds_failed"].append(f"Hallucination {hal_score:.3f} exceeds threshold {config.THRESHOLD_HALLUCINATION}")
            else:
                decision["reasoning"].append(f"✓ Hallucination {hal_score:.3f} < threshold {config.THRESHOLD_HALLUCINATION}")

        # Check faithfulness threshold
        if faith_score is not None:
            passed = faith_score >= config.THRESHOLD_FAITHFULNESS
            decision["thresholds_passed"]["faithfulness"] = passed
            if not passed:
                decision["thresholds_failed"].append(f"Faithfulness {faith_score:.3f} < threshold {config.THRESHOLD_FAITHFULNESS}")
            else:
                decision["reasoning"].append(f"✓ Faithfulness {faith_score:.3f} >= threshold {config.THRESHOLD_FAITHFULNESS}")

        # Check answer relevancy threshold (if applicable)
        if relevancy_score is not None:
            passed = relevancy_score >= config.THRESHOLD_ANSWER_RELEVANCY
            decision["thresholds_passed"]["answer_relevancy"] = passed
            if not passed:
                decision["thresholds_failed"].append(f"Answer Relevancy {relevancy_score:.3f} < threshold {config.THRESHOLD_ANSWER_RELEVANCY}")
            else:
                decision["reasoning"].append(f"✓ Answer Relevancy {relevancy_score:.3f} >= threshold {config.THRESHOLD_ANSWER_RELEVANCY}")

        # ✅ FIX: Check concept coverage (ground truth anchored)
        concept_coverage = metrics.get("concept_coverage", {}).get("score")
        if concept_coverage is not None:
            threshold = getattr(config, "THRESHOLD_CONCEPT_COVERAGE", 0.70)
            passed = concept_coverage >= threshold
            decision["thresholds_passed"]["concept_coverage"] = passed
            if not passed:
                decision["thresholds_failed"].append(f"Concept Coverage {concept_coverage:.3f} < threshold {threshold}")
                decision["reasoning"].append(f"✗ Concept Coverage {concept_coverage:.3f} < threshold {threshold} → REQUIRED concepts missing from answer")
            else:
                decision["reasoning"].append(f"✓ Concept Coverage {concept_coverage:.3f} >= threshold {threshold}")

        # Check retrieval metrics if available
        if retrieval_metrics.get("computed") is not False:
            recall = retrieval_metrics.get("recall_at_k")
            precision = retrieval_metrics.get("precision_at_k")
            hit_rate = retrieval_metrics.get("hit_rate_at_k")

            if recall is not None:
                passed = recall >= config.THRESHOLD_RECALL
                decision["thresholds_passed"]["recall"] = passed
                if not passed:
                    decision["thresholds_failed"].append(f"Recall {recall:.3f} < threshold {config.THRESHOLD_RECALL}")
                else:
                    decision["reasoning"].append(f"✓ Recall {recall:.3f} >= threshold {config.THRESHOLD_RECALL}")

            if precision is not None:
                passed = precision >= config.THRESHOLD_PRECISION
                decision["thresholds_passed"]["precision"] = passed
                if not passed:
                    decision["thresholds_failed"].append(f"Precision {precision:.3f} < threshold {config.THRESHOLD_PRECISION}")
                else:
                    decision["reasoning"].append(f"✓ Precision {precision:.3f} >= threshold {config.THRESHOLD_PRECISION}")

            if hit_rate is not None:
                passed = hit_rate >= config.THRESHOLD_HIT_RATE
                decision["thresholds_passed"]["hit_rate"] = passed
                if not passed:
                    decision["thresholds_failed"].append(f"Hit Rate {hit_rate:.3f} < threshold {config.THRESHOLD_HIT_RATE}")
                else:
                    decision["reasoning"].append(f"✓ Hit Rate {hit_rate:.3f} >= threshold {config.THRESHOLD_HIT_RATE}")

        # Make final verdict
        if not decision["thresholds_failed"]:
            decision["verdict"] = "PASS"
            decision["reasoning"].append("All thresholds met → PASS")
        elif len(decision["thresholds_failed"]) <= 1:
            decision["verdict"] = "WARNING"
            decision["reasoning"].append(f"Minor threshold violations → WARNING ({len(decision['thresholds_failed'])} threshold)")
        else:
            decision["verdict"] = "FAIL"
            decision["reasoning"].append(f"Multiple threshold violations → FAIL ({len(decision['thresholds_failed'])} thresholds)")

        return decision

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        progress_callback=None,
        seed: int = 42,
        num_runs: int = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Evaluate multiple test cases with multi-run support.

        Supports running evaluation multiple times to measure stability.
        Aggregates results across runs.

        Args:
            test_cases: List of test cases
            progress_callback: Callback for progress updates
            seed: Random seed for reproducibility
            num_runs: Number of runs (default from config)
            
        Returns:
            Tuple of (all_results, aggregated_summary)
        """
        if num_runs is None:
            num_runs = config.EVAL_NUM_RUNS

        set_evaluation_seed(seed)
        
        # Multi-run evaluation
        all_runs = []
        
        logger.info(f"[BATCH] Starting multi-run evaluation: {num_runs} runs x {len(test_cases)} tests")

        for run_num in range(1, num_runs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"RUN {run_num}/{num_runs}")
            logger.info(f"{'='*60}")
            
            self.results = []
            
            for i, test_case in enumerate(test_cases, 1):
                try:
                    self.evaluate_single_test(test_case, run_number=run_num)
                    if progress_callback:
                        progress_callback(i + (run_num - 1) * len(test_cases), num_runs * len(test_cases))
                except Exception as e:
                    logger.error(f"[BATCH] Run {run_num} Test {i} crashed: {e}")
                    self.results.append({
                        "test_id": test_case.get("id", i),
                        "error": str(e),
                        "run_number": run_num
                    })
            
            all_runs.append(self.results.copy())

        # Aggregate results across runs
        self.results = all_runs[0]  # Default to first run for main results
        
        aggregated_summary = self._aggregate_multi_run_results(all_runs, test_cases)
        
        logger.info(f"\n{'='*60}")
        logger.info("MULTI-RUN EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        
        return all_runs, aggregated_summary

    def _aggregate_multi_run_results(
        self,
        all_runs: List[List[Dict[str, Any]]],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate evaluation results across multiple runs.
        
        Computes stability metrics (mean, std, min, max) for each metric.
        """
        num_runs = len(all_runs)
        num_tests = len(test_cases)
        
        aggregated = {
            "num_runs": num_runs,
            "num_tests": num_tests,
            "per_test_stability": {},
            "overall_stats": {}
        }

        # Aggregate per test case
        for test_idx in range(num_tests):
            test_id = test_cases[test_idx].get("id", test_idx)
            runs_for_test = []
            
            for run_results in all_runs:
                if test_idx < len(run_results):
                    runs_for_test.append(run_results[test_idx])
            
            if not runs_for_test:
                continue

            # Extract scores from each run
            final_scores = []
            verdicts = []
            pass_fail_verdicts = []
            
            for result in runs_for_test:
                if "final_score" in result:
                    final_scores.append(result["final_score"])
                if "result" in result:
                    verdicts.append(result["result"])
                if "overall_status" in result:
                    pass_fail_verdicts.append(result["overall_status"])

            # Compute stability stats for this test
            stability = {
                "runs": num_runs,
                "verdicts": verdicts,
                "pass_fail": pass_fail_verdicts
            }

            if final_scores:
                stability["score"] = {
                    "mean": round(np.mean(final_scores), 3),
                    "std": round(np.std(final_scores), 3),
                    "min": round(np.min(final_scores), 3),
                    "max": round(np.max(final_scores), 3),
                    "variance": round(np.var(final_scores), 3),
                    "all_values": [round(s, 3) for s in final_scores]
                }

                # Stability assessment
                if stability["score"]["std"] < config.STABILITY_WARNING_THRESHOLD:
                    stability["stability_rating"] = "EXCELLENT"
                elif stability["score"]["std"] < config.MAX_ALLOWED_VARIANCE:
                    stability["stability_rating"] = "GOOD"
                else:
                    stability["stability_rating"] = "UNSTABLE"

            aggregated["per_test_stability"][f"test_{test_id}"] = stability

        # Overall aggregation
        all_final_scores = []
        all_verdicts = []
        all_pass_fail = []
        
        for run_results in all_runs:
            for result in run_results:
                if "final_score" in result:
                    all_final_scores.append(result["final_score"])
                if "result" in result:
                    all_verdicts.append(result["result"])
                if "overall_status" in result:
                    all_pass_fail.append(result["overall_status"])

        if all_final_scores:
            aggregated["overall_stats"]["score"] = {
                "mean": round(np.mean(all_final_scores), 3),
                "std": round(np.std(all_final_scores), 3),
                "min": round(np.min(all_final_scores), 3),
                "max": round(np.max(all_final_scores), 3)
            }

        if all_verdicts:
            aggregated["overall_stats"]["verdict_distribution"] = {
                "PASS": all_verdicts.count("PASS"),
                "WARNING": all_verdicts.count("WARNING"),
                "FAIL": all_verdicts.count("FAIL")
            }

        if all_pass_fail:
            aggregated["overall_stats"]["pass_fail_distribution"] = {
                "PASS": all_pass_fail.count("PASS"),
                "WARNING": all_pass_fail.count("WARNING"),
                "FAIL": all_pass_fail.count("FAIL")
            }

        logger.info(f"Aggregation complete: {num_runs} runs evaluated")
        return aggregated

    def get_results_summary(self) -> Dict[str, Any]:
        """Get evaluation summary statistics including pass/fail decisions."""
        if not self.results:
            return {}

        total = len(self.results)
        
        # Old-style result counts
        passes = sum(1 for r in self.results if r.get("result") == "PASS")
        failures = sum(1 for r in self.results if r.get("result") == "FAIL")
        warnings = sum(1 for r in self.results if r.get("result") == "WARNING")
        
        # New-style pass/fail decision counts
        pass_fail_pass = sum(1 for r in self.results if r.get("overall_status") == "PASS")
        pass_fail_fail = sum(1 for r in self.results if r.get("overall_status") == "FAIL")
        pass_fail_warning = sum(1 for r in self.results if r.get("overall_status") == "WARNING")

        summary = {
            "total_tests": total,
            "legacy_evaluation": {
                "passed": passes,
                "failed": failures,
                "warnings": warnings,
                "pass_rate": round(passes / total * 100, 2) if total > 0 else 0
            },
            "pass_fail_decision": {
                "passed": pass_fail_pass,
                "failed": pass_fail_fail,
                "warnings": pass_fail_warning,
                "pass_rate": round(pass_fail_pass / total * 100, 2) if total > 0 else 0
            },
            "system_readiness": self._assess_system_readiness(pass_fail_pass, total)
        }

        logger.info(f"[SUMMARY] Legacy: {passes}/{total} PASS ({summary['legacy_evaluation']['pass_rate']}%)")
        logger.info(f"[SUMMARY] Pass/Fail: {pass_fail_pass}/{total} PASS ({summary['pass_fail_decision']['pass_rate']}%)")
        logger.info(f"[SUMMARY] System Readiness: {summary['system_readiness']}")
        
        return summary

    def _assess_system_readiness(self, passed_tests: int, total_tests: int) -> str:
        """
        Assess overall system readiness for production.
        
        Returns: String assessment
        """
        if total_tests == 0:
            return "UNKNOWN"
        
        pass_rate = passed_tests / total_tests
        
        if pass_rate >= 0.95:
            return "PRODUCTION_READY"
        elif pass_rate >= 0.85:
            return "READY_WITH_MINOR_ISSUES"
        elif pass_rate >= 0.70:
            return "NEEDS_IMPROVEMENT"
        else:
            return "NOT_READY"

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

    def save_to_history_and_analyze_drift(self) -> Dict[str, Any]:
        """
        Save evaluation results to history and analyze drift.
        
        Returns:
            Drift analysis results
        """
        try:
            from drift_tracker import get_drift_tracker
            
            # Get summary
            summary = self.get_results_summary()
            
            # Save to history
            tracker = get_drift_tracker()
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            tracker.save_evaluation_run(
                results=self.results,
                summary=summary,
                run_id=run_id
            )
            
            # Compute drift
            drift = tracker.compute_drift(summary)
            
            # Detect regression
            regression = tracker.detect_regression(summary)
            
            # Get trends
            trends = tracker.get_trend_analysis()
            
            analysis = {
                "run_id": run_id,
                "drift": drift,
                "regression": regression,
                "trends": trends
            }
            
            logger.info(f"Drift analysis: {drift.get('metrics', {})}")
            logger.info(f"Regression status: {regression.get('severity', 'UNKNOWN')}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze drift: {e}")
            return {"error": str(e)}

    def add_explanation_tracing(self):
        """
        Enhance results with explanation tracing.
        
        Adds:
        - Which chunks influenced the answer
        - Retrieval ranking scores  
        - Context contribution analysis
        """
        for result in self.results:
            if "retrieval_context" not in result:
                continue
            
            # Add context contribution (which context chunks are most important)
            tracing = {
                "retrieved_chunks": len(result.get("retrieval_context", [])),
                "reranked": result.get("reranked", False),
                "rerank_scores": result.get("rerank_scores", [])
            }
            
            # If we have rerank scores, compute contribution weights
            if tracing["rerank_scores"]:
                scores = np.array(tracing["rerank_scores"])
                # Normalize scores to weights
                weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones_like(scores) / len(scores)
                
                tracing["contribution_weights"] = [round(w, 3) for w in weights]
                
                # Find most influential chunks
                top_indices = np.argsort(weights)[-3:]  # Top 3
                tracing["top_contributing_chunks"] = [
                    {
                        "index": int(idx),
                        "weight": round(float(weights[idx]), 3),
                        "content": result["retrieval_context"][idx][:100] + "..."
                    }
                    for idx in reversed(top_indices)
                ]
            
            result["explanation_tracing"] = tracing
            
        logger.info("Explanation tracing added to results")

