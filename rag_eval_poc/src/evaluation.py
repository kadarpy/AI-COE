"""
HYBRID EVALUATION PIPELINE (DeepEval + ML Evaluator)
======================================================
Combines LLM-based judge (DeepEval) with deterministic ML scoring (CrossEncoder).

Metrics:

DeepEval (LLM Judge):
- HallucinationMetric
- FaithfulnessMetric
- AnswerRelevancyMetric
- ContextualRecallMetric

ML Evaluator (CrossEncoder):
- Semantic Relevance (answer-question relevance)
- Context Overlap (answer-context alignment)
- Confidence Score (combined metric)

Returns hybrid scoring with training data collection for model improvement.
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
from functools import lru_cache

from dotenv import load_dotenv

# Initialize logger BEFORE any package imports or usage
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables FIRST, before any package imports
config_path = Path(__file__).parent.parent / "config" / ".env"
if config_path.exists():
    load_dotenv(config_path, override=True)
else:
    load_dotenv(override=True)

# DO NOT set dummy values for environment variables - let packages handle missing credentials gracefully
# Remove: os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
# Remove: os.environ["CONFIDENT_DISABLE_TELEMETRY"] = "1"
# Remove: os.environ["CONFIDENT_API_KEY"] = "dummy_key"

# Configure DeepEval to be quiet (suppress logging from external packages)
os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["CONFIDENT_DISABLE_TELEMETRY"] = "1"

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    HallucinationMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric
)

from config import config

# ===========================
# IMPORT ML EVALUATOR
# ===========================
from ml_evaluator import get_ml_evaluator

# ===========================
# IMPORT MLOPS COMPONENTS (Phase 2)
# ===========================
try:
    from mlops.mlflow_tracker import get_mlflow_tracker
    from mlops.evaluator_model import TrainedEvaluator
    from mlops.thresholds import ThresholdEngine, ThresholdConfig, get_threshold_engine
    MLOPS_AVAILABLE = True
except ImportError:
    logger.warning("MLOps components not available. Running without MLflow/advanced features.")
    MLOPS_AVAILABLE = False

# Import retrieval metrics
try:
    from evaluation.retrieval_metrics import RetrievalMetrics
    RETRIEVAL_METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Retrieval metrics module not available.")
    RETRIEVAL_METRICS_AVAILABLE = False

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
    """
    Get LLM instance for DeepEval metrics evaluation.
    
    Returns:
        LLM instance for DeepEval
        
    Raises:
        ValueError: If LLM provider is not properly configured
    """
    from config import config
    import os

    provider = config.LLM_PROVIDER.lower()
    
    # Handle Groq provider (most common)
    if provider == "groq":
        api_key = config.API_KEY or os.getenv("API_KEY")
        model = config.LLM_MODEL or os.getenv("LLM_MODEL")
        
        if not api_key:
            raise ValueError("API_KEY not configured for Groq provider. Set API_KEY in .env or environment variable")
        if not model:
            raise ValueError("LLM_MODEL not configured for Groq provider. Set LLM_MODEL in .env or environment variable")
        
        logger.info(f"Initializing DeepEval LLM with Groq provider (model: {model})")
        return GroqDeepEvalLLM(api_key=api_key, model_name=model)
    
    # Handle other providers - they would need their own adapter classes
    # For now, only Groq is fully supported for DeepEval
    else:
        raise ValueError(f"DeepEval LLM evaluation currently only supports 'groq' provider. "
                        f"Set LLM_PROVIDER=groq in .env. Current provider: {provider}")


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
# FAILURE ANALYSIS FUNCTION
# =========================

def analyze_failure(
    question: str,
    expected_answer: str,
    actual_answer: str,
    metrics: Dict[str, Any],
    retrieval_context: List[str]
) -> Dict[str, Any]:
    """
    Analyze what type of failure occurred based on metrics.
    
    Args:
        question: Input question
        expected_answer: Expected/ground truth answer
        actual_answer: Actual answer generated by system
        metrics: Dictionary of metric scores
        retrieval_context: Retrieved context documents
        
    Returns:
        Dictionary with failure analysis:
        {
            "failure_type": "correct|hallucination|wrong_answer|context_miss|unanswerable_answered",
            "reasoning": str,
            "metrics_analysis": dict
        }
    """
    metrics_analysis = {}
    
    # Extract metric scores
    hallucination_score = metrics.get("Hallucination", {}).get("score")
    faithfulness_score = metrics.get("Faithfulness", {}).get("score")
    relevance_score = metrics.get("AnswerRelevancy", {}).get("score")
    recall_score = metrics.get("ContextualRecall", {}).get("score")
    
    metrics_analysis = {
        "hallucination": hallucination_score,
        "faithfulness": faithfulness_score,
        "relevance": relevance_score,
        "recall": recall_score
    }
    
    # Check if answer is correct (exact match or very similar)
    is_exact_match = actual_answer.strip().lower() == expected_answer.strip().lower()
    expected_is_negative = "does not contain" in expected_answer.lower() or "i don't know" in expected_answer.lower()
    actual_is_negative = "does not contain" in actual_answer.lower() or "i don't know" in actual_answer.lower()
    
    # If both are "I don't know" or "not in documents", it's correct
    if expected_is_negative and actual_is_negative:
        return {
            "failure_type": "correct",
            "reasoning": "Correctly handled unanswerable question",
            "metrics_analysis": metrics_analysis
        }
    
    # If answer matches expected, it's correct
    if is_exact_match:
        return {
            "failure_type": "correct",
            "reasoning": "Answer matches expected output",
            "metrics_analysis": metrics_analysis
        }
    
    # Analyze failure type based on metrics
    # HIGH hallucination score = fabricating information
    if hallucination_score is not None and hallucination_score > 0.7:
        return {
            "failure_type": "hallucination",
            "reasoning": f"Hallucination score {hallucination_score:.3f} indicates fabricated information",
            "metrics_analysis": metrics_analysis
        }
    
    # LOW faithfulness = answer contradicts or goes beyond context
    if faithfulness_score is not None and faithfulness_score < 0.5:
        return {
            "failure_type": "unfaithful",
            "reasoning": f"Faithfulness score {faithfulness_score:.3f} indicates answer not grounded in context",
            "metrics_analysis": metrics_analysis
        }
    
    # LOW relevance = answer doesn't address the question
    if relevance_score is not None and relevance_score < 0.5:
        return {
            "failure_type": "irrelevant",
            "reasoning": f"Relevance score {relevance_score:.3f} indicates answer doesn't address question",
            "metrics_analysis": metrics_analysis
        }
    
    # LOW recall = answer misses important context
    if recall_score is not None and recall_score < 0.5:
        return {
            "failure_type": "incomplete",
            "reasoning": f"Recall score {recall_score:.3f} indicates missed context",
            "metrics_analysis": metrics_analysis
        }
    
    # If unanswerable question was answered with content
    if expected_is_negative and not actual_is_negative:
        return {
            "failure_type": "hallucination",
            "reasoning": "Answered unanswerable question with fabricated information",
            "metrics_analysis": metrics_analysis
        }
    
    # Default: wrong answer
    return {
        "failure_type": "wrong_answer",
        "reasoning": "Answer is incorrect but doesn't fit other failure categories",
        "metrics_analysis": metrics_analysis
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
    Evaluate RAG test cases with hybrid scoring:
    1. DeepEval (LLM-based judge)
    2. ML Evaluator (CrossEncoder for semantic relevance)
    
    Collects training data for future model improvements.
    Tracks both successful evaluations and ML failures.
    """
    
    # Memory management: cap results to prevent unbounded growth
    MAX_CACHED_RESULTS = 1000

    def __init__(self, qa_chain):
        """
        Args:
            qa_chain: LangChain RAG chain
        """
        self.qa_chain = qa_chain
        self.results = []
        self.test_case_manager = TestCaseManager()
        
        # Track ML evaluator failures for monitoring
        self.ml_failures = 0
        self.deepeval_failures = 0
        
        # Use singleton ML evaluator (reuses model across evaluations)
        try:
            self.ml_evaluator = get_ml_evaluator()
            logger.info("ML Evaluator (singleton) initialized successfully")
        except Exception as e:
            logger.warning(f"ML Evaluator initialization failed: {e}. Continuing without ML metrics.")
            self.ml_evaluator = None
        
        # Initialize Phase 2 components (optional)
        self.mlflow_tracker = None
        self.trained_evaluator = None
        self.threshold_engine = None
        self.retrieval_metrics = None
        
        if MLOPS_AVAILABLE and config.ENABLE_MLFLOW:
            try:
                self.mlflow_tracker = get_mlflow_tracker(
                    tracking_uri=str(config.MLFLOW_TRACKING_DIR)
                )
                logger.info("MLflow tracker initialized")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}")
        
        if MLOPS_AVAILABLE and config.ENABLE_TRAINED_EVAL:
            try:
                self.trained_evaluator = TrainedEvaluator(model_dir=str(config.MODEL_DIR))
                if self.trained_evaluator.is_available():
                    logger.info(f"Trained evaluator initialized with {len(self.trained_evaluator.available_models)} models")
                else:
                    logger.info("No trained models available yet")
            except Exception as e:
                logger.warning(f"Trained evaluator initialization failed: {e}")
        
        if MLOPS_AVAILABLE and config.ENABLE_THRESHOLDS:
            try:
                threshold_config = ThresholdConfig(
                    relevance=config.THRESHOLD_RELEVANCE,
                    faithfulness=config.THRESHOLD_FAITHFULNESS,
                    hallucination=config.THRESHOLD_HALLUCINATION
                )
                self.threshold_engine = ThresholdEngine(threshold_config)
                logger.info("Threshold engine initialized")
            except Exception as e:
                logger.warning(f"Threshold engine initialization failed: {e}")
        
        if RETRIEVAL_METRICS_AVAILABLE:
            try:
                self.retrieval_metrics = RetrievalMetrics(k=config.RETRIEVER_K_FOR_METRICS)
                logger.info("Retrieval metrics initialized")
            except Exception as e:
                logger.warning(f"Retrieval metrics initialization failed: {e}")
        
        logger.info("UIEvaluator initialized")

    def evaluate_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate single test case with hybrid scoring.
        
        Args:
            test_case: Dict with question, expected_answer, labels (optional), etc.
            
        Returns:
            Result dict with DeepEval metrics and ML metrics
        """
        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        test_id = test_case.get("id")
        
        # Extract labels for training (optional ground truth)
        labels = test_case.get("labels", {})
        
        logger.info(f"Evaluating test {test_id}")

        # Run RAG chain
        try:
            result = self.qa_chain.invoke({"query": question})

            actual_answer = result.get("result", "")

            # ALWAYS DEFINE FIRST
            source_docs = result.get("source_documents", []) or []

            retrieval_context = extract_context_from_retrieval(source_docs)
            
            # Extract metadata from RAG result
            metadata = result.get("metadata", {})
            context_length = metadata.get("context_length", len("\n".join(retrieval_context)))
            num_docs = metadata.get("num_docs", len(source_docs))

            #  SAFE DEBUG (after assignment)
            logger.info(f"Retrieved {len(source_docs)} documents, context_length={context_length}")

        except Exception as e:
            logger.error(f"RAG execution failed: {e}")
            self.deepeval_failures += 1
            return {
                "test_id": test_id,
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

        # Run pure DeepEval
        eval_result = EvaluationMetrics.evaluate_response(
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            retrieval_context=retrieval_context
        )

        # =========================================
        # STEP: Run ML Evaluator with optional metadata
        # =========================================
        ml_metrics = {}
        if self.ml_evaluator:
            try:
                logger.debug("Running ML Evaluator...")
                ml_metrics = self.ml_evaluator.evaluate(
                    question=question,
                    answer=actual_answer,
                    context=retrieval_context,
                    # Optional metadata for confidence calibration
                    context_length=context_length,
                    num_docs=num_docs
                )
                logger.info(f"ML metrics computed: {ml_metrics}")
            except Exception as e:
                logger.error(f"ML Evaluator failed: {e}")
                self.ml_failures += 1
                ml_metrics = {
                    "error": str(e),
                    "metrics_source": "ml_evaluator"
                }

        # Assemble output
        output = {
            "test_id": test_id,
            "category": test_case.get("category", "unknown"),
            "difficulty": test_case.get("difficulty", "unknown"),
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "num_retrieved_docs": len(retrieval_context),
            "metrics": eval_result.get("metrics", {}),
            "ml_metrics": ml_metrics,  # NEW: ML-based scores
            "labels": labels,  # NEW: Ground truth labels from test_case
            "timestamp": datetime.now().isoformat()
        }

        # =========================================
        # PHASE 2: TRAINED MODEL PREDICTIONS
        # =========================================
        trained_model_metrics = {}
        if self.trained_evaluator and self.trained_evaluator.is_available():
            try:
                logger.debug("Running trained evaluator...")
                
                # Prepare features for trained model
                features = {
                    "semantic_relevance": ml_metrics.get("semantic_relevance", 0.5),
                    "context_overlap": ml_metrics.get("context_overlap", 0.5),
                    "confidence_score": ml_metrics.get("confidence_score", 0.5),
                    "context_length": context_length,
                    "num_context_docs": num_docs
                }
                
                predictions = self.trained_evaluator.predict(features)
                trained_model_metrics = predictions
                logger.debug(f"Trained model predictions: {trained_model_metrics}")
            except Exception as e:
                logger.error(f"Trained evaluator failed: {e}")
                trained_model_metrics = {"status": "error", "error": str(e)}
        
        output["trained_model_metrics"] = trained_model_metrics

        # =========================================
        # PHASE 2: AUTO-THRESHOLD SCORING
        # =========================================
        pass_fail_result = None
        if self.threshold_engine:
            try:
                logger.debug("Running threshold evaluation...")
                
                # Combine all metrics for threshold evaluation
                all_metrics = {
                    "relevance": ml_metrics.get("semantic_relevance"),
                    "faithfulness": eval_result.get("metrics", {}).get("Faithfulness", {}).get("score"),
                    "hallucination": eval_result.get("metrics", {}).get("Hallucination", {}).get("score")
                }
                
                pass_fail_result = self.threshold_engine.evaluate_pass_fail(all_metrics)
                logger.info(f"Pass/Fail: {pass_fail_result['pass']} - {pass_fail_result['reason']}")
            except Exception as e:
                logger.error(f"Threshold evaluation failed: {e}")
                pass_fail_result = {
                    "pass": False,
                    "reason": f"Threshold evaluation error: {str(e)}",
                    "details": {}
                }
        
        output["pass_fail"] = pass_fail_result

        # =========================================
        # PHASE 2: RETRIEVAL METRICS
        # =========================================
        retrieval_metrics_result = None
        if self.retrieval_metrics:
            try:
                logger.debug("Computing retrieval metrics...")
                
                # Extract ground truth context from test case if available
                ground_truth_context = test_case.get("ground_truth_context", [])
                
                retrieval_metrics_result = self.retrieval_metrics.compute_all(
                    retrieved_docs=retrieval_context,
                    retrieved_context="\n".join(retrieval_context),
                    ground_truth_docs=ground_truth_context
                )
                logger.debug(f"Retrieval metrics: {retrieval_metrics_result}")
            except Exception as e:
                logger.error(f"Retrieval metrics computation failed: {e}")
                retrieval_metrics_result = {"error": str(e)}
        
        output["retrieval_metrics"] = retrieval_metrics_result
        
        # =========================================
        # STEP: Log Training Data (JSONL append)
        # =========================================
        self._log_training_data(output, retrieval_context, labels)

        # Add result with memory management (cap at MAX_CACHED_RESULTS)
        self.results.append(output)
        if len(self.results) > self.MAX_CACHED_RESULTS:
            # Remove oldest result to prevent unbounded memory growth
            self.results.pop(0)
            logger.debug(f"Results cache pruned to {self.MAX_CACHED_RESULTS} entries")
        
        logger.info(f"Test {test_id} complete")
        
        return output

    def _log_training_data(self, result: Dict[str, Any], context: List[str], labels: Dict[str, Any] = None) -> None:
        """
        Log evaluation results for future model training (thread-safe).
        
        Saves structured data to training_data.jsonl (JSONL format).
        Uses append mode for scalability - each record is a single JSON line.
        Thread-safe using a lock file to prevent concurrent write corruption.
        
        Args:
            result: Evaluation result dict
            context: Retrieved context strings
            labels: Optional ground truth labels from test case
        """
        import fcntl
        
        try:
            training_data_path = Path(config.TRAINING_DATA_PATH)
            training_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create training record
            training_record = {
                "timestamp": datetime.now().isoformat(),
                "test_id": result.get("test_id"),
                "category": result.get("category"),
                "difficulty": result.get("difficulty"),
                "question": result.get("question"),
                "expected_answer": result.get("expected_answer"),
                "actual_answer": result.get("actual_answer"),
                "context": context,
                "context_length": len("\n".join(context)) if context else 0,
                "num_context_docs": len(context),
                "deepeval_metrics": result.get("metrics", {}),
                "ml_metrics": result.get("ml_metrics", {}),
                "labels": labels if labels else {
                    # Default structure for optional ground truth
                    "relevance": None,
                    "hallucination": None,
                    "faithfulness": None
                }
            }
            
            # Thread-safe append: use file locking to prevent concurrent write corruption
            # On Windows, fcntl may not work - use try/except fallback
            try:
                with open(training_data_path, 'a') as f:
                    # Try to acquire exclusive lock (Unix-like systems)
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        f.write(json.dumps(training_record) + "\n")
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (AttributeError, OSError):
                        # Windows or system without fcntl - just write (Streamlit mostly single-threaded)
                        f.write(json.dumps(training_record) + "\n")
            except Exception as write_err:
                logger.error(f"Failed to write to training data file: {write_err}")
                raise
            
            logger.debug(f"Training data (JSONL) appended to {training_data_path}")
        
        except Exception as e:
            logger.warning(f"Failed to log training data: {e}")

    
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
        
        With MLflow integration (if enabled):
        - Start MLflow run
        - Log parameters and batch metrics
        - Log training data artifact
        - End run on completion
        
        Args:
            test_cases: List of test case dicts
            progress_callback: Optional progress callback
            
        Returns:
            List of results
        """
        logger.info(f"Batch evaluation: {len(test_cases)} cases")
        
        # Start MLflow run (if available)
        mlflow_run_id = None
        if self.mlflow_tracker:
            try:
                tags = {
                    "num_test_cases": len(test_cases),
                    "mlflow_enabled": True
                }
                mlflow_run_id = self.mlflow_tracker.start_run(
                    run_name=f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags=tags
                )
                logger.info(f"Started MLflow run: {mlflow_run_id}")
            except Exception as e:
                logger.error(f"Failed to start MLflow run: {e}")
        
        # Evaluate all test cases
        for i, test_case in enumerate(test_cases, 1):
            try:
                self.evaluate_single_test(test_case)
                if progress_callback:
                    progress_callback(i, len(test_cases))
            except Exception as e:
                logger.error(f"Test {i} failed: {e}")
                error_result = {
                    "test_id": test_case.get("id", i),
                    "error": str(e)
                }
                self.results.append(error_result)
                # Add memory management for error results too
                if len(self.results) > self.MAX_CACHED_RESULTS:
                    self.results.pop(0)
        
        # Log metrics to MLflow
        if self.mlflow_tracker and mlflow_run_id:
            try:
                summary = self.get_results_summary()
                
                # Log batch-level parameters
                params = {
                    "num_tests": len(test_cases),
                    "relevance_threshold": config.THRESHOLD_RELEVANCE,
                    "faithfulness_threshold": config.THRESHOLD_FAITHFULNESS,
                    "hallucination_threshold": config.THRESHOLD_HALLUCINATION
                }
                self.mlflow_tracker.log_params(params)
                
                # Log batch-level metrics
                metrics = {
                    "total_tests": summary.get("total_tests", 0),
                    "ml_failures": summary.get("ml_failures", 0),
                    "deepeval_failures": summary.get("deepeval_failures", 0)
                }
                
                # Add metric statistics
                for metric_name, stats in summary.get("metric_statistics", {}).items():
                    metrics[f"deepeval_{metric_name}_mean"] = stats.get("mean", 0)
                    metrics[f"deepeval_{metric_name}_min"] = stats.get("min", 0)
                    metrics[f"deepeval_{metric_name}_max"] = stats.get("max", 0)
                
                # Add ML metrics statistics
                for ml_metric_name, stats in summary.get("ml_metrics_statistics", {}).items():
                    metrics[f"ml_{ml_metric_name}_mean"] = stats.get("mean", 0)
                    metrics[f"ml_{ml_metric_name}_min"] = stats.get("min", 0)
                    metrics[f"ml_{ml_metric_name}_max"] = stats.get("max", 0)
                
                self.mlflow_tracker.log_metrics(metrics)
                
                # Log training data artifact
                try:
                    training_data_path = str(config.TRAINING_DATA_PATH)
                    if Path(training_data_path).exists():
                        self.mlflow_tracker.log_artifact(training_data_path, artifact_path="data")
                except Exception as e:
                    logger.warning(f"Failed to log training data artifact: {e}")
                
                # End run
                self.mlflow_tracker.end_run()
                logger.info(f"MLflow run completed: {mlflow_run_id}")
            except Exception as e:
                logger.error(f"Failed to log metrics to MLflow: {e}")
        
        return self.results

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Return raw metric statistics (no interpretation).
        Includes both DeepEval and ML metrics plus failure tracking.
        
        Returns:
            Summary dict with metric stats and failure counts
        """
        if not self.results:
            return {}

        total = len(self.results)
        
        # Collect scores by metric (DeepEval)
        metric_scores = {
            "Hallucination": [],
            "Faithfulness": [],
            "AnswerRelevancy": [],
            "ContextualRecall": []
        }

        # Collect ML metric scores
        ml_metric_scores = {
            "semantic_relevance": [],
            "context_overlap": [],
            "confidence_score": []
        }

        for result in self.results:
            metrics = result.get("metrics", {})
            for metric_name in metric_scores:
                score = metrics.get(metric_name, {}).get("score")
                if score is not None:
                    metric_scores[metric_name].append(score)
            
            # Collect ML metrics
            ml_metrics = result.get("ml_metrics", {})
            if ml_metrics and "error" not in ml_metrics:
                for ml_metric_name in ml_metric_scores:
                    score = ml_metrics.get(ml_metric_name)
                    if score is not None:
                        ml_metric_scores[ml_metric_name].append(score)

        # Compute statistics
        summary = {
            "total_tests": total,
            "ml_failures": self.ml_failures,  # NEW: Track ML evaluator failures
            "deepeval_failures": self.deepeval_failures,  # NEW: Track DeepEval failures
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

        # Add ML metrics statistics
        if any(len(scores) > 0 for scores in ml_metric_scores.values()):
            summary["ml_metrics_statistics"] = {}
            for ml_metric_name, scores in ml_metric_scores.items():
                if scores:
                    import statistics
                    summary["ml_metrics_statistics"][ml_metric_name] = {
                        "count": len(scores),
                        "mean": statistics.mean(scores),
                        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                        "min": min(scores),
                        "max": max(scores)
                    }

        logger.info(f"Summary: {total} tests evaluated, ML failures: {self.ml_failures}, DeepEval failures: {self.deepeval_failures}")
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
        return str(output_file)