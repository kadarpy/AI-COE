"""
PURE EVALUATION LAYER
======================
Evaluates RAG results using hybrid metrics.

INPUT:
{
    "question": str,
    "answer": str,
    "context": List[str]
}

OUTPUT:
{
    "deepeval_metrics": {
        "Hallucination": {"score": float, "reason": str},
        "Faithfulness": {"score": float, "reason": str},
        "AnswerRelevancy": {"score": float, "reason": str},
        "ContextualRecall": {"score": float, "reason": str}
    },
    "ml_metrics": {
        "semantic_relevance": float,
        "context_overlap": float,
        "confidence_score": float
    }
}

KEY PRINCIPLE: Evaluation is INDEPENDENT of RAG.
- No RAG chain imports
- No side effects
- Pure data transformation: (question, answer, context) → metrics
"""

import logging
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Configure DeepEval to be quiet
os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["CONFIDENT_DISABLE_TELEMETRY"] = "1"


def get_deepeval_llm():
    """
    Get LLM instance for DeepEval metrics.

    Returns:
        DeepEval LLM instance

    Raises:
        ValueError: If LLM is not properly configured
    """
    from config import config

    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        from groq import Groq

        # Custom adapter for Groq
        class GroqDeepEvalLLM:
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
                    temperature=0.0  # Deterministic for evaluation
                )
                return response.choices[0].message.content

            async def a_generate(self, prompt: str) -> str:
                return self.generate(prompt)

            def get_model_name(self) -> str:
                return self.model_name

        api_key = config.API_KEY
        model = config.LLM_MODEL

        if not api_key or not model:
            raise ValueError("API_KEY and LLM_MODEL required for Groq provider")

        logger.info(f"Initializing DeepEval LLM with Groq (model: {model})")
        return GroqDeepEvalLLM(api_key=api_key, model_name=model)

    else:
        raise ValueError(
            f"DeepEval currently supports 'groq' provider only. "
            f"Set LLM_PROVIDER=groq in .env. Current: {provider}"
        )


def evaluate_deepeval(
    question: str,
    actual_answer: str,
    expected_answer: str,
    retrieval_context: List[str]
) -> Dict[str, Any]:
    """
    Run DeepEval metrics (LLM-based judge).

    Args:
        question: Input question
        actual_answer: Generated answer
        expected_answer: Ground truth answer
        retrieval_context: Retrieved context

    Returns:
        {
            "Hallucination": {"score": float, "reason": str},
            "Faithfulness": {"score": float, "reason": str},
            "AnswerRelevancy": {"score": float, "reason": str},
            "ContextualRecall": {"score": float, "reason": str}
        }
    """
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        HallucinationMetric,
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric
    )

    logger.info(f"Running DeepEval metrics")

    # Create LLMTestCase
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieval_context if retrieval_context else [],
        context=retrieval_context if retrieval_context else []
    )

    llm = get_deepeval_llm()
    results = {}

    # Metric 1: Hallucination
    try:
        logger.debug("Running Hallucination metric...")
        metric = HallucinationMetric(model=llm)
        metric.measure(test_case)
        results["Hallucination"] = {
            "score": float(metric.score),
            "reason": metric.reason
        }
        logger.debug(f"Hallucination: {metric.score:.3f}")
    except Exception as e:
        logger.error(f"Hallucination metric failed: {e}")
        results["Hallucination"] = {"score": None, "reason": f"error: {str(e)}"}

    # Metric 2: Faithfulness
    try:
        logger.debug("Running Faithfulness metric...")
        metric = FaithfulnessMetric(model=llm)
        metric.measure(test_case)
        results["Faithfulness"] = {
            "score": float(metric.score),
            "reason": metric.reason
        }
        logger.debug(f"Faithfulness: {metric.score:.3f}")
    except Exception as e:
        logger.error(f"Faithfulness metric failed: {e}")
        results["Faithfulness"] = {"score": None, "reason": f"error: {str(e)}"}

    # Metric 3: AnswerRelevancy
    try:
        logger.debug("Running AnswerRelevancy metric...")
        metric = AnswerRelevancyMetric(model=llm)
        metric.measure(test_case)
        results["AnswerRelevancy"] = {
            "score": float(metric.score),
            "reason": metric.reason
        }
        logger.debug(f"AnswerRelevancy: {metric.score:.3f}")
    except Exception as e:
        logger.error(f"AnswerRelevancy metric failed: {e}")
        results["AnswerRelevancy"] = {"score": None, "reason": f"error: {str(e)}"}

    # Metric 4: ContextualRecall
    try:
        logger.debug("Running ContextualRecall metric...")
        metric = ContextualRecallMetric(model=llm)
        metric.measure(test_case)
        results["ContextualRecall"] = {
            "score": float(metric.score),
            "reason": metric.reason
        }
        logger.debug(f"ContextualRecall: {metric.score:.3f}")
    except Exception as e:
        logger.error(f"ContextualRecall metric failed: {e}")
        results["ContextualRecall"] = {"score": None, "reason": f"error: {str(e)}"}

    logger.info(f"DeepEval evaluation complete")
    return results


def evaluate_ml_metrics(
    question: str,
    answer: str,
    context: List[str],
    context_length: Optional[int] = None,
    num_docs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run ML-based metrics (CrossEncoder).

    Args:
        question: Input question
        answer: Generated answer
        context: Retrieved context
        context_length: Optional context length for calibration
        num_docs: Optional number of docs for calibration

    Returns:
        {
            "semantic_relevance": float,
            "context_overlap": float,
            "confidence_score": float
        }
    """
    from ml_evaluator import get_ml_evaluator

    logger.info(f"Running ML metrics")

    try:
        ml_evaluator = get_ml_evaluator()
        ml_metrics = ml_evaluator.evaluate(
            question=question,
            answer=answer,
            context=context,
            context_length=context_length,
            num_docs=num_docs
        )
        logger.info(f"ML metrics: {ml_metrics}")
        return ml_metrics
    except Exception as e:
        logger.error(f"ML evaluation failed: {e}")
        return {
            "semantic_relevance": 0.5,
            "context_overlap": 0.5,
            "confidence_score": 0.5,
            "error": str(e)
        }


def evaluate(
    question: str,
    answer: str,
    context: List[str],
    expected_answer: Optional[str] = None,
    context_length: Optional[int] = None,
    num_docs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Pure evaluation function - hybrid DeepEval + ML metrics.

    THIS IS THE PUBLIC EVALUATION API.

    Args:
        question: Input question
        answer: Generated answer
        context: Retrieved context (List of strings)
        expected_answer: Optional ground truth
        context_length: Optional context length
        num_docs: Optional number of docs

    Returns:
        {
            "deepeval_metrics": {...},
            "ml_metrics": {...}
        }
    """
    logger.info(f"Pure Evaluation: question={question[:50]}...")

    if not expected_answer:
        expected_answer = ""

    # Run both metric sets
    deepeval_metrics = evaluate_deepeval(
        question=question,
        actual_answer=answer,
        expected_answer=expected_answer,
        retrieval_context=context
    )

    ml_metrics = evaluate_ml_metrics(
        question=question,
        answer=answer,
        context=context,
        context_length=context_length,
        num_docs=num_docs
    )

    return {
        "deepeval_metrics": deepeval_metrics,
        "ml_metrics": ml_metrics
    }
