"""
ORCHESTRATION LAYER - RAG Evaluation Pipeline
==============================================
Composes Pure RAG + Pure Evaluation with optional advanced features.

STRICT ARCHITECTURE:
1. RAG Layer (Pure) → {answer, documents}
2. Evaluation Layer (Pure) → {metrics}
3. Orchestration (This module) → {result}

OPTIONAL (Advanced):
- Retry logic (isolated)
- Decision engine (isolated)
- Training data logging (isolated)
- MLOps integration (isolated)

UNIDIRECTIONAL FLOW:
RAG → Evaluation → Analysis

NO CIRCULAR DEPENDENCIES.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RAGEvaluationPipeline:
    """
    Composes Pure RAG + Pure Evaluation.
    Orchestrates the unidirectional flow.
    """

    def __init__(
        self,
        rag,  # Pure RAG instance
        enable_ml_metrics: bool = True,
        enable_deepeval: bool = True,
        enable_retry: bool = False,
        enable_training_logging: bool = False,
        training_data_path: Optional[Path] = None
    ):
        """
        Initialize pipeline.

        Args:
            rag: Pure RAG instance
            enable_ml_metrics: Include ML metrics in evaluation
            enable_deepeval: Include DeepEval metrics
            enable_retry: Enable retry logic on low confidence
            enable_training_logging: Log results for model training
            training_data_path: Path to training data file
        """
        self.rag = rag
        self.enable_ml_metrics = enable_ml_metrics
        self.enable_deepeval = enable_deepeval
        self.enable_retry = enable_retry
        self.enable_training_logging = enable_training_logging
        self.training_data_path = training_data_path

        logger.info(
            f"RAGEvaluationPipeline initialized: "
            f"ml_metrics={enable_ml_metrics}, "
            f"deepeval={enable_deepeval}, "
            f"retry={enable_retry}, "
            f"logging={enable_training_logging}"
        )

    def run(
        self,
        query: str,
        expected_answer: Optional[str] = None,
        max_retries: int = 0
    ) -> Dict[str, Any]:
        """
        Run full RAG → Evaluation pipeline.

        Args:
            query: Input question
            expected_answer: Optional ground truth answer
            max_retries: Maximum retries if confidence is low (requires enable_retry)

        Returns:
            {
                "result": {
                    "answer": str,
                    "documents": List,
                    "metadata": {...}
                },
                "evaluation": {
                    "deepeval_metrics": {...},
                    "ml_metrics": {...}
                },
                "pipeline_metadata": {
                    "retry_count": int,
                    "total_time_ms": float
                }
            }
        """
        import time
        start_time = time.time()
        retry_count = 0

        logger.info(f"Pipeline.run: {query[:100]}...")

        # STEP 1: RAG
        logger.debug("STEP 1: Running Pure RAG...")
        try:
            rag_result = self.rag.invoke(query)
            answer = rag_result["answer"]
            documents = rag_result["documents"]
            context = [doc.page_content for doc in documents]
        except Exception as e:
            logger.error(f"RAG failed: {e}")
            raise

        # STEP 2: EVALUATION
        logger.debug("STEP 2: Running Pure Evaluation...")
        try:
            from evaluation.pure_evaluator import evaluate

            evaluation_result = evaluate(
                question=query,
                answer=answer,
                context=context,
                expected_answer=expected_answer,
                context_length=rag_result["metadata"]["context_length"],
                num_docs=rag_result["metadata"]["num_docs"]
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            evaluation_result = {
                "deepeval_metrics": {},
                "ml_metrics": {"error": str(e)}
            }

        # STEP 3: OPTIONAL RETRY LOGIC (Isolated)
        if self.enable_retry and max_retries > 0:
            logger.debug("STEP 3: Checking for retry...")
            if self._should_retry(evaluation_result):
                logger.info("Retry triggered - re-running RAG with different parameters")
                # This is isolated retry logic - can be customized
                retry_count = self._retry_with_expanded_retrieval(
                    query=query,
                    expected_answer=expected_answer,
                    max_retries=max_retries,
                    current_result=rag_result,
                    current_evaluation=evaluation_result
                )

        # STEP 4: OPTIONAL TRAINING LOGGING (Isolated)
        if self.enable_training_logging:
            logger.debug("STEP 4: Logging training data...")
            try:
                self._log_training_data(
                    query=query,
                    answer=answer,
                    context=context,
                    evaluation=evaluation_result
                )
            except Exception as e:
                logger.warning(f"Training logging failed: {e}")

        # BUILD FINAL OUTPUT
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        return {
            "result": rag_result,
            "evaluation": evaluation_result,
            "pipeline_metadata": {
                "retry_count": retry_count,
                "total_time_ms": round(total_time_ms, 2)
            }
        }

    def _should_retry(self, evaluation_result: Dict[str, Any]) -> bool:
        """
        Check if result should trigger retry (isolated logic).

        Args:
            evaluation_result: Evaluation metrics

        Returns:
            True if confidence is below threshold
        """
        try:
            from config import config
            ml_metrics = evaluation_result.get("ml_metrics", {})
            confidence = ml_metrics.get("confidence_score", 1.0)
            threshold = config.MIN_CONFIDENCE_THRESHOLD

            should_retry = confidence < threshold
            if should_retry:
                logger.info(f"Should retry: confidence {confidence:.3f} < threshold {threshold:.3f}")
            return should_retry
        except Exception as e:
            logger.warning(f"Retry check failed: {e}")
            return False

    def _retry_with_expanded_retrieval(
        self,
        query: str,
        expected_answer: Optional[str],
        max_retries: int,
        current_result: Dict[str, Any],
        current_evaluation: Dict[str, Any]
    ) -> int:
        """
        Retry with expanded retrieval (isolated retry implementation).
        This is OPTIONAL and can be customized per use case.

        Returns:
            Number of successful retries
        """
        # This is a placeholder for advanced retry logic
        # Can be customized without affecting core RAG/Evaluation
        logger.info(f"Retry logic placeholder (max_retries={max_retries})")
        return 0

    def _log_training_data(
        self,
        query: str,
        answer: str,
        context: List[str],
        evaluation: Dict[str, Any]
    ) -> None:
        """
        Log results to training data (isolated logging).

        This is OPTIONAL and does NOT affect RAG or Evaluation.
        """
        if not self.training_data_path:
            return

        try:
            self.training_data_path.parent.mkdir(parents=True, exist_ok=True)

            record = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "answer": answer,
                "context": context,
                "evaluation": evaluation
            }

            with open(self.training_data_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            logger.debug(f"Training data logged to {self.training_data_path}")
        except Exception as e:
            logger.warning(f"Failed to log training data: {e}")


def create_pipeline(
    vectordb,
    enable_ml_metrics: bool = True,
    enable_deepeval: bool = True,
    enable_retry: bool = False,
    enable_training_logging: bool = False
) -> RAGEvaluationPipeline:
    """
    Factory function to create a fully configured pipeline.

    Args:
        vectordb: Vector store instance
        enable_ml_metrics: Include ML-based evaluation
        enable_deepeval: Include LLM-based evaluation (DeepEval)
        enable_retry: Enable retry logic
        enable_training_logging: Enable training data logging

    Returns:
        Configured RAGEvaluationPipeline
    """
    from rag.pure_rag import build_pure_rag
    from config import config

    logger.info("Creating RAG Evaluation Pipeline...")

    try:
        # Build Pure RAG
        rag = build_pure_rag(vectordb)

        # Determine training data path
        training_path = None
        if enable_training_logging:
            training_path = config.TRAINING_DATA_PATH

        # Create pipeline
        pipeline = RAGEvaluationPipeline(
            rag=rag,
            enable_ml_metrics=enable_ml_metrics,
            enable_deepeval=enable_deepeval,
            enable_retry=enable_retry,
            enable_training_logging=enable_training_logging,
            training_data_path=training_path
        )

        logger.info("Pipeline created successfully")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise
