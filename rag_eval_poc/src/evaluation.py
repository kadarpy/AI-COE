"""
RAG Bot Evaluation Module
Integrated evaluation logic for Streamlit UI
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from deepeval import test_case
from streamlit import context, metric
from streamlit import metric
import yaml
import os
from dotenv import load_dotenv

# Ensure environment is loaded before importing config
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


def _get_groq_llm():
    """
    Lazy-load Groq LLM for metrics (on-demand, not at import time)
    This ensures environment variables are loaded before configuration
    """
    try:
        from langchain_groq import ChatGroq
        from deepeval.models import DeepEvalBaseLLM
        
        # Reload environment variables to ensure they're fresh in Streamlit context
        config_path = Path(__file__).parent.parent / "config" / ".env"
        if config_path.exists():
            load_dotenv(config_path, override=True)
        
        # Try to get API key from environment or config
        llm_key = os.getenv("API_KEY")
        if not llm_key:
            try:
                llm_key = config.API_KEY
            except:
                llm_key = None
        
        if not llm_key:
            logger.warning("API_KEY not found in environment or config")
            return None
        
        
        # Create custom DeepEval-compatible Groq wrapper
        class GroqModel(DeepEvalBaseLLM):
            def __init__(self, api_key: str, model_name: str = "mixtral-8x7b-32768"):
                self.api_key = api_key
                self.model_name = model_name
                self.groq_client = ChatGroq(
                    api_key=api_key,
                    model=model_name,
                    temperature=0.0
                )
            
            def load_model(self):
                return self.groq_client
            
            def get_model_name(self) -> str:
                return self.model_name
            
            def generate(self, prompt: str) -> str:
                try:
                    response = self.groq_client.invoke(prompt)
                    return response.content
                except Exception as e:
                    logger.error(f"Groq generation error: {str(e)}")
                    raise
            
            async def a_generate(self, prompt: str) -> str:
                try:
                    response = await self.groq_client.ainvoke(prompt)
                    return response.content
                except Exception as e:
                    logger.error(f"Groq async generation error: {str(e)}")
                    raise
        
        # Create instance
        LLM_MODEL = GroqModel(api_key=llm_key, model_name=config.LLM_MODEL)
        logger.info(f"✓ Configured Groq ({config.LLM_MODEL}) as metric judge")
        return LLM_MODEL
        
    except Exception as e:
        logger.error(f"Failed to configure Groq for metrics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Lazy-load Groq - don't configure at module import time
_groq_llm = None
_groq_llm_initialized = False

def _ensure_groq_configured():
    """Ensure Groq is configured, doing it lazily on first call"""
    global _groq_llm, _groq_llm_initialized
    if not _groq_llm_initialized:
        _groq_llm = _get_groq_llm()
        _groq_llm_initialized = True
    return _groq_llm


class EvaluationMetrics:
    """Helper class for metric evaluation"""
    
    @staticmethod
    def _check_llm_configured() -> Tuple[bool, str]:
        """
        Check if Groq is configured for metrics
        Returns (is_configured, message)
        """
        # Reload environment variables to ensure they're fresh in Streamlit context
        config_path = Path(__file__).parent.parent / "config" / ".env"
        if config_path.exists():
            load_dotenv(config_path, override=True)
        
        llm_key = os.getenv("API_KEY") or config.API_KEY
        if llm_key:
            return True, f"✓ Using Groq ({config.LLM_MODEL}) for evaluation metrics"
        
        return False, (
            "LLM API Key Not Configured!\n\n"
            "Evaluation metrics require API_KEY."
        )
    
    
    @staticmethod
    def evaluate_response(question: str, actual_answer: str, expected_answer: str, 
                        retrieval_context: List[str]) -> Dict[str, Any]:
        """
        DeepEval-based evaluation (UI + run_eval consistent)
        """

        groq_llm = _ensure_groq_configured()
        def safe_measure(metric, test_case, name, retries=3):
            import time
            for i in range(retries):
                try:
                    metric.measure(test_case)
                    return
                except Exception as e:
                    if "429" in str(e):
                        time.sleep((2 ** i) * 5)
                    else:
                        raise
            raise RuntimeError(f"{name} failed after retries")

        if not groq_llm:
            return {
                "metrics": {},
                "overall_passed": False,
                "error": "Groq not configured"
            }

        # Create DeepEval test case
        llm_test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            context=retrieval_context if retrieval_context else ["No context"],
            retrieval_context=retrieval_context if retrieval_context else ["No context"]
        )

        # Get dynamic thresholds
        try:
            import streamlit as st
            t = st.session_state.get("eval_profile", {
                "faithfulness": 0.7,
                "relevancy": 0.7,
                "recall": 0.6,
                "hallucination": 0.0
            })
        except:
            t = {
                "faithfulness": 0.7,
                "relevancy": 0.7,
                "recall": 0.6,
                "hallucination": 0.0
            }

        metrics_results = {}

        # =========================
        # 1. Hallucination
        # =========================
        try:
            metric = HallucinationMetric(model=groq_llm)
            safe_measure(metric, llm_test_case, "Hallucination")

            metrics_results["Hallucination"] = {
                "score": metric.score,
                "reason": getattr(metric, "reason", ""),
                "threshold": t["hallucination"],
                "passed": metric.score <= t["hallucination"]
            }

        except Exception as e:
            metrics_results["Hallucination"] = {
                "score": None,
                "error": str(e),
                "passed": False
            }

        # =========================
        # 2. Faithfulness
        # =========================
        try:
            metric = FaithfulnessMetric(model=groq_llm)
            safe_measure(metric, llm_test_case, "Faithfulness")

            metrics_results["Faithfulness"] = {
                "score": metric.score,
                "reason": getattr(metric, "reason", ""),
                "threshold": t["faithfulness"],
                "passed": metric.score >= t["faithfulness"]
            }

        except Exception as e:
            metrics_results["Faithfulness"] = {
                "score": None,
                "error": str(e),
                "passed": False
            }

        # =========================
        # 3. Answer Relevancy
        # =========================
        try:
            metric = AnswerRelevancyMetric(model=groq_llm)
            safe_measure(metric, llm_test_case, "AnswerRelevancy")

            metrics_results["AnswerRelevancy"] = {
                "score": metric.score,
                "reason": getattr(metric, "reason", ""),
                "threshold": t["relevancy"],
                "passed": metric.score >= t["relevancy"]
            }

        except Exception as e:
            metrics_results["AnswerRelevancy"] = {
                "score": None,
                "error": str(e),
                "passed": False
            }

        # =========================
        # 4. Contextual Recall
        # =========================
        try:
            metric = ContextualRecallMetric(model=groq_llm)
            safe_measure(metric, llm_test_case, "ContextualRecall")

            metrics_results["ContextualRecall"] = {
                "score": metric.score,
                "reason": getattr(metric, "reason", ""),
                "threshold": t["recall"],
                "passed": metric.score >= t["recall"]
            }

        except Exception as e:
            metrics_results["ContextualRecall"] = {
                "score": None,
                "error": str(e),
                "passed": False
            }

        # =========================
        # STRICT PASS LOGIC
        # =========================
        hallucination = metrics_results["Hallucination"].get("score")
        faithfulness = metrics_results["Faithfulness"].get("score")
        relevancy = metrics_results["AnswerRelevancy"].get("score")
        recall = metrics_results["ContextualRecall"].get("score")

        if None in [hallucination, faithfulness, relevancy, recall]:
            overall_passed = False
        else:
            overall_passed = (
                hallucination <= t["hallucination"] and
                faithfulness >= t["faithfulness"] and
                relevancy >= t["relevancy"] and
                recall >= t["recall"]
            )

        return {
            "metrics": metrics_results,
            "overall_passed": overall_passed
        }


class TestCaseManager:
    """Manage test cases loaded from YAML"""
    
    def __init__(self):
        """Initialize test case manager"""
        self.test_cases = []
        self.default_file = Path(__file__).parent.parent / "tests" / "evaluation" / "test_cases.yaml"
    
    def load_test_cases(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load test cases from YAML file
        
        Args:
            file_path: Path to YAML test cases file
            
        Returns:
            List of test cases
        """
        yaml_file = Path(file_path) if file_path else self.default_file
        
        if not yaml_file.exists():
            raise FileNotFoundError(f"Test cases file not found: {yaml_file}")
        
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            self.test_cases = data.get("test_cases", [])
            logger.info(f"Loaded {len(self.test_cases)} test cases")
            return self.test_cases
            
        except Exception as e:
            logger.error(f"Error loading test cases: {str(e)}")
            raise
    
    def get_test_cases(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get test cases, optionally filtered by category
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of test cases
        """
        if category:
            return [tc for tc in self.test_cases if tc.get("category") == category]
        return self.test_cases
    
    def get_test_case_by_id(self, test_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific test case by ID"""
        for tc in self.test_cases:
            if tc.get("id") == test_id:
                return tc
        return None


class UIEvaluator:
    """Evaluator designed for Streamlit UI integration"""
    
    def __init__(self, qa_chain):
        """
        Initialize evaluator
        
        Args:
            qa_chain: The RAG chain to evaluate
        """
        self.qa_chain = qa_chain
        self.results = []
        self.test_case_manager = TestCaseManager()
    
    @staticmethod
    def _check_llm_for_metrics() -> Tuple[bool, str]:
        """Check if LLM judge is configured for metrics"""
        return EvaluationMetrics._check_llm_configured()
    
    def evaluate_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test case
        
        Args:
            test_case: Test case dict with question, expected_answer, etc.
            
        Returns:
            Evaluation result with metrics
        """
        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        
        logger.info(f"Evaluating Q#{test_case['id']}: {question[:50]}...")
        
        # Get RAG bot answer
        try:
            result = self.qa_chain.invoke({"query": question})
            actual_answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # Extract context from source documents
            retrieval_context = [
                doc.page_content for doc in source_docs
            ] if source_docs else ["No context retrieved"]
            
        except Exception as e:
            logger.error(f"Error getting RAG answer: {str(e)}")
            return {
                "test_id": test_case["id"],
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        # Evaluate metrics
        eval_result = EvaluationMetrics.evaluate_response(
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            retrieval_context=retrieval_context
        )
        
        # Compile result
        result_data = {
            "test_id": test_case["id"],
            "category": test_case.get("category", "unknown"),
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "context": "\n\n".join(retrieval_context),
            "num_retrieved_docs": len(source_docs),
            "metrics": eval_result["metrics"],
            "overall_passed": eval_result["overall_passed"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result_data)
        return result_data
    
    def evaluate_batch(self, test_cases: List[Dict[str, Any]], 
                      progress_callback=None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases
        
        Args:
            test_cases: List of test cases to evaluate
            progress_callback: Optional callback function for progress updates
                             Receives (current, total) as arguments
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for idx, test_case in enumerate(test_cases):
            try:
                result = self.evaluate_single_test(test_case)

                # Ensure structure consistency
                if "metrics" not in result:
                    result["metrics"] = {}
                    result["overall_passed"] = False
                    result["error"] = result.get("error", "Unknown error")

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "rate limit" in error_msg.lower():
                    error_msg = "Rate limit reached. Please retry after some time."

                result = {
                    "test_id": test_case.get("id"),
                    "category": test_case.get("category", "unknown"),
                    "question": test_case.get("question"),
                    "error": error_msg,
                    "metrics": {},
                    "overall_passed": False,
                    "timestamp": datetime.now().isoformat()
                }

                results.append(result)

                if progress_callback:
                    progress_callback(idx + 1, len(test_cases))

                continue

            results.append(result)

            if progress_callback:
                progress_callback(idx + 1, len(test_cases))
        
        return results
    
    def get_results_summary(self, results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Generate summary statistics from results
        
        Args:
            results: Optional list of results (uses self.results if not provided)
            
        Returns:
            Summary dict with statistics
        """
        eval_results = results if results is not None else self.results
        
        if not eval_results:
            return {}
        
        total_tests = len(results) if results is not None else len(self.results)

        summary = {
            "total_tests": total_tests,
            "completed_tests": len(eval_results),
            "failed_tests": sum(1 for r in eval_results if r.get("error") or not r.get("overall_passed", False)),
            "passed_tests": sum(1 for r in eval_results if r.get("overall_passed", False)),
            "metrics": {}
        }
        
        # Calculate per-metric statistics
        for metric_name in ["Hallucination", "Faithfulness", "AnswerRelevancy", "ContextualRecall"]:
            scores = []
            passed_count = 0
            
            for result in eval_results:
                if "metrics" in result:
                    metric = result["metrics"].get(metric_name, {})
                    if metric.get("score") is not None:
                        scores.append(metric["score"])
                        if metric.get("passed"):
                            passed_count += 1
            
            if scores:
                summary["metrics"][metric_name] = {
                    "avg_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "passed": passed_count,
                    "total": len(scores)
                }
        
        # Category breakdown
        summary["by_category"] = {}
        for category in set(r.get("category") for r in eval_results if "category" in r):
            category_results = [r for r in eval_results if r.get("category") == category]
            summary["by_category"][category] = {
                "count": len(category_results),
                "passed": sum(1 for r in category_results if r.get("overall_passed", False))
            }
        
        return summary
    
    def export_results_json(self, output_file: Optional[str] = None, 
                           results: Optional[List[Dict]] = None) -> str:
        """
        Export results to JSON file
        
        Args:
            output_file: Path to save JSON (auto-generated if not provided)
            results: Optional list of results (uses self.results if not provided)
            
        Returns:
            Path to output file
        """
        eval_results = results if results is not None else self.results
        
        if output_file is None:
            output_dir = Path(__file__).parent.parent / "tests" / "evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(output_dir / f"evaluation_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        return output_file
