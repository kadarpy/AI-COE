"""
DeepEval Integration for RAG Bot Evaluation

This module runs structured LLM evaluations using DeepEval metrics:
- Hallucination: Did the bot make up facts?
- Faithfulness: Did the bot stick to the source documents?
- Answer Relevancy: Did the bot answer the actual question?
- Contextual Recall: Did the bot retrieve relevant context?
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import time
from LLM_MODEL import GroqModel
from dotenv import load_dotenv

# Ensure environment is loaded before importing config
config_path = Path(__file__).parent.parent / "config" / ".env"
if config_path.exists():
    load_dotenv(config_path, override=True)
else:
    load_dotenv(override=True)

# Add src to path - handle different working directories
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# DeepEval imports
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    HallucinationMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric
)

# RAG Bot imports
from config import config
from rag.loader import load_documents
from rag.vector_store import build_vector_store, get_embeddings
from rag.rag_chain import build_rag_chain

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

DEEPEVAL_AVAILABLE = True

EVAL_PROFILES = {
    "poc": {
        "faithfulness": 0.70,
        "relevancy": 0.75,
        "recall": 0.70,
        "hallucination": 0.25
    },
    "strong": {
        "faithfulness": 0.85,
        "relevancy": 0.85,
        "recall": 0.80,
        "hallucination": 0.15
    },
    "production": {
        "faithfulness": 0.90,
        "relevancy": 0.90,
        "recall": 0.90,
        "hallucination": 0.05
    }
}


class TestCaseLoader:
    """Load and validate test cases from YAML"""

class RAGEvaluator:
    """Evaluate RAG bot using DeepEval metrics"""
    
    def __init__(self, documents_dir: str = None):
        """Initialize evaluator with RAG bot and metrics"""
        self.documents_dir = documents_dir or str(config.DOCUMENTS_DIR)
        self.results = []
        self.metrics_summary = {}
        self.qa_chain = None
        self._setup_rag_bot()
        self.groq_llm = GroqModel(
            api_key=os.getenv("API_KEY") or config.API_KEY,
            model_name=os.getenv("LLM_MODEL") or config.LLM_MODEL
        )
    
    def _setup_rag_bot(self):
        """Initialize RAG bot with documents"""
        logger.info("Setting up RAG bot...")
        
        try:
            # Find all text and PDF files
            doc_files = []
            doc_path = Path(self.documents_dir)
            
            if not doc_path.exists():
                logger.error(f"Documents directory not found: {self.documents_dir}")
                raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")
            
            # Get all TXT files
            doc_files.extend(doc_path.glob("*.txt"))
            # Get all PDF files
            doc_files.extend(doc_path.glob("*.pdf"))
            
            if not doc_files:
                logger.error(f"No documents found in {self.documents_dir}")
                raise FileNotFoundError(f"No documents found in {self.documents_dir}")
            
            logger.info(f"Found {len(doc_files)} documents")
            
            # Load and chunk documents
            all_chunks = []
            for doc_file in doc_files:
                logger.info(f"Loading {doc_file.name}...")
                try:
                    if doc_file.suffix.lower() == '.pdf':
                        from langchain_community.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(str(doc_file))
                        docs = loader.load()
                    else:  # TXT files
                        from langchain_community.document_loaders import TextLoader
                        loader = TextLoader(str(doc_file))
                        docs = loader.load()
                    
                    # Chunk documents
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=config.PDF_CHUNK_SIZE,
                        chunk_overlap=config.PDF_CHUNK_OVERLAP,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    logger.info(f"  ✓ Loaded {len(chunks)} chunks from {doc_file.name}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Error loading {doc_file}: {str(e)}")
                    continue
            
            if not all_chunks:
                raise ValueError("No document chunks created")
            
            logger.info(f"Total chunks loaded: {len(all_chunks)}")
            
            # Build vector store
            logger.info("Building vector store...")
            vectordb = build_vector_store(all_chunks)
            logger.info("✓ Vector store built")
            
            # Build RAG chain
            logger.info("Building RAG chain...")
            self.qa_chain = build_rag_chain(vectordb)
            logger.info("✓ RAG chain built successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG bot: {str(e)}")
            raise
    def safe_measure(self, metric, test_case, metric_name, max_retries=3):
        """Run metric with retry + backoff"""
        for attempt in range(max_retries):
            try:
                metric.measure(test_case)
                return metric.score, metric.reason

            except Exception as e:
                error_str = str(e)

                # Handle rate limit specifically
                if "429" in error_str or "rate_limit" in error_str:
                    wait_time = (2 ** attempt) * 5  # exponential backoff
                    logger.warning(f"{metric_name} rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e

        raise RuntimeError(f"{metric_name} failed after {max_retries} retries")
    def get_rag_answer(self, question: str, max_retries=3) -> tuple:
        """Get answer from RAG bot with retry"""

        for attempt in range(max_retries):
            try:
                result = self.qa_chain.invoke({"query": question})

                answer = result.get("result", "")
                source_docs = result.get("source_documents", [])
                context = [doc.page_content for doc in source_docs]

                return answer, context, source_docs

            except Exception as e:
                error_str = str(e)

                if "429" in error_str or "rate_limit" in error_str:
                    wait_time = (2 ** attempt) * 5
                    logger.warning(f"RAG call rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e

        logger.error("RAG call failed after retries")
        return "", [], []
        
    def get_eval_profile(self):
        if not hasattr(self, "_cached_profile"):
            self._cached_profile = getattr(self, "eval_profile", EVAL_PROFILES["poc"])
        return self._cached_profile
    
    def evaluate_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test case with all metrics
        
        Args:
            test_case: Dict with id, question, expected_answer, source_context
            
        Returns:
            Dict with metric scores and details
        """
        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        logger.info(f"Evaluating Q#{test_case['id']}: {question[:50]}...")
        
        # Get RAG bot answer
        actual_answer, context, source_docs = self.get_rag_answer(question)

        #CRITICAL VALIDATION
        if not actual_answer or actual_answer.strip() == "":
            raise ValueError(f"Empty answer generated for test {test_case['id']}")

        if not source_docs:
            logger.warning(f"No context retrieved for test {test_case['id']}")
        
        # Prepare retrieval context
        retrieval_context = [doc.page_content for doc in source_docs] if source_docs else ["No context retrieved"]
        
        # Create test case for DeepEval
        llm_test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            context=context,
            retrieval_context=retrieval_context
        )
        
        # Run metrics with error handling
        metrics_results = {}

        t = self.get_eval_profile()
        logger.debug(f"Evaluation profile: {t}")
        
        # 1. Hallucination Metric (Lower is better, 0 is perfect)
        try:
            hallucination_metric = HallucinationMetric(model=self.groq_llm)
            score, reason = self.safe_measure(
                hallucination_metric,
                llm_test_case,
                "Hallucination"
            )

            metrics_results["Hallucination"] = {
                "score": score,
                "reason": reason,
                "threshold": t["hallucination"],
                "passed": score <= t["hallucination"]
            }
            logger.debug(f"  Hallucination: {hallucination_metric.score:.2f}")
        except Exception as e:
            logger.warning(f"  Hallucination metric failed: {str(e)}")
            metrics_results["Hallucination"] = {"score": None, "error": str(e), "passed": False}
        
        # 2. Faithfulness Metric (Higher is better, 0-1 scale)
        try:
            faithfulness_metric = FaithfulnessMetric(model=self.groq_llm)
            #faithfulness_metric = FaithfulnessMetric()
            score, reason = self.safe_measure(
                faithfulness_metric,
                llm_test_case,
                "Faithfulness"
            )

            metrics_results["Faithfulness"] = {
                "score": score,
                "reason": reason,
                "threshold": t["faithfulness"],
                "passed": faithfulness_metric.score >= t["faithfulness"]  
            }
            logger.debug(f"  Faithfulness: {faithfulness_metric.score:.2f}")
        except Exception as e:
            logger.warning(f"  Faithfulness metric failed: {str(e)}")
            metrics_results["Faithfulness"] = {"score": None, "error": str(e), "passed": False}
        
        # 3. Answer Relevancy Metric (Higher is better, 0-1 scale)
        try:
            relevancy_metric = AnswerRelevancyMetric(model=self.groq_llm)
            score, reason = self.safe_measure(
                relevancy_metric,
                llm_test_case,
                "AnswerRelevancy"
            )

            metrics_results["AnswerRelevancy"] = {
                "score": score,
                "reason": reason,
                "threshold": t["relevancy"],
                "passed": relevancy_metric.score >= t["relevancy"]
            }
            logger.debug(f"  Answer Relevancy: {relevancy_metric.score:.2f}")
        except Exception as e:
            logger.warning(f"  Answer Relevancy metric failed: {str(e)}")
            metrics_results["AnswerRelevancy"] = {"score": None, "error": str(e), "passed": False}
        
        # 4. Contextual Recall Metric (Higher is better, 0-1 scale)
        try:
            contextual_recall_metric = ContextualRecallMetric(model=self.groq_llm)
            score, reason = self.safe_measure(
                contextual_recall_metric,
                llm_test_case,
                "ContextualRecall"
            )

            metrics_results["ContextualRecall"] = {
                "score": score,
                "reason": reason,
                "threshold": t["recall"],
                "passed": contextual_recall_metric.score >= t["recall"]
            }
            logger.debug(f"  Contextual Recall: {contextual_recall_metric.score:.2f}")
        except Exception as e:
            logger.warning(f"  Contextual Recall metric failed: {str(e)}")
            metrics_results["ContextualRecall"] = {"score": None, "error": str(e), "passed": False}

        for m_name, m_val in metrics_results.items():
            if m_val.get("score") is None:
                logger.error(f"{m_name} failed for test {test_case['id']}")
                metrics_results[m_name]["score"] = 0
                metrics_results[m_name]["passed"] = False
        # =========================
        # STRICT PASS LOGIC
        # =========================

        hallucination = metrics_results.get("Hallucination", {}).get("score")
        faithfulness = metrics_results.get("Faithfulness", {}).get("score")
        relevancy = metrics_results.get("AnswerRelevancy", {}).get("score")
        contextual_recall = metrics_results.get("ContextualRecall", {}).get("score")

        # Fail fast if any metric missing
        if hallucination is None or faithfulness is None or relevancy is None or contextual_recall is None:
            raise ValueError(f"Incomplete metric scores for test {test_case['id']}")

        # Strict evaluation logic

        overall_passed = (
            hallucination <= t["hallucination"] and
            faithfulness >= t["faithfulness"] and
            relevancy >= t["relevancy"] and
            contextual_recall >= t["recall"]
        ) # strict pass logic (all metrics must satisfy thresholds)

        # =========================
        # SPECIAL CASE: UNANSWERABLE
        # =========================

        if expected_answer and expected_answer.strip().lower() in ["", "unknown", "not available"]:
            # For unanswerable, hallucination must be ZERO
            if hallucination > 0.0:
                overall_passed = False

        logger.info(f"Hallucination: {hallucination:.2f}, Faithfulness: {faithfulness:.2f}, Relevancy: {relevancy:.2f}, Recall: {contextual_recall:.2f}")

        result = {
            "test_id": test_case["id"],
            "category": test_case.get("category", "unknown"),
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "context": context,
            "num_retrieved_docs": len(source_docs),
            "Bot_temperature": config.TEMPERATURE,
            "metrics": metrics_results,
            "overall_passed": overall_passed,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def run_evaluation(self, test_cases_file: str = None) -> List[Dict[str, Any]]:

        if test_cases_file is None:
            test_cases_file = str(Path(__file__).parent / "test_cases.yaml")

        logger.info(f"Loading test cases from {test_cases_file}...")

        with open(test_cases_file, 'r') as f:
            data = yaml.safe_load(f)

        test_cases = data.get("test_cases", [])
        expected_count = len(test_cases)

        logger.info(f"Loaded {expected_count} test cases")

        self.results = []
        failed_ids = []

        for i, test_case in enumerate(test_cases):
            test_id = test_case.get("id")
            # Prevent hitting daily token burst
            if i < len(test_cases) - 1:
                # adaptive delay (increase spacing as tests progress)
                delay = 3 + (i * 1.5)
                time.sleep(delay)

            try:
                logger.info(f"Running test {test_id}")

                result = self.evaluate_test_case(test_case)

                metrics = result.get("metrics", {})

                # HARD VALIDATION
                if not metrics or any(m.get("score") is None for m in metrics.values()):
                    raise ValueError(f"Incomplete metric results for test {test_id}")

                self.results.append(result)

            except Exception as e:
                logger.error(f"FAILED TEST {test_id}: {e}")
                failed_ids.append(test_id)

                self.results.append({
                    "test_id": test_id,
                    "category": test_case.get("category", "unknown"),
                    "question": test_case.get("question"),
                    "expected_answer": test_case.get("expected_answer"),
                    "actual_answer": None,
                    "error": str(e),
                    "failure_type": "evaluation_error",
                    "overall_passed": False
                })



        # =========================
        #  VALIDATION CHECK
        # =========================
        actual_count = len(self.results)

        if actual_count != expected_count:
            raise RuntimeError(
                f"Mismatch in test execution! Expected {expected_count}, got {actual_count}"
            )

        logger.info(f"✓ All {actual_count} tests executed")

        if failed_ids:
            logger.warning(f"Failed test IDs: {failed_ids}")

        return self.results
    
    def save_results(self, output_file: str = None) -> str:
        """Save evaluation results to JSON file"""
        if output_file is None:
            output_dir = Path(__file__).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(output_dir / f"evaluation_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"✓ Results saved to {output_file}")
        return output_file
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results"""
        if not self.results:
            return {}

        t = self.get_eval_profile()

        summary = {
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.get("overall_passed", False)),
            "failed_tests": sum(1 for r in self.results if not r.get("overall_passed", True)),
            "metrics": {},
            "profile_used": t
        }
        
        # Calculate per-metric statistics
        t = self.get_eval_profile()

        def is_pass(metric_name, score):
            if metric_name == "Hallucination":
                return score <= t["hallucination"]
            elif metric_name == "Faithfulness":
                return score >= t["faithfulness"]
            elif metric_name == "AnswerRelevancy":
                return score >= t["relevancy"]
            elif metric_name == "ContextualRecall":
                return score >= t["recall"]
            return False

        # Calculate per-metric statistics
        for metric_name in ["Hallucination", "Faithfulness", "AnswerRelevancy", "ContextualRecall"]:
            scores = []
            for result in self.results:
                if "metrics" in result:
                    metric = result["metrics"].get(metric_name, {})
                    if metric.get("score") is not None:
                        scores.append(metric["score"])
            
            if scores:
                summary["metrics"][metric_name] = {
                    "avg_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "passed": sum(1 for s in scores if is_pass(metric_name, s))
                }

        # Category breakdown
        summary["by_category"] = {}
        for category in set(r.get("category") for r in self.results if "category" in r):
            category_results = [r for r in self.results if r.get("category") == category]
            summary["by_category"][category] = {
                "count": len(category_results),
                "passed": sum(1 for r in category_results if r.get("overall_passed", False))
            }

        summary["failed_test_ids"] = [
            r["test_id"] for r in self.results if not r.get("overall_passed", False)
        ]

        summary["failed_test_ids"] = sorted(summary["failed_test_ids"])

        summary["pass_rate"] = round(
            summary["passed_tests"] / summary["total_tests"] * 100, 2
        )

        return summary
    
    def print_summary(self):
        """Print evaluation summary to console"""
        summary = self.generate_summary()

        print("\n" + "="*80)
        print("RAG BOT EVALUATION SUMMARY")
        print("="*80)

        # Overall
        pass_rate = summary.get("pass_rate", 0)
        print(f"\nOverall: {summary['passed_tests']}/{summary['total_tests']} tests passed ({pass_rate:.1f}%)")

        # Profile
        if "profile_used" in summary:
            print(f"Evaluation Profile: {summary['profile_used']}")

        # Metrics
        print("\nMetric Performance:")
        for metric, stats in summary.get("metrics", {}).items():
            print(f"  {metric}:")
            print(f"    Avg Score: {stats['avg_score']:.2f}")
            print(f"    Range: {stats['min_score']:.2f} - {stats['max_score']:.2f}")
            print(f"    Passed: {stats['passed']}/{summary['total_tests']}")

        # Category breakdown
        print("\nResults by Category:")
        for category, stats in summary.get("by_category", {}).items():
            print(f"  {category}: {stats['passed']}/{stats['count']} passed")

        # Failed tests
        failed = summary.get("failed_test_ids", [])
        if failed:
            print(f"\nFailed Test IDs: {failed}")

        print("="*80 + "\n")

        return summary
    
    def export_detailed_report(self, output_file: str = None) -> str:
        """Export detailed report in markdown format"""

        if output_file is None:
            output_dir = Path(__file__).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(output_dir / f"evaluation_report_{timestamp}.md")

        summary = self.generate_summary()

        pass_rate = summary.get("pass_rate", 0)

        with open(output_file, 'w') as f:
            f.write("# RAG Bot Evaluation Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Tests**: {summary['total_tests']}\n")
            f.write(f"- **Passed**: {summary['passed_tests']}\n")
            f.write(f"- **Failed**: {summary['failed_tests']}\n")
            f.write(f"- **Pass Rate**: {pass_rate:.1f}%\n")

            if "profile_used" in summary:
                f.write(f"- **Profile Used**: {summary['profile_used']}\n")

            f.write("\n")

            # Failed Tests
            if summary.get("failed_test_ids"):
                f.write("## Failed Test Cases\n\n")
                f.write(", ".join(map(str, summary["failed_test_ids"])) + "\n\n")

            # Metrics
            f.write("## Metric Performance\n\n")
            for metric, stats in summary.get("metrics", {}).items():
                f.write(f"### {metric}\n")
                f.write(f"- Average Score: {stats['avg_score']:.2f}\n")
                f.write(f"- Min: {stats['min_score']:.2f}, Max: {stats['max_score']:.2f}\n")
                f.write(f"- Passed: {stats['passed']}/{summary['total_tests']}\n\n")

            # Categories
            f.write("## Results by Category\n\n")
            for category, stats in summary.get("by_category", {}).items():
                f.write(f"- **{category}**: {stats['passed']}/{stats['count']} passed\n")

            # Detailed results
            f.write("\n## Detailed Results\n\n")
            for result in self.results:
                f.write(f"### Test #{result.get('test_id')}\n")
                f.write(f"**Category**: {result.get('category', 'unknown')}\n")

                if "error" in result:
                    f.write(f"**Status**: FAILED\n")
                    f.write(f"**Error**: {result.get('error')}\n\n")
                else:
                    f.write(f"**Question**: {result.get('question', 'N/A')}\n\n")
                    f.write(f"**Expected**: {result.get('expected_answer', 'N/A')}\n\n")
                    f.write(f"**Actual**: {result.get('actual_answer', 'N/A')}\n\n")

                if "metrics" in result:
                    f.write("**Metrics**:\n")
                    for metric, m_data in result["metrics"].items():
                        score = m_data.get("score")
                        if score is not None:
                            passed = "PASS" if m_data.get("passed") else "FAIL"
                            f.write(f"- {metric}: {score:.2f} {passed}\n")
                            if m_data.get("reason"):
                                f.write(f"  - Reason: {m_data['reason']}\n")

                f.write("\n---\n\n")

        logger.info(f"✓ Report saved to {output_file}")
        return output_file


def main():
    """Main evaluation runner"""
    # Set up GROQ API key for DeepEval metrics
    if config.API_KEY:
        os.environ["API_KEY"] = config.API_KEY
    else:
        logger.error(" API_KEY not found in environment or config!")
        logger.error("Please set API_KEY in config/.env")
        sys.exit(1)
    
    logger.info("Starting RAG Bot Evaluation with DeepEval")
    logger.info(f"LLM Provider: {config.LLM_PROVIDER}")
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator()
        
        # Run evaluation
        test_cases_file = str(Path(__file__).parent / "test_cases.yaml")
        evaluator.run_evaluation(test_cases_file)
        
        # Save results
        json_file = evaluator.save_results()
        
        # Print summary
        evaluator.print_summary()
        
        # Export detailed report
        report_file = evaluator.export_detailed_report()
        
        logger.info(f"\n✓ Evaluation complete!")
        logger.info(f"  - JSON Results: {json_file}")
        logger.info(f"  - Markdown Report: {report_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
