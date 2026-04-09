"""
HYBRID EVALUATION RUNNER - DeepEval + ML Evaluator
====================================================
Runs structured LLM evaluations combined with ML-based scoring.

This module executes:
- DeepEval metrics: Hallucination, Faithfulness, AnswerRelevancy, ContextualRecall
- ML Evaluator: Semantic relevance, context overlap, confidence scoring
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
from dotenv import load_dotenv

# Add src to path - MUST be done before importing local modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # Add current directory for LLM_MODEL

# Ensure environment is loaded before importing config
config_path = Path(__file__).parent.parent / "config" / ".env"
if config_path.exists():
    load_dotenv(config_path, override=True)
else:
    load_dotenv(override=True)

from LLM_MODEL import GroqModel

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


class TestCaseLoader:
    """Load and validate test cases from YAML"""

class RAGEvaluator:
    """Evaluate RAG bot using hybrid evaluation (DeepEval + ML Evaluator)"""
    
    def __init__(self, documents_dir: str = None):
        """Initialize RAG chain and evaluator"""
        self.documents_dir = documents_dir or str(config.DOCUMENTS_DIR)
        self.qa_chain = None
        self.evaluator = None
        self._setup_rag_bot()
        
    def _setup_rag_bot(self):
        """Initialize RAG bot"""
        logger.info("Setting up RAG bot...")
        
        try:
            from rag.loader import load_documents
            from rag.vector_store import build_vector_store, load_vector_store
            from rag.rag_chain import build_rag_chain
            
            # Try to load existing vector store, otherwise build from documents
            try:
                vectordb = load_vector_store()
                logger.info("Vector store loaded from disk")
            except:
                logger.info("Building new vector store...")
                doc_files = list(Path(self.documents_dir).glob("*.txt"))
                doc_files.extend(Path(self.documents_dir).glob("*.pdf"))
                
                if not doc_files:
                    raise FileNotFoundError(f"No documents in {self.documents_dir}")
                
                chunks = load_documents(str(doc_files[0]))
                vectordb = build_vector_store(chunks)
            
            self.qa_chain = build_rag_chain(vectordb)
            self.evaluator = UIEvaluator(self.qa_chain)
            logger.info("RAG bot setup complete")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def run_evaluation(self, test_cases_file: str = None):
        """Run evaluation on test cases"""
        try:
            test_manager = TestCaseManager()
            test_cases = test_manager.load_test_cases(test_cases_file)
            logger.info(f"Loaded {len(test_cases)} test cases")
            
            results = self.evaluator.evaluate_batch(test_cases)
            summary = self.evaluator.get_results_summary()
            
            logger.info(f"Evaluation complete: {len(results)} cases")
            logger.info(f"Summary: {summary}")
            
            return results, summary
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def main():
    """Main evaluation runner - pure DeepEval pipeline"""
    if not config.API_KEY:
        logger.error("API_KEY not set")
        sys.exit(1)
    
    try:
        evaluator = RAGEvaluator()
        test_cases_file = str(Path(__file__).parent / "test_cases.yaml")
        results, summary = evaluator.run_evaluation(test_cases_file)
        
        logger.info(f"Evaluation complete: {len(results)} tests")
        logger.info(f"Summary: {summary}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
