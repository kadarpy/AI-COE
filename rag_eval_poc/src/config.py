"""
Configuration module for RAG Bot
Fully dynamic - all settings loaded from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from config folder
config_path = Path(__file__).parent.parent / "config" / ".env"
if config_path.exists():
    load_dotenv(config_path)
else:
    load_dotenv()


class Config:
    """Main configuration class - dynamically loaded from environment variables"""

    # ===================== PATHS CONFIGURATION =====================
    BASE_DIR = Path(__file__).parent
    PROJECT_ROOT = BASE_DIR.parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    EVALUATION_DIR = PROJECT_ROOT / "tests" / "evaluation"
    DEFAULT_DOCUMENT_PATH = DATA_DIR / "document.txt"
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

    # ===================== LLM PROVIDER CONFIGURATION =====================
    LLM_PROVIDER = os.getenv("LLM_PROVIDER").lower()
    LLM_MODEL = os.getenv("LLM_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

    # ===================== MODEL BEHAVIOR CONFIGURATION =====================
    TEMPERATURE = float(os.getenv("TEMPERATURE"))
    EVAL_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE"))

    # ===================== DOCUMENT PROCESSING CONFIGURATION =====================
    DOC_CHUNK_SIZE = int(os.getenv("DOC_CHUNK_SIZE"))
    DOC_CHUNK_OVERLAP = int(os.getenv("DOC_CHUNK_OVERLAP"))

    # ===================== VECTOR STORE CONFIGURATION =====================
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE").lower()

    # ===================== RETRIEVER CONFIGURATION =====================
    RETRIEVER_K = int(os.getenv("RETRIEVER_K"))
    RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K"))

    # ===================== VALIDATION CONSTRAINTS =====================
    MIN_QUESTION_LENGTH = int(os.getenv("MIN_QUESTION_LENGTH"))
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH"))
    MIN_ANSWER_LENGTH = int(os.getenv("MIN_ANSWER_LENGTH"))
    MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH"))

    # ===================== CONTEXT & RESPONSE CONFIGURATION =====================
    STRICT_CONTEXT_MODE = os.getenv("STRICT_CONTEXT_MODE")

    # ===================== LOGGING CONFIGURATION =====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ===================== EVALUATION CONFIGURATION =====================
    EVAL_TEST_CASES_PATH = EVALUATION_DIR / "test_cases.yaml"
    EVAL_NUM_RUNS = int(os.getenv("EVAL_NUM_RUNS", "3"))
    EVAL_SEED = int(os.getenv("EVAL_SEED", "42"))

    # ===================== RERANKER CONFIGURATION =====================
    ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
    RERANKER_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", "0.3"))

    # ===================== CACHE CONFIGURATION =====================
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_DIR = os.getenv("CACHE_DIR", ".cache")

    # ===================== RETRIEVAL EVALUATION CONFIGURATION =====================
    RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.7"))
    COMPUTE_RETRIEVAL_METRICS = os.getenv("COMPUTE_RETRIEVAL_METRICS", "true").lower() == "true"

    # ===================== PASS/FAIL THRESHOLDS =====================
    THRESHOLD_FAITHFULNESS = float(os.getenv("THRESHOLD_FAITHFULNESS", "0.9"))
    THRESHOLD_HALLUCINATION = float(os.getenv("THRESHOLD_HALLUCINATION", "0.1"))
    THRESHOLD_ANSWER_RELEVANCY = float(os.getenv("THRESHOLD_ANSWER_RELEVANCY", "0.85"))
    THRESHOLD_RECALL = float(os.getenv("THRESHOLD_RECALL", "0.85"))
    THRESHOLD_PRECISION = float(os.getenv("THRESHOLD_PRECISION", "0.80"))
    THRESHOLD_HIT_RATE = float(os.getenv("THRESHOLD_HIT_RATE", "0.80"))

    # ===================== STABILITY THRESHOLDS =====================
    MAX_ALLOWED_VARIANCE = float(os.getenv("MAX_ALLOWED_VARIANCE", "0.15"))
    STABILITY_WARNING_THRESHOLD = float(os.getenv("STABILITY_WARNING_THRESHOLD", "0.10"))

    @staticmethod
    def validate():
        """Validate configuration based on selected LLM provider"""
        provider = Config.LLM_PROVIDER

        # Validate provider and required fields
        if provider == "groq":
            if not Config.API_KEY:
                raise ValueError("API_KEY is required for Groq provider")
            if not Config.LLM_MODEL:
                raise ValueError("LLM_MODEL is required for Groq provider")
        elif provider == "deepseek":
            if not Config.API_KEY:
                raise ValueError("API_KEY is required for Deepseek provider")
            if not Config.LLM_MODEL:
                raise ValueError("LLM_MODEL is required for Deepseek provider")
        elif provider == "ollama":
            if not Config.OLLAMA_BASE_URL:
                raise ValueError("OLLAMA_BASE_URL is required for Ollama provider")
            if not Config.LLM_MODEL:
                raise ValueError("LLM_MODEL is required for Ollama provider")
        elif provider == "openai":
            if not Config.API_KEY:
                raise ValueError("API_KEY is required for OpenAI provider")
            if not Config.LLM_MODEL:
                raise ValueError("LLM_MODEL is required for OpenAI provider")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

        # Create required directories
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        Config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

        return True


# Create config instance
config = Config()
