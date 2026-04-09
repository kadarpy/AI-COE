"""
Configuration module for RAG Bot
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
    """Main configuration class"""

    # Paths
    BASE_DIR = Path(__file__).parent
    PROJECT_ROOT = BASE_DIR.parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    EVALUATION_DIR = PROJECT_ROOT / "tests" / "evaluation"
    DEFAULT_DOCUMENT_PATH = DOCUMENTS_DIR / "document.txt"

    # LLM Provider Configuration (supports multiple providers)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
    
    # Groq Configuration (FREE - RECOMMENDED)
    API_KEY = os.getenv("API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL")
    CONFIDENT_API_KEY = os.getenv("CONFIDENT_API_KEY", "")
    
    # Temperature for all providers
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    EVAL_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.0"))

    # Document Loading Configuration
    PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "500"))
    PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "50"))
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

    # Vector Store Configuration
    VECTOR_STORE_TYPE = str(os.getenv("VECTOR_STORE_TYPE", "chroma").lower())
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))

    # Validation Configuration
    MIN_QUESTION_LENGTH = int(os.getenv("MIN_QUESTION_LENGTH", "5"))
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "500"))
    MIN_ANSWER_LENGTH = int(os.getenv("MIN_ANSWER_LENGTH", "5"))
    MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH", "2000"))

    # Retrieval tuning
    RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "10"))  # Number of documents to fetch before final selection
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))  # Number of top documents to use for final context (strict mode)

    # Strict mode
    STRICT_CONTEXT_MODE = os.getenv("STRICT_CONTEXT_MODE", "false").lower() in ("true", "1", "yes")

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Evaluation Configuration
    EVAL_TEST_CASES_PATH = EVALUATION_DIR / "test_cases.yaml"
    TRAINING_DATA_PATH = PROJECT_ROOT / "tests" / "results" / "training_data.jsonl"
    
    # MLflow Configuration (Phase 2)
    ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "true").lower() in ("true", "1", "yes")
    MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
    MLFLOW_EXPERIMENT_NAME = "rag_evaluation"
    
    # Model Training Configuration
    ENABLE_TRAINED_EVAL = os.getenv("ENABLE_TRAINED_EVAL", "true").lower() in ("true", "1", "yes")
    MODEL_DIR = PROJECT_ROOT / "models"
    
    # Threshold Scoring Configuration
    ENABLE_THRESHOLDS = os.getenv("ENABLE_THRESHOLDS", "true").lower() in ("true", "1", "yes")
    THRESHOLD_RELEVANCE = float(os.getenv("THRESHOLD_RELEVANCE", "0.75"))
    THRESHOLD_FAITHFULNESS = float(os.getenv("THRESHOLD_FAITHFULNESS", "0.80"))
    THRESHOLD_HALLUCINATION = float(os.getenv("THRESHOLD_HALLUCINATION", "0.20"))
    
    # Retrieval metrics configuration
    RETRIEVER_K_FOR_METRICS = int(os.getenv("RETRIEVER_K_FOR_METRICS", "3"))
    
    @staticmethod
    def validate():
        """Validate configuration"""
        provider = Config.LLM_PROVIDER
        
        # Validate based on selected provide
        if provider == "groq":
            if not Config.API_KEY:
                raise ValueError("API_KEY is required for Groq provider")
            if not Config.LLM_MODEL:
                raise ValueError("LLM_MODEL is required for Groq provider")
        elif provider == "confident":
            if not Config.CONFIDENT_API_KEY:
                raise ValueError("CONFIDENT_API_KEY is required for Confident provider")
        elif provider == "deepseek":
            if not Config.API_KEY:
                raise ValueError("API_KEY is required for Deepseek provider")
            if not Config.LLM_MODEL:
                raise ValueError("LLM_MODEL is required for Deepseek provider") 
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

        # Create required directories
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        Config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        Config.MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

        return True


# Create config instance
config = Config()
