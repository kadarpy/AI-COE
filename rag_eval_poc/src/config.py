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
    LLM_MODEL = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")  # Default Groq model
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
    
    # Deepseek Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # Confident Configuration (optional)
    CONFIDENT_API_KEY = os.getenv("CONFIDENT_API_KEY", "")
    
    # Temperature for all providers
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    EVAL_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.0"))

    # ML Model Configuration
    ML_EMBEDDINGS_MODEL = os.getenv("ML_EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")
    ML_EVALUATOR_MODEL = os.getenv("ML_EVALUATOR_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

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
    
    # ========================================
    # SELF-IMPROVING RAG CONFIGURATION (NEW)
    # ========================================
    
    # Reranking Configuration
    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() in ("true", "1", "yes")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Answer Validation Configuration
    ENABLE_ANSWER_VALIDATION = os.getenv("ENABLE_ANSWER_VALIDATION", "true").lower() in ("true", "1", "yes")
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.5"))
    
    # Adaptive Retry Configuration
    ENABLE_ADAPTIVE_RETRY = os.getenv("ENABLE_ADAPTIVE_RETRY", "true").lower() in ("true", "1", "yes")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
    RETRIEVER_K_INCREMENT = int(os.getenv("RETRIEVER_K_INCREMENT", "2"))  # Increase K by this amount on retry
    
    # Decision Engine Configuration
    ENABLE_DECISION_ENGINE = os.getenv("ENABLE_DECISION_ENGINE", "true").lower() in ("true", "1", "yes")
    HALLUCINATION_REJECT_THRESHOLD = float(os.getenv("HALLUCINATION_REJECT_THRESHOLD", "0.5"))
    
    # Feedback Loop Configuration
    ENABLE_FEEDBACK_LOOP = os.getenv("ENABLE_FEEDBACK_LOOP", "true").lower() in ("true", "1", "yes")
    
    # Trained Model Configuration (NEW)
    ENABLE_TRAINED_EVAL = os.getenv("ENABLE_TRAINED_EVAL", "true").lower() in ("true", "1", "yes")
    MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent.parent / "models")))
    TRAINING_DATA_PATH = Path(os.getenv("TRAINING_DATA_PATH", str(Path(__file__).parent.parent / "training_data" / "training_data.jsonl")))
    
    # Retrieval metrics configuration
    
    @staticmethod
    def validate():
        """Validate configuration"""
        provider = Config.LLM_PROVIDER.lower()
        
        # Validate based on selected provider
        if provider == "groq":
            if not Config.API_KEY:
                raise ValueError("API_KEY is required for Groq provider. Set API_KEY in .env or environment")
            if not Config.LLM_MODEL:
                raise ValueError("LLM_MODEL is required for Groq provider. Set LLM_MODEL in .env or environment")
                
        elif provider == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider. Set OPENAI_API_KEY in .env or environment")
            if not Config.OPENAI_MODEL:
                raise ValueError("OPENAI_MODEL is required for OpenAI provider. Set OPENAI_MODEL in .env or environment")
                
        elif provider == "ollama":
            if not Config.OLLAMA_BASE_URL:
                raise ValueError("OLLAMA_BASE_URL is required for Ollama provider. Set OLLAMA_BASE_URL in .env or environment")
            if not Config.OLLAMA_MODEL:
                raise ValueError("OLLAMA_MODEL is required for Ollama provider. Set OLLAMA_MODEL in .env or environment")
                
        elif provider == "deepseek":
            if not Config.DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY is required for Deepseek provider. Set DEEPSEEK_API_KEY in .env or environment")
            if not Config.DEEPSEEK_MODEL:
                raise ValueError("DEEPSEEK_MODEL is required for Deepseek provider. Set DEEPSEEK_MODEL in .env or environment")
        
        elif provider == "confident":
            if not Config.CONFIDENT_API_KEY:
                raise ValueError("CONFIDENT_API_KEY is required for Confident provider. Set CONFIDENT_API_KEY in .env or environment")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Supported: groq, openai, ollama, deepseek, confident")

        # Create required directories
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        Config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        Config.MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

        return True


# Create config instance
config = Config()
