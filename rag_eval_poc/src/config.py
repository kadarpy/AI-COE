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
    DOCUMENTS_DIR = DATA_DIR
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    EVALUATION_DIR = PROJECT_ROOT / "tests" / "evaluation"
    DEFAULT_DOCUMENT_PATH = DATA_DIR / "document.txt"

    # LLM Provider Configuration (supports multiple providers)
    LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "").lower()
    
    # Groq Configuration (FREE - RECOMMENDED)
    API_KEY = os.getenv("API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL")
    CONFIDENT_API_KEY = os.getenv("CONFIDENT_API_KEY", "")
    BASE_URL = os.getenv("BASE_URL", "")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
    
    # Temperature for all providers
    TEMPERATURE = float(os.getenv("TEMPERATURE"))
    EVAL_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE"))

    # Document Loading Configuration
    DOC_CHUNK_SIZE = int(os.getenv("DOC_CHUNK_SIZE"))
    DOC_CHUNK_OVERLAP = int(os.getenv("DOC_CHUNK_OVERLAP"))
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

    # Vector Store Configuration
    VECTOR_STORE_TYPE = str(os.getenv("VECTOR_STORE_TYPE").lower())
    RETRIEVER_K = int(os.getenv("RETRIEVER_K"))

    # Validation Configuration
    MIN_QUESTION_LENGTH = int(os.getenv("MIN_QUESTION_LENGTH"))
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH"))
    MIN_ANSWER_LENGTH = int(os.getenv("MIN_ANSWER_LENGTH"))
    MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH"))

    # Retrieval tuning
    RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K"))  # Number of documents to fetch before final selection
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K"))  # Number of top documents to use for final context (strict mode)

    # Strict mode
    STRICT_CONTEXT_MODE = os.getenv("STRICT_CONTEXT_MODE")

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL").upper()
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Evaluation Configuration
    EVAL_TEST_CASES_PATH = EVALUATION_DIR / "test_cases.yaml"
    
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
        elif provider == "ollama":
            if not Config.OLLAMA_BASE_URL:
                print("OLLAMA_BASE_URL is required for Ollama provider")
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
