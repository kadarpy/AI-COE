"""
Vector store management module for RAG Bot
"""
import warnings
warnings.filterwarnings("ignore")
import logging
import numpy as np
from pathlib import Path
from langchain_chroma import Chroma
from config import config

logger = logging.getLogger(__name__)

# def get_embeddings():
#     """
#     Get embeddings using HuggingFace sentence-transformers (384 dims)
#     Falls back to TF-IDF if sentence-transformers unavailable
    
#     Returns:
#         Embeddings instance
#     """
#     try:
#         try:
#             from langchain_huggingface import HuggingFaceEmbeddings
#         except:
#             raise ImportError("Skip HF embeddings")
        
#         logger.info("Using HuggingFace sentence-transformers embeddings (384 dimensions)")
#         embeddings = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-base-en-v1.5",  # 384 dimensional embeddings
#             model_kwargs={"device": "cpu"}
#         )
#         return embeddings
        
#     except Exception as e:
#         logger.warning(f"HuggingFace embeddings failed: {str(e)}")
#         logger.info("Falling back to TF-IDF embeddings")
        
#         from sklearn.feature_extraction.text import TfidfVectorizer
#         from langchain_core.embeddings import Embeddings
#         import numpy as np
        
#         class TFIDFEmbeddings(Embeddings):
#             """Consistent TF-IDF embeddings with fixed dimensionality"""
            
#             def __init__(self):
#                 # Use fixed vocabulary to ensure consistent dimensions
#                 self.vectorizer = TfidfVectorizer(
#                     max_features=384,  # Fixed at 384 to match default
#                     min_df=1,
#                     stop_words='english'
#                 )
#                 self.fitted = False
#                 self.fitted_texts = []
            
#             def _pad_vector(self, vector):
#                 """Pad or trim vector to exactly 384 dimensions"""
#                 if len(vector) < 384:
#                     # Pad with zeros to reach 384 dimensions
#                     return vector + [0.0] * (384 - len(vector))
#                 elif len(vector) > 384:
#                     # Trim to 384 dimensions
#                     return vector[:384]
#                 else:
#                     return vector
            
#             def embed_documents(self, texts):
#                 """Embed documents - fits on first call"""
#                 if not self.fitted:
#                     # Fit on all texts at once for consistency
#                     self.fitted_texts = texts
#                     try:
#                         vectors = self.vectorizer.fit_transform(texts).toarray()
#                         # Pad each vector to 384 dimensions
#                         vectors = np.array([self._pad_vector(v.tolist()) for v in vectors])
#                     except Exception as fit_err:
#                         logger.error(f"Failed to fit TF-IDF: {fit_err}")
#                         # If fit fails, return dummy vectors as fallback
#                         vectors = np.zeros((len(texts), 384))
#                     self.fitted = True
#                     return vectors.tolist()
#                 else:
#                     # Use existing fitted vectorizer
#                     try:
#                         vectors = self.vectorizer.transform(texts).toarray()
#                         # Pad each vector to 384 dimensions
#                         vectors = np.array([self._pad_vector(v.tolist()) for v in vectors])
#                     except Exception as transform_err:
#                         logger.warning(f"Failed to transform texts: {transform_err}")
#                         vectors = np.zeros((len(texts), 384))
#                     return vectors.tolist()
            
#             def embed_query(self, text):
#                 """Embed query"""
#                 if not self.fitted:
#                     logger.warning("Embedding query before TFIDFEmbeddings was fitted with documents")
#                     # Fit on the query itself as a fallback
#                     _ = self.embed_documents([text])
                
#                 try:
#                     vector = self.vectorizer.transform([text]).toarray()[0]
#                     vector = self._pad_vector(vector.tolist())
#                 except Exception as e:
#                     logger.warning(f"Failed to embed query: {e}")
#                     vector = np.zeros(384).tolist()
                
#                 return vector
        
#         logger.info("Using TF-IDF embeddings (384 dimensions)")
#         return TFIDFEmbeddings()
def get_embeddings():
    from config import config

    provider = config.LLM_PROVIDER

    try:
        if provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            logger.info(f"Using Ollama embeddings with model: {config.EMBEDDING_MODEL}")
            return OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL
            )

        else:
            from langchain.embeddings import HuggingFaceEmbeddings
            logger.info("Using HuggingFace embeddings")
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5"
            )

    except Exception as e:
        logger.warning(f"Embedding init failed → fallback TF-IDF: {e}")

        from sklearn.feature_extraction.text import TfidfVectorizer
        from langchain_core.embeddings import Embeddings

        class TFIDFEmbeddings(Embeddings):
            def __init__(self):
                self.vectorizer = TfidfVectorizer(max_features=384)
                self.fitted = False

            def embed_documents(self, texts):
                vectors = self.vectorizer.fit_transform(texts).toarray()
                self.fitted = True
                return vectors.tolist()

            def embed_query(self, text):
                return self.vectorizer.transform([text]).toarray()[0].tolist()

        return TFIDFEmbeddings()

def build_vector_store(chunks):
    """
    Build and persist vector store from document chunks
    
    Args:
        chunks: List of document chunks
        
    Returns:
        Chroma vector store instance
        
    Raises:
        Exception: If vector store creation fails
    """
    if not chunks:
        logger.error("No chunks provided for vector store")
        raise ValueError("Cannot build vector store from empty chunks")

    logger.info(f"Building vector store with {len(chunks)} chunks")

    try:
        logger.debug(f"Initializing embeddings for provider: {config.LLM_PROVIDER}")
        embeddings = get_embeddings()

        logger.debug(f"Creating Chroma database at {config.CHROMA_DB_DIR}")
        vectordb = Chroma(
            persist_directory=str(config.CHROMA_DB_DIR),
            embedding_function=embeddings,
            collection_name="rag_documents"
        )

        # ADD THIS BLOCK (CRITICAL FIX)
        logger.info("Adding documents to vector store...")
        vectordb.add_documents(chunks)


        logger.info(f"{len(chunks)} documents successfully stored in vector DB")

        # 🔥 FIX: ensure TF-IDF is fitted after loading
        try:
            if hasattr(embeddings, "fitted") and not embeddings.fitted:
                logger.warning("TF-IDF not fitted → forcing fit using stored documents")

                # pull all documents from DB
                docs = vectordb.get()["documents"]

                if docs:
                    embeddings.embed_documents(docs)  # this fits vectorizer
                    logger.info("TF-IDF successfully fitted after loading")
        except Exception as e:
            logger.warning(f"TF-IDF refit failed: {e}")

        logger.info("Vector database created successfully")
        logger.debug(f"Persisting database to {config.CHROMA_DB_DIR}")

        return vectordb

    except Exception as e:
        logger.error(f"Error building vector store: {str(e)}", exc_info=True)
        raise


def load_vector_store(reset_on_mismatch=True):
    """
    Load existing vector store from disk
    
    Args:
        reset_on_mismatch: If True, reset DB on dimension mismatch instead of crashing
    
    Returns:
        Chroma vector store instance
        
    Raises:
        Exception: If vector store cannot be loaded
    """
    logger.info(f"Loading vector store from {config.CHROMA_DB_DIR}")

    if not Path(config.CHROMA_DB_DIR).exists():
        logger.error(f"Vector store not found at {config.CHROMA_DB_DIR}")
        raise FileNotFoundError(
            f"Vector store not found. Please run document loading first. "
            f"Expected at: {config.CHROMA_DB_DIR}"
        )

    try:
        logger.debug(f"Initializing embeddings for provider: {config.LLM_PROVIDER}")
        embeddings = get_embeddings()

        logger.debug(f"Loading Chroma database from {config.CHROMA_DB_DIR}")
        vectordb = Chroma(
            persist_directory=str(config.CHROMA_DB_DIR),
            embedding_function=embeddings,
            collection_name="rag_documents"
        )
        
        logger.info("Vector store loaded successfully")
        return vectordb

    except Exception as e:
        error_str = str(e)
        
        # Check if it's a dimension mismatch error
        if "expecting embedding with dimension" in error_str.lower() and reset_on_mismatch:
            logger.warning(f"Dimension mismatch detected: {error_str}")
            logger.warning("Resetting vector store...")
            
            import shutil
            try:
                # Remove the corrupted vector store
                shutil.rmtree(config.CHROMA_DB_DIR)
                logger.info("Removed corrupted vector store")
                logger.info("Vector store will be rebuilt on next document load")
                raise FileNotFoundError(
                    f"Vector store had dimension mismatch and was reset. "
                    f"Please load documents again to rebuild it."
                )
            except Exception as cleanup_err:
                logger.error(f"Failed to cleanup corrupted vector store: {str(cleanup_err)}")
                raise
        else:
            logger.error(f"Error loading vector store: {error_str}", exc_info=True)
            raise



    if Path(config.CHROMA_DB_DIR).exists():
        logger.warning(f"Deleting vector store at {config.CHROMA_DB_DIR}")
        shutil.rmtree(config.CHROMA_DB_DIR)
        logger.info("Vector store deleted successfully")
    else:
        logger.info("Vector store does not exist")
