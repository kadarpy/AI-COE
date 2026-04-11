"""
PURE RAG LAYER - Generation Only
=================================
This module implements TRUE RAG with a strict unidirectional flow:

1. Retrieval: Query → Vector Store → Documents
2. Generation: Documents → LLM → Answer

STRICT REQUIREMENTS:
✓ No evaluation logic
✓ No decision engine
✓ No retry logic
✓ No training data logging
✓ No ML validation
✓ No assumptions about answer quality

OUTPUT FORMAT:
{
    "answer": str,
    "documents": List[Document],
    "metadata": {
        "num_docs": int,
        "context_length": int
    }
}

Use this module as a PURE data transformation: query → answer
"""

import logging
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

logger = logging.getLogger(__name__)


def get_llm():
    """
    Get LLM based on configured provider.
    Returns a langchain LLM instance.

    Raises:
        ValueError: If provider is not configured or credentials are missing
    """
    from config import config

    provider = config.LLM_PROVIDER.lower()

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            if not config.API_KEY or not config.OPENAI_MODEL:
                raise ValueError("API_KEY and OPENAI_MODEL required for OpenAI")
            return ChatOpenAI(
                model=config.OPENAI_MODEL,
                temperature=config.TEMPERATURE,
                api_key=config.API_KEY,
                max_retries=3
            )

        elif provider == "groq":
            from langchain_groq import ChatGroq
            if not config.API_KEY or not config.LLM_MODEL:
                raise ValueError("API_KEY and LLM_MODEL required for Groq")
            return ChatGroq(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                api_key=config.API_KEY,
                max_retries=3
            )

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            if not config.OLLAMA_BASE_URL or not config.OLLAMA_MODEL:
                raise ValueError("OLLAMA_BASE_URL and OLLAMA_MODEL required for Ollama")
            return ChatOllama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.TEMPERATURE,
                top_p=0.9
            )

        elif provider == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            if not config.API_KEY or not config.DEEPSEEK_MODEL:
                raise ValueError("API_KEY and DEEPSEEK_MODEL required for Deepseek")
            return ChatDeepSeek(
                model=config.DEEPSEEK_MODEL,
                temperature=config.TEMPERATURE,
                api_key=config.API_KEY,
                max_retries=3
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except ImportError as e:
        logger.error(f"Failed to import LLM module for provider '{provider}': {e}")
        raise ValueError(f"LLM provider '{provider}' is not installed: {e}")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise


class PureRAG:
    """
    Pure RAG implementation - ONLY retrieval + generation.
    No side effects, no dependencies on evaluation, decision-making, or training.
    """

    def __init__(self, vectordb, llm):
        """
        Initialize Pure RAG.

        Args:
            vectordb: Chroma or LangChain vector store
            llm: LangChain LLM instance
        """
        from config import config

        self.vectordb = vectordb
        self.llm = llm
        self.retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVER_K,
                "fetch_k": config.RETRIEVER_FETCH_K
            }
        )
        logger.info(f"PureRAG initialized with k={config.RETRIEVER_K}")

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Invoke pure RAG pipeline: retrieval + generation.

        Args:
            query: Input question string

        Returns:
            {
                "answer": str,
                "documents": List[Document],
                "metadata": {...}
            }

        Raises:
            RuntimeError: If retrieval or generation fails
        """
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be non-empty string, got: {type(query)}")

        logger.info(f"PureRAG.invoke: {query[:100]}")

        # STEP 1: RETRIEVAL
        logger.debug("STEP 1: Retrieving documents...")
        try:
            docs = self.retriever.invoke(query)
            if not isinstance(docs, list):
                raise RuntimeError(f"Expected list, got {type(docs)}")
            logger.info(f"Retrieved {len(docs)} documents")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Document retrieval failed: {e}")

        if not docs:
            logger.warning("No documents retrieved")
            return {
                "answer": "The documents do not contain this information.",
                "documents": [],
                "metadata": {
                    "num_docs": 0,
                    "context_length": 0
                }
            }

        # STEP 2: CONTEXT BUILDING
        logger.debug("STEP 2: Building context from documents...")
        context_parts = []
        for doc in docs:
            context_parts.append(doc.page_content)
        context = "\n\n".join(context_parts)
        logger.debug(f"Context: {len(context)} characters")

        # STEP 3: PROMPT & GENERATION
        logger.debug("STEP 3: Generating answer...")
        prompt_template = ChatPromptTemplate.from_template(
            """
CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
You are a RAG assistant. Answer ONLY using information from the provided documents.
- Use information explicitly stated in the documents
- If not in documents, respond: "The documents do not contain this information."
- Do NOT speculate or infer beyond what is written
- Keep answers concise and direct

Answer:
"""
        )

        try:
            chain = (
                RunnableParallel(
                    context=lambda x: context,
                    question=RunnablePassthrough()
                )
                | prompt_template
                | self.llm
            )

            response = chain.invoke({"question": query})
            answer = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Generated answer: {answer[:100]}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Answer generation failed: {e}")

        # RETURN PURE RESULT
        return {
            "answer": answer,
            "documents": docs,
            "metadata": {
                "num_docs": len(docs),
                "context_length": len(context)
            }
        }


def build_pure_rag(vectordb) -> PureRAG:
    """
    Build Pure RAG from vector store.

    Args:
        vectordb: Vector store instance

    Returns:
        PureRAG instance

    Raises:
        ValueError: If initialization fails
    """
    if vectordb is None:
        raise ValueError("Vector store cannot be None")

    logger.info("Building Pure RAG")
    try:
        llm = get_llm()
        rag = PureRAG(vectordb, llm)
        logger.info("Pure RAG built successfully")
        return rag
    except Exception as e:
        logger.error(f"Failed to build Pure RAG: {e}")
        raise
