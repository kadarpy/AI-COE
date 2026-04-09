"""
RAG Chain module for RAG Bot - TRUE RAG with mandatory retrieval
"""
import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from config import config

logger = logging.getLogger(__name__)


def get_llm():
    """
    Get LLM based on configured provider
    
    Returns:
        LLM instance for the configured provider
    """
    provider = config.LLM_PROVIDER.lower()
    
    if provider == "openai":
        logger.debug("Using OpenAI LLM")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY,
            max_retries=3
        )
    
    elif provider == "groq":
        logger.debug("Using Groq LLM")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.API_KEY,
            max_retries=3
        )
    
    elif provider == "ollama":
        logger.debug("Using Ollama LLM")
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
            top_p=0.9
        )
    elif provider == "deepseek":
        logger.debug("Using Deepseek LLM")
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.API_KEY,
            max_retries=3
        )
    
    else:
        logger.error(f"Unknown provider: {provider}")
        raise ValueError(f"Unknown LLM provider: {provider}")


class RAGChain:
    """TRUE RAG Chain - MUST use retrieval before generation"""

    def __init__(self, vectordb, llm):
        self.vectordb = vectordb
        self.llm = llm
        self.retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVER_K,
                "fetch_k": config.RETRIEVER_FETCH_K
            }
        )
        logger.info(f"RAGChain initialized with retriever k={config.RETRIEVER_K}")

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke TRUE RAG chain - RETRIEVAL HAPPENS FIRST
        
        Returns:
            Dictionary with:
            - result: Generated answer
            - source_documents: Retrieved documents (proof of RAG)
            - retrieval_count: Number of documents retrieved
        """
        query = input_dict.get("query", "")
        
        # NOTE: Query rewriting removed for evaluation integrity
        # Do NOT mutate queries - they should be evaluated as-is
        logger.info(f"RAGChain.invoke called with query: {query[:100]}")

        # STEP 1: MANDATORY RETRIEVAL (this is what makes it RAG, not LLM)
        logger.debug("STEP 1: Retrieving documents from vector store...")
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents from vector store")
        
        if not docs:
            logger.warning("No documents retrieved - returning 'not found' response")
            return {
                "result": "The documents do not contain this information.",
                "source_documents": [],
                "retrieval_count": 0,
                "is_rag": True
            }

        # STEP 2: CONTEXT BUILDING from retrieved documents
        logger.debug("STEP 2: Building context from retrieved documents...")
        context_parts = []
        docs = docs[:config.RERANK_TOP_K]  
        # final strict selection
        for i, doc in enumerate(docs, 1):
            source_info = doc.metadata.get('source', f'Document {i}')
            context_parts.append(doc.page_content)
        
        context = "\n\n".join(context_parts)
        logger.debug(f"Context length: {len(context)} characters")

        # STEP 3: PROMPT WITH BALANCED INSTRUCTIONS
        logger.debug("STEP 3: Creating prompt with RAG instructions...")
        prompt_template = ChatPromptTemplate.from_template(
            """You are a RAG assistant chatbot. Your role is to answer questions using ONLY information from provided documents.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:

1. Answer ONLY using explicitly stated information in the documents
2. If the answer is not in the documents, respond: "The documents do not contain this information."
3. Do NOT infer, speculate, or provide information from training data
4. Do NOT include assumptions beyond what is written
5. Keep answers concise and direct
6. Provide brief explanations ONLY if the question explicitly asks for "explain", "why", or "how"
7. Do NOT include document names, page numbers, or citations unless explicitly asked
8. If asked for sources, provide ONLY the document names without additional commentary

Answer concisely based on the context provided."""
        )

        # STEP 4: LLM GENERATION (generation happens AFTER retrieval with context)
        logger.debug("STEP 4: Generating answer with LLM using retrieved context...")
        chain = (
            RunnableParallel(
                context=lambda x: context,
                question=RunnablePassthrough()
            )
            | prompt_template
            | self.llm
        )

        response = chain.invoke({
            "question": query
        })
        answer = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Generated answer: {answer[:100]}")

        # =========================================
        # STEP: Add metadata for ML evaluation (NEW)
        # =========================================
        output = {
            "result": answer,
            "source_documents": docs,
            "retrieval_count": len(docs),
            "is_rag": True,  # Proof this is TRUE RAG
            # NEW: Metadata for ML evaluation and training
            "metadata": {
                "context_length": len(context),  # Character count of combined context
                "num_docs": len(docs),  # Number of retrieved documents
                "generated_answer_length": len(answer),  # Length of generated answer
                "retrieval_model": "mmr"  # Metadata about retrieval strategy
            }
        }
        
        return output


def build_rag_chain(vectordb):
    """
    Build RAG chain from vector store
    
    Args:
        vectordb: Chroma vector store instance
        
    Returns:
        RAG chain instance
        
    Raises:
        Exception: If chain creation fails
    """
    if vectordb is None:
        logger.error("Vector store is None")
        raise ValueError("Vector store cannot be None")

    logger.info("Building RAG chain")

    try:
        logger.debug(f"Creating retriever with k={config.RETRIEVER_K}")

        logger.debug(f"Initializing LLM for provider: {config.LLM_PROVIDER}")
        llm = get_llm()

        logger.debug("Creating RAG chain")
        qa_chain = RAGChain(vectordb, llm)

        logger.info("RAG chain built successfully")

        return qa_chain

    except Exception as e:
        logger.error(f"Error building RAG chain: {str(e)}", exc_info=True)
        raise


def invoke_chain(qa_chain, question: str, timeout: int = 30):
    """
    Invoke RAG chain with question
    
    Args:
        qa_chain: RAG chain instance
        question: User question
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with 'result' and 'source_documents'
        
    Raises:
        Exception: If chain invocation fails
    """
    logger.debug(f"Invoking chain with question: {question[:100]}...")

    try:
        response = qa_chain.invoke({"query": question})

        if not response:
            logger.error("Chain returned empty response")
            raise ValueError("Chain returned empty response")

        logger.debug(f"Chain response received with {len(response.get('source_documents', []))} sources")

        return response

    except Exception as e:
        logger.error(f"Error invoking chain: {str(e)}", exc_info=True)
        raise

