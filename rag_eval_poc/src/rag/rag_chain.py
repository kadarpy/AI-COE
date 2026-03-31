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

        if query.lower().startswith("what is"):
            query += " definition meaning explanation"
        logger.info(f"RAGChain.invoke called with query: {query[:100]}")

        # STEP 1: MANDATORY RETRIEVAL (this is what makes it RAG, not LLM)
        logger.debug("STEP 1: Retrieving documents from vector store...")
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents from vector store")
        
        if not docs:
            logger.warning("No documents retrieved - returning 'not found' response")
            return {
                "result": "I could not find any relevant information in the documents to answer your question.",
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
            """
            You are the world's most smart and knowledgeable rag assistant chatbot.
            
            Use ONLY the provided context to answer the question.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            STRICT RULES:

            1. ONLY answer using explicitly stated information in the documents
            2. DO NOT infer or guess missing definitions
            3. If the exact answer is not found, respond:
            "The documents do not contain this information."
            4. DO NOT provide assumptions or inferred explanations
            5. Keep answers concise and direct.
            6. Do NOT include explanations unless explicitly asked for them
            7. DO NOT include document names, references, or citations in your answer untill and unless explicitly asked for them.
            If asked for them, provide ONLY the document names or references without any additional commentary.

            """
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

        return {
            "result": answer,
            "source_documents": docs,
            "retrieval_count": len(docs),
            "is_rag": True  # Proof this is TRUE RAG
        }


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

