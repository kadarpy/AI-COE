"""
RAG Chain module for RAG Bot - TRUE RAG with mandatory retrieval
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from config import config

logger = logging.getLogger(__name__)


def get_llm():
    """
    Get LLM based on configured provider
    
    Returns:
        LLM instance for the configured provider
        
    Raises:
        ValueError: If provider is not configured or credentials are missing
    """
    provider = config.LLM_PROVIDER.lower()
    
    try:
        if provider == "openai":
            logger.debug("Using OpenAI LLM")
            from langchain_openai import ChatOpenAI
            
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured. Set OPENAI_API_KEY in .env or environment")
            if not config.OPENAI_MODEL:
                raise ValueError("OPENAI_MODEL not configured. Set OPENAI_MODEL in .env or environment")
                
            return ChatOpenAI(
                model=config.OPENAI_MODEL,
                temperature=config.TEMPERATURE,
                api_key=config.OPENAI_API_KEY,
                max_retries=3
            )
        
        elif provider == "groq":
            logger.debug("Using Groq LLM")
            from langchain_groq import ChatGroq
            
            if not config.API_KEY:
                raise ValueError("API_KEY not configured for Groq. Set API_KEY in .env or environment")
            if not config.LLM_MODEL:
                raise ValueError("LLM_MODEL not configured for Groq. Set LLM_MODEL in .env or environment")
                
            return ChatGroq(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                api_key=config.API_KEY,
                max_retries=3
            )
        
        elif provider == "ollama":
            logger.debug("Using Ollama LLM")
            from langchain_community.chat_models import ChatOllama
            
            if not config.OLLAMA_BASE_URL:
                raise ValueError("OLLAMA_BASE_URL not configured. Set OLLAMA_BASE_URL in .env or environment")
            if not config.OLLAMA_MODEL:
                raise ValueError("OLLAMA_MODEL not configured. Set OLLAMA_MODEL in .env or environment")
                
            return ChatOllama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.TEMPERATURE,
                top_p=0.9
            )
            
        elif provider == "deepseek":
            logger.debug("Using Deepseek LLM")
            from langchain_deepseek import ChatDeepSeek
            
            if not config.DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY not configured. Set DEEPSEEK_API_KEY in .env or environment")
            if not config.DEEPSEEK_MODEL:
                raise ValueError("DEEPSEEK_MODEL not configured. Set DEEPSEEK_MODEL in .env or environment")
                
            return ChatDeepSeek(
                model=config.DEEPSEEK_MODEL,
                temperature=config.TEMPERATURE,
                api_key=config.DEEPSEEK_API_KEY,
                max_retries=3
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported: groq, openai, ollama, deepseek")
            
    except ImportError as e:
        logger.error(f"Failed to import LLM module for provider '{provider}': {e}")
        raise ValueError(f"LLM provider '{provider}' is not installed. Install required dependencies.")
    except Exception as e:
        logger.error(f"Error initializing LLM with provider '{provider}': {e}")
        raise


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

    def _get_trained_predictions(
        self,
        ml_validation: Dict[str, Any],
        num_docs: int,
        context_length: int
    ) -> Dict[str, Any]:
        """
        Get trained model predictions for answer evaluation.
        
        Args:
            ml_validation: ML validation results with scores
            num_docs: Number of retrieved documents
            context_length: Length of combined context
        
        Returns:
            Dictionary with trained model predictions or empty dict if unavailable
        """
        if not config.ENABLE_TRAINED_EVAL or ml_validation is None:
            return {}
        
        try:
            from mlops.evaluator_model import TrainedEvaluator
            
            trained_eval = TrainedEvaluator(model_dir=str(config.MODEL_DIR))
            
            if not trained_eval.is_available():
                logger.debug("No trained evaluator models available")
                return {}
            
            # Prepare features from ML validation
            features = {
                "semantic_relevance": ml_validation.get("semantic_relevance", 0.5),
                "context_overlap": ml_validation.get("context_overlap", 0.5),
                "confidence_score": ml_validation.get("confidence_score", 0.5),
                "context_length": context_length,
                "num_context_docs": num_docs
            }
            
            # Get predictions
            predictions = trained_eval.predict(features)
            logger.info(f"Trained model predictions: {predictions}")
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Failed to get trained predictions: {e}")
            return {}

    def _compute_retrieval_metrics(
        self,
        docs: List[Document],
        context: str
    ) -> Dict[str, float]:
        """
        Compute retrieval quality metrics.
        
        Args:
            docs: Retrieved documents
            context: Combined context string
        
        Returns:
            Dictionary with retrieval metrics
        """
        if not docs or not context:
            return {}
        
        try:
            from evaluation.retrieval_metrics import RetrievalMetrics
            
            metrics_computer = RetrievalMetrics()
            
            # Extract document contents
            retrieved_docs = [doc.page_content for doc in docs]
            
            # Note: We don't have ground truth in real-time, so we compute metrics
            # that don't require it. For full evaluation, ground truth would come
            # from human feedback or labeled datasets.
            metrics = {
                "num_retrieved": len(docs),
                "context_length": len(context),
                "avg_doc_length":  len(context) / len(docs) if docs else 0
            }
            
            logger.info(f"Retrieval metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to compute retrieval metrics: {e}")
            return {}

    def _log_to_training_data(
        self,
        query: str,
        answer: str,
        context: str,
        ml_validation: Dict[str, Any],
        decision: Dict[str, Any],
        trained_predictions: Dict[str, Any],
        retrieval_metrics: Dict[str, float],
        num_docs: int,
        retry_count: int
    ) -> None:
        """
        Log generation and evaluation results to training data file for model improvement.
        
        Args:
            query: Input query
            answer: Generated answer
            context: Retrieved context
            ml_validation: ML validation scores
            decision: Decision engine result
            trained_predictions: Trained model predictions
            retrieval_metrics: Retrieval quality metrics
            num_docs: Number of retrieved documents
            retry_count: Number of retries performed
        """
        if not config.ENABLE_FEEDBACK_LOOP:
            logger.debug("Feedback loop disabled - not logging to training data")
            return
        
        try:
            # Create training data entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "answer": answer,
                "context_length": len(context),
                "num_docs": num_docs,
                "answer_length": len(answer),
                "ml_validation": ml_validation,
                "trained_predictions": trained_predictions,
                "retrieval_metrics": retrieval_metrics,
                "decision": decision,
                "retry_count": retry_count
            }
            
            # Ensure training data directory exists
            training_data_path = config.TRAINING_DATA_PATH
            training_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to JSONL file
            with open(training_data_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            logger.info(f"Logged training data to {training_data_path}")
            
        except Exception as e:
            logger.warning(f"Failed to log training data: {e}")

    def _generate_answer_with_validation(
        self,
        query: str,
        docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Generate and validate answer for given query and documents.
        
        Internal helper method for retry logic.
        
        Args:
            query: Input query string
            docs: List of Document objects to use for context
        
        Returns:
            Dictionary with generation and validation results
        """
        # STEP 2: CONTEXT BUILDING from retrieved documents
        logger.debug("Building context from retrieved documents...")
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source_info = doc.metadata.get('source', f'Document {i}')
            context_parts.append(doc.page_content)
        
        context = "\n\n".join(context_parts)
        logger.debug(f"Context length: {len(context)} characters")

        # STEP 3: PROMPT WITH BALANCED INSTRUCTIONS
        logger.debug("Creating prompt with RAG instructions...")
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
        logger.debug("Generating answer with LLM using retrieved context...")
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
        # STEP 5: ML VALIDATION (NEW - Self-Improving RAG)
        # =========================================
        ml_validation = None
        confidence_score = 1.0
        
        if config.ENABLE_ANSWER_VALIDATION:
            logger.debug("Running ML validation on generated answer...")
            try:
                from ml_evaluator import get_ml_evaluator
                
                ml_evaluator = get_ml_evaluator()
                ml_validation = ml_evaluator.evaluate(
                    question=query,
                    answer=answer,
                    context=[doc.page_content for doc in docs],
                    context_length=len(context),
                    num_docs=len(docs)
                )
                confidence_score = ml_validation.get("confidence_score", 1.0)
                logger.info(f"ML Validation - Confidence: {confidence_score:.3f}")
                
            except Exception as e:
                logger.warning(f"ML validation failed, continuing without validation: {e}")
                ml_validation = None
                confidence_score = 1.0
        
        # =========================================
        # STEP 6: DECISION ENGINE (NEW - Self-Improving RAG)
        # =========================================
        decision_result = None
        
        if config.ENABLE_DECISION_ENGINE:
            logger.debug("Running decision engine...")
            try:
                from mlops.decision_engine import get_decision_engine
                
                decision_engine = get_decision_engine()
                decision_result = decision_engine.make_decision(ml_validation)
                logger.info(f"Decision: {decision_result['action'].upper()} - {decision_result['reason']}")
                
            except Exception as e:
                logger.warning(f"Decision engine failed, accepting answer: {e}")
                decision_result = {
                    "action": "accept",
                    "confidence_score": confidence_score,
                    "hallucination_score": None,
                    "reason": "Decision engine unavailable",
                    "accepted": True
                }
        
        return {
            "answer": answer,
            "context": context,
            "ml_validation": ml_validation,
            "confidence_score": confidence_score,
            "decision": decision_result
        }

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
        
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query: expected non-empty string, got {type(query)}")
            return {
                "result": "Error: Invalid query provided",
                "source_documents": [],
                "retrieval_count": 0,
                "is_rag": True,
                "error": "Invalid query type"
            }
        
        # NOTE: Query rewriting removed for evaluation integrity
        # Do NOT mutate queries - they should be evaluated as-is
        logger.info(f"RAGChain.invoke called with query: {query[:100]}")

        # STEP 1: MANDATORY RETRIEVAL (this is what makes it RAG, not LLM)
        logger.debug("STEP 1: Retrieving documents from vector store...")
        try:
            docs = self.retriever.invoke(query)
            
            # Defensive check: ensure docs is a list
            if docs is None:
                logger.error(f"Retriever returned None for query: {query[:100]}")
                raise RuntimeError("Retriever failure: returned None (likely embedding failure)")
            
            if not isinstance(docs, list):
                logger.error(f"Retriever returned unexpected type {type(docs)}, expected list")
                raise RuntimeError(f"Retriever returned unexpected type: {type(docs)}")
            
            logger.info(f"Retrieved {len(docs)} documents from vector store")
            
        except Exception as retrieval_error:
            logger.error(f"Retrieval failed: {retrieval_error}")
            return {
                "result": f"Error during document retrieval: {str(retrieval_error)[:100]}",
                "source_documents": [],
                "retrieval_count": 0,
                "is_rag": True,
                "error": str(retrieval_error)
            }
        
        # STEP 1.5: OPTIONAL RERANKING (NEW - Self-Improving RAG)
        if config.ENABLE_RERANKING and docs:
            logger.debug("STEP 1.5: Reranking documents by relevance...")
            try:
                from mlops.reranker import get_reranker
                reranker = get_reranker()
                # Rerank and limit to RERANK_TOP_K
                docs = reranker.rerank(query, docs, top_k=config.RERANK_TOP_K)
                logger.info(f"Reranked to top {len(docs)} documents by relevance")
            except Exception as e:
                logger.warning(f"Reranking failed, continuing with original document order: {e}")
                # Fall back to original ordering if reranking fails
                docs = docs[:config.RERANK_TOP_K]
        else:
            if not docs:
                logger.warning("No documents retrieved - returning 'not found' response")
            # Apply strict limit without reranking
            docs = docs[:config.RERANK_TOP_K]
        
        if not docs:
            logger.warning("No documents retrieved - returning 'not found' response")
            return {
                "result": "The documents do not contain this information.",
                "source_documents": [],
                "retrieval_count": 0,
                "is_rag": True,
                "decision": {
                    "action": "reject",
                    "reason": "No documents retrieved",
                    "accepted": False
                },
                "retry_count": 0
            }

        # =========================================
        # STEP 2-6: GENERATION + VALIDATION WITH ADAPTIVE RETRY
        # =========================================
        current_retriever_k = config.RETRIEVER_K
        retry_count = 0
        generation_result = None
        max_retries_allowed = config.MAX_RETRIES
        
        while retry_count <= max_retries_allowed:
            try:
                logger.info(f"Generation attempt {retry_count + 1} of {max_retries_allowed + 1}")
                
                # Generate and validate answer with current docs
                generation_result = self._generate_answer_with_validation(query, docs)
                
                # Extract decision from generation result
                decision = generation_result.get("decision", {})
                decision_action = decision.get("action", "accept")
                
                logger.info(f"Decision: {decision_action.upper()}")
                
                # Check if we should retry
                if decision_action == "retry" and retry_count < max_retries_allowed:
                    logger.info(
                        f"Retry triggered: {decision.get('reason')} "
                        f"(attempt {retry_count + 1} of {max_retries_allowed})"
                    )
                    
                    if config.ENABLE_ADAPTIVE_RETRY:
                        # Implement adaptive retry: increase K and re-retrieve
                        old_k = current_retriever_k
                        current_retriever_k += config.RETRIEVER_K_INCREMENT
                        
                        logger.info(
                            f"Adaptive retry enabled: increasing RETRIEVER_K "
                            f"from {old_k} to {current_retriever_k}"
                        )
                        
                        # Re-retrieve documents with increased K
                        updated_retriever = self.vectordb.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": current_retriever_k,
                                "fetch_k": config.RETRIEVER_FETCH_K
                            }
                        )
                        docs = updated_retriever.invoke(query)
                        logger.info(f"Retrieved {len(docs)} documents on retry with K={current_retriever_k}")
                        
                        # Re-rerank if enabled
                        if config.ENABLE_RERANKING and docs:
                            logger.debug("Re-ranking documents after retrieval expansion...")
                            try:
                                from mlops.reranker import get_reranker
                                reranker = get_reranker()
                                docs = reranker.rerank(query, docs, top_k=config.RERANK_TOP_K)
                                logger.info(f"Reranked to top {len(docs)} documents")
                            except Exception as e:
                                logger.warning(f"Reranking failed on retry: {e}")
                                docs = docs[:config.RERANK_TOP_K]
                        else:
                            docs = docs[:config.RERANK_TOP_K]
                        
                        retry_count += 1
                        # Continue to next iteration of while loop
                        continue
                    else:
                        logger.info("Adaptive retry disabled - stopping retries")
                        break
                else:
                    # Accept or Reject - stop retrying
                    if decision_action == "retry":
                        logger.warning(f"Max retries ({max_retries_allowed}) reached - accepting answer")
                    else:
                        logger.info(f"Decision is {decision_action} - stopping retries")
                    break
                    
            except Exception as e:
                logger.error(f"Error during generation attempt {retry_count + 1}: {e}")
                if retry_count < max_retries_allowed:
                    logger.info("Retrying after error...")
                    retry_count += 1
                    continue
                else:
                    logger.error(f"Max retries reached after error - returning error response")
                    return {
                        "result": f"Error generating answer after {retry_count} attempts: {str(e)}",
                        "source_documents": docs,
                        "retrieval_count": len(docs),
                        "is_rag": True,
                        "decision": {
                            "action": "reject",
                            "reason": f"Generation error: {str(e)[:100]}",
                            "accepted": False
                        },
                        "retry_count": retry_count
                    }

        # =========================================
        # STEP 7: BUILD FINAL OUTPUT
        # =========================================
        if generation_result is None:
            # This shouldn't happen but handle gracefully
            logger.warning("No generation result - creating fallback response")
            generation_result = {
                "answer": "Error: Unable to generate answer",
                "context": "",
                "ml_validation": None,
                "confidence_score": 0.0,
                "decision": {
                    "action": "reject",
                    "reason": "No generation result",
                    "accepted": False
                }
            }
        
        # Build final output with all relevant data
        context = generation_result.get("context", "")
        answer = generation_result.get("answer", "")
        ml_validation = generation_result.get("ml_validation")
        decision = generation_result.get("decision", {})
        
        # =========================================
        # STEP 6: GET TRAINED MODEL PREDICTIONS (NEW)
        # =========================================
        trained_predictions = self._get_trained_predictions(
            ml_validation=ml_validation,
            num_docs=len(docs),
            context_length=len(context)
        )
        
        # =========================================
        # STEP 8: COMPUTE RETRIEVAL METRICS (NEW)
        # =========================================
        retrieval_metrics = self._compute_retrieval_metrics(
            docs=docs,
            context=context
        )
        
        # =========================================
        # STEP 7: LOG TO TRAINING DATA FOR FEEDBACK LOOP (NEW)
        # =========================================
        self._log_to_training_data(
            query=query,
            answer=answer,
            context=context,
            ml_validation=ml_validation,
            decision=decision,
            trained_predictions=trained_predictions,
            retrieval_metrics=retrieval_metrics,
            num_docs=len(docs),
            retry_count=retry_count
        )
        
        # Build final output with all relevant data
        output = {
            "result": answer,
            "source_documents": docs,
            "retrieval_count": len(docs),
            "is_rag": True,
            # Metadata for evaluation and learning
            "metadata": {
                "context_length": len(context),
                "num_docs": len(docs),
                "generated_answer_length": len(answer),
                "retrieval_model": "mmr",
                "retriever_k_final": current_retriever_k  # K used for final answer
            },
            # ML Validation scores
            "validation": ml_validation,
            "confidence_score": generation_result.get("confidence_score", 1.0),
            # NEW: Trained model predictions
            "trained_predictions": trained_predictions,
            # NEW: Retrieval quality metrics
            "retrieval_metrics": retrieval_metrics,
            # Decision engine result
            "decision": decision,
            # Retry information for learning
            "retry_count": retry_count
        }
        
        logger.info(
            f"RAG invocation complete: "
            f"decision={output['decision'].get('action')}, "
            f"retries={retry_count}, "
            f"confidence={output['confidence_score']:.3f}, "
            f"trained_predictions_available={bool(trained_predictions)}"
        )
        
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

