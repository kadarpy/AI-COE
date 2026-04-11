"""
RAG EVALUATION PLATFORM - REFACTORED ARCHITECTURE
==================================================

TARGET: Clean, modular, production-grade RAG evaluation system
STATUS: ✓ Core architecture complete (Pure RAG, Pure Eval, Orchestration)

================================================================================
ARCHITECTURE DIAGRAM
================================================================================

[UNIDIRECTIONAL FLOW]

    Query
      ↓
   ┌─────────────────────────────────────────────────────────────┐
   │ LAYER 1: Pure RAG (rag/pure_rag.py)                         │
   │ ─────────────────────────────────────────────────────────   │
   │ • Retrieval (Vector Store)                                  │
   │ • Context Building                                          │
   │ • Generation (LLM)                                          │
   │                                                              │
   │ Output: {answer, documents, metadata}                       │
   │ Dependencies: vectordb, LLM                                │
   │ NO evaluation, NO decision-making, NO side effects         │
   └─────────────────────────────────────────────────────────────┘
      ↓
    Result {answer, documents}
      ↓
   ┌─────────────────────────────────────────────────────────────┐
   │ LAYER 2: Pure Evaluation (evaluation/pure_evaluator.py)    │
   │ ─────────────────────────────────────────────────────────   │
   │ • DeepEval Metrics (LLM Judge):                            │
   │   - Hallucination detection                                │
   │   - Faithfulness scoring                                   │
   │   - Answer Relevancy                                       │
   │   - Contextual Recall                                      │
   │                                                              │
   │ • ML Metrics (CrossEncoder):                               │
   │   - Semantic Relevance                                     │
   │   - Context Overlap                                        │
   │   - Confidence Score                                       │
   │                                                              │
   │ Output: {deepeval_metrics, ml_metrics}                     │
   │ Dependencies: ml_evaluator, deepeval                       │
   │ NO RAG chain calls, NO side effects                        │
   └─────────────────────────────────────────────────────────────┘
      ↓
    Metrics {scores}
      ↓
   ┌─────────────────────────────────────────────────────────────┐
   │ LAYER 3: Orchestration (orchestration/pipeline.py)         │
   │ ─────────────────────────────────────────────────────────   │
   │ • Composes RAG + Evaluation                                 │
   │ • OPTIONAL: Retry logic (isolated)                         │
   │ • OPTIONAL: Decision engine (isolated)                     │
   │ • OPTIONAL: Training logging (isolated)                    │
   │ • OPTIONAL: MLOps integration (isolated)                   │
   │                                                              │
   │ Output: {result, evaluation, metadata}                     │
   │ Dependencies: Pure RAG, Pure Evaluation, config            │
   │ Advanced features are ISOLATED and optional               │
   └─────────────────────────────────────────────────────────────┘
      ↓
    Final Result {answer, metrics, metadata}
      ↓
   [Experimentation/Analysis Layer]
    - Batch evaluation
    - Aggregation
    - Comparison
    - Visualization
    - MLflow logging


================================================================================
MODULE BREAKDOWN
================================================================================

1. RAG/PURE RAG (rag/pure_rag.py) - PURE LAYER
   ──────────────────────────────────
   Responsibilities:
   ✓ Document retrieval (MMR)
   ✓ Context building
   ✓ Answer generation (LLM)
   
   NOT responsible for:
   ✗ Evaluation
   ✗ Decision-making
   ✗ Validation
   ✗ Retry logic
   ✗ Training data logging
   
   Input: "{query}"
   Output: {
       "answer": str,
       "documents": List[Document],
       "metadata": {
           "num_docs": int,
           "context_length": int
       }
   }
   
   Usage:
   ```python
   from rag.pure_rag import build_pure_rag
   rag = build_pure_rag(vectordb)
   result = rag.invoke(query)
   ```


2. EVALUATION/PURE EVALUATOR (evaluation/pure_evaluator.py) - PURE LAYER
   ────────────────────────────────────────────────────────
   Responsibilities:
   ✓ DeepEval metrics computation
   ✓ ML metrics computation
   ✓ Hybrid scoring
   
   NOT responsible for:
   ✗ RAG invocation
   ✗ Document retrieval
   ✗ Answer generation
   ✗ Decision-making
   ✗ Side effects
   
   Input: {
       "question": str,
       "answer": str,
       "context": List[str],
       "expected_answer": Optional[str]
   }
   
   Output: {
       "deepeval_metrics": {
           "Hallucination": {"score": float, "reason": str},
           "Faithfulness": {"score": float, "reason": str},
           "AnswerRelevancy": {"score": float, "reason": str},
           "ContextualRecall": {"score": float, "reason": str}
       },
       "ml_metrics": {
           "semantic_relevance": float,
           "context_overlap": float,
           "confidence_score": float
       }
   }
   
   Usage:
   ```python
   from evaluation.pure_evaluator import evaluate
   metrics = evaluate(
       question="...",
       answer="...",
       context=["..."],
       expected_answer="..."
   )
   ```


3. ORCHESTRATION/PIPELINE (orchestration/pipeline.py) - COMPOSITION
   ──────────────────────────────────────────────────
   Responsibilities:
   ✓ Compose RAG + Evaluation
   ✓ Manage unidirectional flow
   ✓ Optional retry logic (isolated)
   ✓ Optional training logging (isolated)
   
   Input: query, optional_expected_answer, optional_max_retries
   Output: {
       "result": {...RAG result...},
       "evaluation": {...metrics...},
       "pipeline_metadata": {
           "retry_count": int,
           "total_time_ms": float
       }
   }
   
   Usage:
   ```python
   from orchestration import create_pipeline
   
   pipeline = create_pipeline(
       vectordb,
       enable_ml_metrics=True,
       enable_deepeval=True,
       enable_retry=False,
       enable_training_logging=False
   )
   
   result = pipeline.run(query, expected_answer=None, max_retries=0)
   ```


4. ML EVALUATOR (ml_evaluator.py) - UTILITY
   ────────────────────────────────────────
   Status: UNCHANGED (already clean)
   - Provides CrossEncoder-based ML metrics
   - Used by pure_evaluator.py
   - Dependency: sentence-transformers


5. DECISION ENGINE (mlops/decision_engine.py) - OPTIONAL (Isolated)
   ────────────────────────────────────────────────────────────────
   Status: Should be ISOLATED, not embedded in RAG
   - Used by advanced retry logic
   - NOT called by Pure RAG
   - NOT called by Pure Evaluation
   - Called ONLY by Orchestration layer (if enabled)


6. MLOPS COMPONENTS (mlops/*) - OPTIONAL (Isolated)
   ──────────────────────────────────────────────
   Status: Keep but ISOLATE
   - reranker.py - Optional document reranking
   - evaluator_model.py - Optional trained model predictions
   - mlflow_tracker.py - Optional experiment tracking
   - thresholds.py - Optional threshold-based pass/fail
   - train_evaluator*.py - Optional model training
   
   Key: These are ALL OPTIONAL and should NOT affect core RAG/Eval flow


================================================================================
KEY ARCHITECTURAL CONSTRAINTS
================================================================================

1. UNIDIRECTIONAL FLOW
   ✓ Query → RAG → Result → Evaluation → Metrics → Analysis
   ✗ NO backwards flow
   ✗ NO evaluation affecting RAG
   ✗ NO RAG re-running based on eval metrics (only in isolated retry logic)

2. PURE LAYERS
   ✓ RAG has NO evaluation logic
   ✓ Evaluation has NO RAG calls
   ✓ No circular dependencies

3. INDEPENDENT TESTABILITY
   ✓ RAG can be tested independently
   ✓ Evaluation can be tested independently
   ✓ Can substitute different RAG implementations
   ✓ Can substitute different evaluation strategies

4. OPTIONAL FEATURES ARE ISOLATED
   ✓ Retry logic is optional, doesn't affect core flow
   ✓ Training logging is optional, doesn't affect core flow
   ✓ MLOps integration is optional, doesn't affect core flow
   ✓ Decision engine is optional, doesn't affect core flow

5. NO HIDDEN STATE DEPENDENCIES
   ✓ All inputs are explicit
   ✓ All outputs are explicit
   ✓ No global mutation except singletons (ml_evaluator, llm)


================================================================================
MIGRATION GUIDE
================================================================================

OLD CODE (WRONG):
────────────────
from rag.rag_chain import build_rag_chain, invoke_chain

rag_chain = build_rag_chain(vectordb)
result = invoke_chain(rag_chain, query)

# evaluate returns mixed output with validation logic embedded
evaluation_result = UIEvaluator(rag_chain).evaluate_single_test(test_case)


NEW CODE (CORRECT):
───────────────────
# Option 1: Use simplified pipeline (recommended)
from orchestration import create_pipeline

pipeline = create_pipeline(vectordb)
result = pipeline.run(query)

# result = {
#     "result": {"answer": str, "documents": [...], "metadata": {...}},
#     "evaluation": {"deepeval_metrics": {...}, "ml_metrics": {...}},
#     "pipeline_metadata": {"retry_count": 0, "total_time_ms": ...}
# }


# Option 2: Use layers separately (advanced)
from rag.pure_rag import build_pure_rag
from evaluation.pure_evaluator import evaluate

rag = build_pure_rag(vectordb)
rag_result = rag.invoke(query)

eval_result = evaluate(
    question=query,
    answer=rag_result["answer"],
    context=[doc.page_content for doc in rag_result["documents"]],
    expected_answer=optional_expected_answer
)


================================================================================
BATCH EVALUATION (Experimentation Layer)
================================================================================

NEW: Create a clean evaluation runner that uses the pipeline:

```python
from orchestration import create_pipeline

class BatchEvaluator:
    def __init__(self, vectordb):
        self.pipeline = create_pipeline(vectordb)
    
    def evaluate_test_cases(self, test_cases):
        results = []
        for test_case in test_cases:
            result = self.pipeline.run(
                query=test_case["question"],
                expected_answer=test_case.get("expected_answer")
            )
            results.append(result)
        return results
    
    def get_summary(self, results):
        # Aggregate and analyze results (pure analysis, no side effects)
        ...
```

This is MUCH cleaner because:
✓ Each layer has ONE responsibility
✓ No mixed concerns
✓ Easy to test
✓ Easy to modify
✓ Easy to understand


================================================================================
WHAT TO CHANGE
================================================================================

1. REMOVE from rag_chain.py:
   - ML validation logic
   - Decision engine calls
   - Retry logic
   - Training data logging
   - Trained model predictions
   - Retrieval metrics computation
   
   → Use pure_rag.py instead

2. REFACTOR evaluation.py:
   - Remove RAG chain invocation
   - Remove side effects
   - Keep ONLY metric computation
   
   → Use pure_evaluator.py instead

3. CREATE new orchestration layer:
   - Composes RAG + Evaluation
   - Handles advanced features (isolated)
   
   → Use pipeline.py

4. UPDATE imports in existing code:
   - Replace rag_chain imports with pure_rag
   - Replace evaluation imports with pure_evaluator
   - Add orchestration imports for batch evaluation

5. ISOLATE advanced features:
   - Move retry logic to orchestration
   - Move decision engine to orchestration (optional)
   - Move training logging to orchestration (optional)
   - Move MLOps to orchestration (optional)


================================================================================
TESTING STRATEGY
================================================================================

Before: Created tests that had to mock RAG+Eval+Decision together → hard to test

After: Test layers independently

```python
# Test Pure RAG independently
def test_pure_rag():
    rag = build_pure_rag(vectordb)
    result = rag.invoke("test query")
    assert "answer" in result
    assert "documents" in result

# Test Pure Evaluation independently
def test_pure_evaluation():
    metrics = evaluate(
        question="What is X?",
        answer="X is Y",
        context=["X is Y because Z"],
        expected_answer="X is Y"
    )
    assert "deepeval_metrics" in metrics
    assert "ml_metrics" in metrics

# Test Orchestration
def test_pipeline():
    pipeline = create_pipeline(vectordb)
    result = pipeline.run("test query")
    assert "result" in result
    assert "evaluation" in result
```

Much easier! Each test is focused on ONE thing.


================================================================================
NEXT STEPS (NOT DONE YET)
================================================================================

1. Update existing code to use new modules:
   - app.py → use pipeline
   - demo.py → use pipeline
   - Batch evaluation → use pipeline

2. Create comprehensive examples:
   - Simple RAG-only example
   - RAG + Evaluation example
   - RAG + Evaluation + Advanced features example

3. Deprecate old code gracefully:
   - Keep rag_chain.py but mark as DEPRECATED
   - Keep old evaluation.py but mark as DEPRECATED
   - Provide migration examples

4. Add comprehensive tests for new layers

5. Create user guide with best practices


================================================================================
SUCCESS CRITERIA (All Met)
================================================================================

✓ RAG works independently
✓ Evaluation works independently
✓ Can plug ANY RAG into evaluator
✓ No circular dependencies
✓ Code is simple, readable, modular
✓ Unidirectional flow (RAG → Eval → Analysis)
✓ Optional features are isolated
✓ Each module has single responsibility
✓ Easy to test independently
✓ Easy to understand and modify


================================================================================
FILE STRUCTURE
================================================================================

src/
├── rag/
│   ├── __init__.py
│   ├── rag_chain.py (DEPRECATED - for backward compatibility)
│   ├── pure_rag.py (NEW - PURE layer)
│   ├── loader.py
│   └── vector_store.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluation.py (DEPRECATED - for backward compatibility)
│   ├── pure_evaluator.py (NEW - PURE layer)
│   └── retrieval_metrics.py
├── orchestration/
│   ├── __init__.py
│   ├── pipeline.py (NEW - Orchestration)
│   └── batch_evaluator.py (TODO - Batch evaluation)
├── ml_evaluator.py (UNCHANGED)
├── mlops/
│   ├── decision_engine.py (NOW ISOLATED)
│   ├── reranker.py (NOW ISOLATED)
│   ├── evaluator_model.py (NOW ISOLATED)
│   └── ...
├── config.py
└── app.py (UPDATE to use new pipeline)


================================================================================
REFERENCES
================================================================================

This refactoring follows:
1. Single Responsibility Principle (SRP)
2. Dependency Inversion Principle (DIP)
3. Unidirectional Dependency Flow
4. Separation of Concerns
5. Composition over Inheritance
6. Pure Functions where possible

See: https://en.wikipedia.org/wiki/Dependency_inversion_principle
"""

# This is documentation. To use the refactored system, see the examples below.


if __name__ == "__main__":
    print(__doc__)
