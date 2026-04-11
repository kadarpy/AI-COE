"""
USAGE EXAMPLES - RAG Evaluation Platform (Refactored)
=====================================================

These examples show how to use the new refactored architecture.
"""

# ============================================================================
# EXAMPLE 1: Simple RAG (No Evaluation)
# ============================================================================
"""
Use case: Just need answers from RAG, no evaluation metrics
"""

def simple_rag_example():
    from rag.vector_store import build_vector_store
    from rag.pure_rag import build_pure_rag

    # Initialize vector store
    vectordb = build_vector_store()

    # Build pure RAG
    rag = build_pure_rag(vectordb)

    # Run RAG
    query = "What is machine learning?"
    result = rag.invoke(query)

    # Result contains:
    # - answer: Generated answer
    # - documents: Retrieved documents
    # - metadata: {"num_docs": 3, "context_length": 1234}

    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['documents'])} documents")
    return result


# ============================================================================
# EXAMPLE 2: RAG + Evaluation (Recommended)
# ============================================================================
"""
Use case: Generate answer AND evaluate its quality with metrics
"""

def rag_with_evaluation_example():
    from rag.vector_store import build_vector_store
    from orchestration import create_pipeline

    # Initialize vector store
    vectordb = build_vector_store()

    # Create pipeline (automatic RAG + Evaluation)
    pipeline = create_pipeline(
        vectordb,
        enable_ml_metrics=True,
        enable_deepeval=True,
        enable_retry=False,
        enable_training_logging=False
    )

    # Run pipeline
    query = "What is machine learning?"
    result = pipeline.run(query)

    # Result contains:
    # - result: {"answer": str, "documents": [...], "metadata": {...}}
    # - evaluation: {"deepeval_metrics": {...}, "ml_metrics": {...}}
    # - pipeline_metadata: {"retry_count": 0, "total_time_ms": 123.45}

    print(f"Answer: {result['result']['answer']}")
    print(f"Confidence: {result['evaluation']['ml_metrics']['confidence_score']:.2%}")
    print(f"Hallucination Score: {result['evaluation']['deepeval_metrics']['Hallucination']['score']}")
    return result


# ============================================================================
# EXAMPLE 3: Manual Layer-by-Layer (Advanced)
# ============================================================================
"""
Use case: Fine-grained control over each layer
"""

def manual_layers_example():
    from rag.vector_store import build_vector_store
    from rag.pure_rag import build_pure_rag
    from evaluation.pure_evaluator import evaluate

    # Initialize
    vectordb = build_vector_store()
    rag = build_pure_rag(vectordb)

    # STEP 1: RAG only
    query = "What is machine learning?"
    rag_result = rag.invoke(query)

    # STEP 2: Evaluation only
    eval_result = evaluate(
        question=query,
        answer=rag_result["answer"],
        context=[doc.page_content for doc in rag_result["documents"]],
        expected_answer="Machine learning is a subset of AI..."
    )

    # STEP 3: Combine results
    combined = {
        "answer": rag_result["answer"],
        "documents": rag_result["documents"],
        "evaluation": eval_result,
        "metadata": rag_result["metadata"]
    }

    return combined


# ============================================================================
# EXAMPLE 4: Batch Evaluation (Multiple Test Cases)
# ============================================================================
"""
Use case: Evaluate multiple QA pairs at once
"""

def batch_evaluation_example():
    from rag.vector_store import build_vector_store
    from orchestration import create_pipeline

    # Test cases
    test_cases = [
        {
            "question": "What is machine learning?",
            "expected_answer": "Machine learning is..."
        },
        {
            "question": "Explain supervised learning",
            "expected_answer": "Supervised learning is..."
        },
        {
            "question": "What is neural networks?",
            "expected_answer": "Neural networks are..."
        }
    ]

    # Initialize
    vectordb = build_vector_store()
    pipeline = create_pipeline(vectordb)

    # Run batch evaluation
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"Evaluating test {i}/{len(test_cases)}...")
        result = pipeline.run(
            query=test_case["question"],
            expected_answer=test_case.get("expected_answer")
        )
        results.append(result)

    # Analyze results
    avg_confidence = sum(
        r["evaluation"]["ml_metrics"]["confidence_score"]
        for r in results
    ) / len(results)

    print(f"Average Confidence: {avg_confidence:.2%}")
    return results


# ============================================================================
# EXAMPLE 5: With Training Data Logging
# ============================================================================
"""
Use case: Collect training data for future model improvement
"""

def with_training_logging_example():
    from rag.vector_store import build_vector_store
    from orchestration import create_pipeline
    from pathlib import Path

    # Initialize
    vectordb = build_vector_store()
    training_path = Path("training_data.jsonl")

    # Create pipeline with logging enabled
    pipeline = create_pipeline(
        vectordb,
        enable_training_logging=True
    )

    # Run - results will be logged to training_data.jsonl
    result = pipeline.run("What is machine learning?")

    print(f"Training data logged to {training_path}")
    return result


# ============================================================================
# EXAMPLE 6: Pure Evaluation (External Answers)
# ============================================================================
"""
Use case: Evaluate answers that came from somewhere else (not RAG)
"""

def pure_evaluation_example():
    from evaluation.pure_evaluator import evaluate

    # You have answers from external source (API, other system, etc.)
    question = "What is machine learning?"
    answer = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    context = [
        "Machine learning is a subset of artificial intelligence.",
        "ML systems learn patterns from training data.",
        "Common ML techniques include supervised and unsupervised learning."
    ]
    expected_answer = "Machine learning is a subset of AI that learns from data."

    # Evaluate
    evaluation = evaluate(
        question=question,
        answer=answer,
        context=context,
        expected_answer=expected_answer
    )

    # Get metrics
    print(f"Hallucination: {evaluation['deepeval_metrics']['Hallucination']['score']}")
    print(f"Confidence: {evaluation['ml_metrics']['confidence_score']:.2%}")
    return evaluation


# ============================================================================
# EXAMPLE 7: Comparison - Old vs New
# ============================================================================
"""
How to migrate from old code to new code
"""

def old_vs_new_comparison():
    """
    OLD CODE (ANTI-PATTERN):
    ========================

    from rag.rag_chain import build_rag_chain
    from evaluation import UIEvaluator

    rag_chain = build_rag_chain(vectordb)

    # This does EVERYTHING: RAG + evaluation + decision + retry + logging
    # Hard to understand, hard to test, hard to modify
    evaluator = UIEvaluator(rag_chain)
    result = evaluator.evaluate_single_test({
        "question": "...",
        "expected_answer": "..."
    })

    # Result is a mess of mixed concerns
    print(result)


    NEW CODE (BEST PRACTICES):
    ==========================
    """

    from rag.vector_store import build_vector_store
    from orchestration import create_pipeline

    vectordb = build_vector_store()

    # Clean, composable pipeline
    pipeline = create_pipeline(vectordb)
    result = pipeline.run("What is machine learning?")

    # Result is clean and well-separated:
    # - result: RAG output (answer + documents)
    # - evaluation: Metrics (deepeval + ml)
    # - pipeline_metadata: Orchestration info

    return result


# ============================================================================
# EXAMPLE 8: Custom Evaluation (Advanced)
# ============================================================================
"""
Use case: Add custom metrics without modifying core layers
"""

def custom_metrics_example():
    from rag.vector_store import build_vector_store
    from orchestration import create_pipeline

    # Get base evaluation
    vectordb = build_vector_store()
    pipeline = create_pipeline(vectordb)
    result = pipeline.run("What is machine learning?")

    # Layer 3: Add custom metrics (don't modify core RAG or Eval)
    def add_custom_metrics(pipeline_result):
        answer = pipeline_result["result"]["answer"]

        # Custom metric 1: Answer length
        answer_length = len(answer.split())

        # Custom metric 2: Question coverage
        # (how many key terms from question are in answer)
        key_terms = ["machine", "learning", "model"]
        term_coverage = sum(1 for term in key_terms if term in answer.lower()) / len(key_terms)

        # Custom metric 3: Citation presence
        has_sources = len(pipeline_result["result"]["documents"]) > 0

        return {
            **pipeline_result,
            "custom_metrics": {
                "answer_length_words": answer_length,
                "key_term_coverage": term_coverage,
                "has_sources": has_sources
            }
        }

    enhanced_result = add_custom_metrics(result)
    return enhanced_result


# ============================================================================
# EXAMPLE 9: Error Handling
# ============================================================================
"""
Use case: Handle errors gracefully
"""

def error_handling_example():
    from rag.vector_store import build_vector_store
    from orchestration import create_pipeline

    try:
        vectordb = build_vector_store()
        pipeline = create_pipeline(vectordb)
        result = pipeline.run("")  # Empty query
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None
    except RuntimeError as e:
        print(f"RAG failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    return result


# ============================================================================
# EXAMPLE 10: Test Suite
# ============================================================================
"""
Use case: Test individual layers independently
"""

def test_suite_example():
    from rag.vector_store import build_vector_store
    from rag.pure_rag import build_pure_rag
    from evaluation.pure_evaluator import evaluate

    # Initialize once
    vectordb = build_vector_store()

    print("Testing Pure RAG...")
    rag = build_pure_rag(vectordb)
    rag_result = rag.invoke("What is machine learning?")
    assert "answer" in rag_result
    assert "documents" in rag_result
    print("✓ Pure RAG works")

    print("\nTesting Pure Evaluation...")
    eval_result = evaluate(
        question="What is ML?",
        answer="ML is artificial intelligence",
        context=["ML is a subset of AI"],
        expected_answer="ML is AI"
    )
    assert "deepeval_metrics" in eval_result
    assert "ml_metrics" in eval_result
    print("✓ Pure Evaluation works")

    print("\nTesting Pipeline...")
    from orchestration import create_pipeline
    pipeline = create_pipeline(vectordb)
    pipeline_result = pipeline.run("What is machine learning?")
    assert "result" in pipeline_result
    assert "evaluation" in pipeline_result
    print("✓ Pipeline works")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    import sys

    print("RAG Evaluation Platform - Usage Examples")
    print("=" * 50)
    print("\nChoose an example:")
    print("1. Simple RAG")
    print("2. RAG + Evaluation (Recommended)")
    print("3. Manual Layers")
    print("4. Batch Evaluation")
    print("5. With Training Logging")
    print("6. Pure Evaluation")
    print("7. Old vs New")
    print("8. Custom Metrics")
    print("9. Error Handling")
    print("10. Test Suite")

    # Uncomment to run example:
    # simple_rag_example()
    # rag_with_evaluation_example()
    # batch_evaluation_example()
