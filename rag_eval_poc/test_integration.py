"""
Integration Test Suite - Validates all enhancements
This script verifies that all new features integrate correctly without breaking existing functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all new modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    try:
        print(" Importing config...")
        from config import config
        assert hasattr(config, 'DEFAULT_DOCUMENT_PATH'), "Missing DEFAULT_DOCUMENT_PATH"
        print("   DEFAULT_DOCUMENT_PATH is defined")
        
        print(" Importing evaluation...")
        from evaluation import UIEvaluator, analyze_failure, EvaluationMetrics
        print("   analyze_failure imported")
        print("   UIEvaluator imported")
        print("   EvaluationMetrics imported")
        
        print(" Importing analysis_report...")
        from analysis_report import AnalysisReport
        print("   AnalysisReport imported")
        
        print(" Importing evaluation_findings_generator...")
        from evaluation_findings_generator import generate_evaluation_findings
        print("   generate_evaluation_findings imported")
        
        print(" Importing demo...")
        from demo import RAGBotDemo
        print("   RAGBotDemo imported")
        
        print(" Importing rag_chain...")
        from rag.rag_chain import RAGChain, get_llm
        print("   RAGChain imported")
        print("   get_llm imported")
        
        print(" Importing validators...")
        from validators import InputValidator, OutputValidator
        print("   Validators imported")
        
        return True
    except Exception as e:
        print(f" Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze_failure_function():
    """Test the analyze_failure function logic"""
    print("\n" + "="*60)
    print("TEST 2: Failure Analysis Function")
    print("="*60)
    
    try:
        from evaluation import analyze_failure
        
        # Test case 1: Hallucination
        print("\n  Test Case 1: Hallucination Detection")
        result = analyze_failure(
            question="What is water scarcity?",
            expected_answer="Water scarcity occurs when freshwater is insufficient.",
            actual_answer="Water scarcity was discovered by Dr. John Smith in 1987.",
            metrics={
                "Hallucination": {"score": 0.8},
                "Faithfulness": {"score": 0.3},
                "AnswerRelevancy": {"score": 0.5},
                "ContextualRecall": {"score": 0.2},
                "ExactMatch": {"score": 0.0}
            },
            retrieval_context=["Water scarcity info"]
        )
        assert result["failure_type"] == "hallucination", f"Expected hallucination, got {result['failure_type']}"
        print("     Correctly identified hallucination")
        
        # Test case 2: Correct answer
        print("\n  Test Case 2: Correct Answer Detection")
        result = analyze_failure(
            question="What is water scarcity?",
            expected_answer="Water scarcity occurs when freshwater is insufficient.",
            actual_answer="Water scarcity occurs when freshwater is insufficient.",
            metrics={
                "Hallucination": {"score": 0.1},
                "Faithfulness": {"score": 0.95},
                "AnswerRelevancy": {"score": 0.95},
                "ContextualRecall": {"score": 0.9},
                "ExactMatch": {"score": 1.0}
            },
            retrieval_context=["Water scarcity info"]
        )
        assert result["failure_type"] == "correct", f"Expected correct, got {result['failure_type']}"
        print("     Correctly identified correct answer")
        
        # Test case 3: Unanswerable question handled correctly
        print("\n  Test Case 3: Unanswerable Question Handling")
        result = analyze_failure(
            question="Who discovered water scarcity?",
            expected_answer="The documents do not contain this information.",
            actual_answer="The documents do not contain this information.",
            metrics={
                "Hallucination": {"score": 0.1},
                "Faithfulness": {"score": 0.9},
                "AnswerRelevancy": {"score": 1.0},
                "ContextualRecall": {"score": 0.8},
                "ExactMatch": {"score": 1.0}
            },
            retrieval_context=["Water info"]
        )
        assert result["failure_type"] == "correct", f"Expected correct, got {result['failure_type']}"
        print("     Correctly identified proper unanswerable handling")
        
        # Test case 4: Unanswerable answered as hallucination
        print("\n  Test Case 4: Hallucination on Unanswerable")
        result = analyze_failure(
            question="Who discovered water scarcity?",
            expected_answer="The documents do not contain this information.",
            actual_answer="Dr. John Smith discovered water scarcity in 1987.",
            metrics={
                "Hallucination": {"score": 0.9},
                "Faithfulness": {"score": 0.2},
                "AnswerRelevancy": {"score": 0.4},
                "ContextualRecall": {"score": 0.3},
                "ExactMatch": {"score": 0.0}
            },
            retrieval_context=["Water info"]
        )
        assert result["failure_type"] == "hallucination", f"Expected hallucination, got {result['failure_type']}"
        print("     Correctly identified hallucination on unanswerable question")
        
        print("\n All analyze_failure test cases passed")
        return True
        
    except Exception as e:
        print(f" analyze_failure test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis_report_class():
    """Test the AnalysisReport class"""
    print("\n" + "="*60)
    print("TEST 3: Analysis Report Class")
    print("="*60)
    
    try:
        from analysis_report import AnalysisReport
        
        # Create instance
        report = AnalysisReport()
        print(" AnalysisReport instance created")
        
        # Check methods exist
        assert hasattr(report, 'analyze_result'), "Missing analyze_result method"
        assert hasattr(report, 'generate_comparison_report'), "Missing generate_comparison_report method"
        assert hasattr(report, 'export_comparison_csv'), "Missing export_comparison_csv method"
        assert hasattr(report, 'export_comparison_json'), "Missing export_comparison_json method"
        print(" All required methods exist")
        
        # Test analyze_result
        test_result = {
            "test_id": 1,
            "category": "straightforward",
            "question": "What is water?",
            "expected_answer": "H2O",
            "actual_answer": "H2O",
            "metrics": {
                "Hallucination": {"score": 0.1},
                "Faithfulness": {"score": 0.9}
            },
            "failure_analysis": {
                "failure_type": "correct",
                "reason": "Answer is correct"
            }
        }
        
        row = report.analyze_result(test_result)
        assert row["test_id"] == 1, "test_id not preserved"
        assert "verdict" in row, "verdict not in result"
        assert "manual_judgment" in row, "manual_judgment not in result"
        print(" analyze_result works correctly")
        
        print("\n All AnalysisReport tests passed")
        return True
        
    except Exception as e:
        print(f" AnalysisReport test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_findings_generator():
    """Test the evaluation findings generator"""
    print("\n" + "="*60)
    print("TEST 4: Evaluation Findings Generator")
    print("="*60)

    try:
        from evaluation_findings_generator import generate_evaluation_findings

        # Create mock data
        mock_results = [
            {
                "test_id": 1,
                "question": "What is water scarcity?",
                "expected_answer": "Water scarcity is when freshwater is insufficient.",
                "actual_answer": "Water scarcity is when freshwater is insufficient.",
                "category": "straightforward",
                "overall_passed": True,
                "failure_analysis": {
                    "failure_type": "correct",
                    "reason": "Answer is correct"
                },
                "metrics": {
                    "Hallucination": {"score": 0.1},
                    "Faithfulness": {"score": 0.95}
                }
            }
        ]

        mock_summary = {
            "total_tests": 1,
            "passed_tests": 1,
            "failed_tests": 0,
            "pass_rate": 100.0,
            "failure_distribution": {"correct": 1},
            "metrics": {
                "Hallucination": {"avg_score": 0.1, "min_score": 0.1, "max_score": 0.1},
                "Faithfulness": {"avg_score": 0.95, "min_score": 0.95, "max_score": 0.95}
            }
        }

        # Generate report
        output_file = Path("test_outputs") / "test_findings.md"
        result_file = generate_evaluation_findings(mock_results, mock_summary, output_file)

        assert result_file is not None, "No output file returned"
        print(f"   Report generated: {result_file}")

        # Check file exists
        if Path(result_file).exists():
            file_size = Path(result_file).stat().st_size
            assert file_size > 0, "Generated file is empty"
            print(f"   File exists with {file_size} bytes")

        print("\n All evaluation_findings_generator tests passed")
        return True

    except Exception as e:
        print(f" evaluation_findings_generator test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config_has_defaults():
    """Test that config has all required defaults"""
    print("\n" + "="*60)
    print("TEST 5: Configuration Defaults")
    print("="*60)
    
    try:
        from config import config
        
        required_attrs = [
            'DEFAULT_DOCUMENT_PATH',
            'EVAL_TEST_CASES_PATH',
            'DOC_CHUNK_SIZE',
            'DOC_CHUNK_OVERLAP',
            'RETRIEVER_K',
            'LLM_MODEL',
            'API_KEY'
        ]
        
        for attr in required_attrs:
            assert hasattr(config, attr), f"Missing config attribute: {attr}"
            print(f" config.{attr} exists")
        
        print("\n All configuration defaults present")
        return True
        
    except Exception as e:
        print(f" Configuration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_test_cases_loaded():
    """Test that test cases can be loaded"""
    print("\n" + "="*60)
    print("TEST 6: Test Cases Loading")
    print("="*60)
    
    try:
        from evaluation import TestCaseManager
        from pathlib import Path
        
        manager = TestCaseManager()
        test_cases = manager.load_test_cases()
        
        assert test_cases is not None, "Test cases is None"
        assert len(test_cases) > 0, "No test cases loaded"
        print(f" Loaded {len(test_cases)} test cases")
        
        # Check for unanswerable cases
        unanswerable = [tc for tc in test_cases if tc.get("category") == "unanswerable"]
        assert len(unanswerable) >= 9, f"Expected at least 9 unanswerable cases, got {len(unanswerable)}"
        print(f" Found {len(unanswerable)} unanswerable test cases (including new ones 31-35)")
        
        # Check for IDs 31-35
        new_ids = [31, 32, 33, 34, 35]
        for test_id in new_ids:
            case = manager.get_test_case_by_id(test_id)
            assert case is not None, f"Test case {test_id} not found"
        print(f" All new test cases (31-35) present")
        
        print("\n All test case loading tests passed")
        return True
        
    except Exception as e:
        print(f" Test case loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RAG EVALUATION POC - INTEGRATION TEST SUITE")
    print("="*60)
    print("Testing all enhancements and validating integration...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Failure Analysis Function", test_analyze_failure_function),
        ("Analysis Report Class", test_analysis_report_class),
        ("Evaluation Findings Generator", test_evaluation_findings_generator),
        ("Configuration Defaults", test_config_has_defaults),
        ("Test Cases Loading", test_test_cases_loaded)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n Test '{test_name}' crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {passed}/{total} tests passed ({pass_rate:.1f}%)")
    print(f"{'='*60}")
    
    if passed == total:
        print("\n ALL TESTS PASSED - INTEGRATION SUCCESSFUL \n")
        return 0
    else:
        print(f"\n {total - passed} TESTS FAILED \n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
