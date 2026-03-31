"""
Evaluation Findings Report Generator - Create comprehensive markdown findings
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_evaluation_findings(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_file: Optional[str] = None
) -> str:
    """
    Generate comprehensive evaluation findings markdown report
    
    Includes:
    1. Summary statistics
    2. Key insights about metric performance
    3. Failure case studies (3-5 examples)
    4. Conclusion answering key question
    """
    
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_findings_{timestamp}.md"

    # =========================
    # EXTRACT KEY DATA
    # =========================
    
    total_tests = summary.get("total_tests", 0)
    passed_tests = summary.get("passed_tests", 0)
    failed_tests = summary.get("failed_tests", 0)
    pass_rate = summary.get("pass_rate", 0)
    failure_dist = summary.get("failure_distribution", {})
    
    # =========================
    # BUILD REPORT
    # =========================
    
    report = []
    report.append("# Evaluation Findings Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # =====================================================================
    # SECTION 1: SUMMARY
    # =====================================================================
    
    report.append("## 1. Executive Summary")
    report.append("")
    report.append(f"- **Total Tests Run:** {total_tests}")
    report.append(f"- **Tests Passed:** {passed_tests} ({pass_rate:.1f}%)")
    report.append(f"- **Tests Failed:** {failed_tests} ({100-pass_rate:.1f}%)")
    report.append("")
    
    # =====================================================================
    # SECTION 2: FAILURE DISTRIBUTION
    # =====================================================================
    
    report.append("### Failure Distribution")
    report.append("")
    
    if failure_dist:
        for failure_type, count in sorted(failure_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_tests * 100) if total_tests > 0 else 0
            report.append(f"- **{failure_type.title()}:** {count} ({percentage:.1f}%)")
    else:
        report.append("- All tests passed successfully")
    
    report.append("")
    
    # =====================================================================
    # SECTION 3: KEY INSIGHTS
    # =====================================================================
    
    report.append("## 2. Key Insights About Metric Performance")
    report.append("")
    
    insights = _generate_insights(results, summary)
    
    for i, insight in enumerate(insights, 1):
        report.append(f"### {i}. {insight['title']}")
        report.append("")
        report.append(insight['description'])
        report.append("")
    
    # =====================================================================
    # SECTION 4: HALLUCINATION ANALYSIS
    # =====================================================================
    
    unanswerable_results = [r for r in results if r.get("category") == "unanswerable"]
    
    report.append("## 3. Hallucination Detection Analysis")
    report.append("")
    
    if unanswerable_results:
        hallucination_count = sum(
            1 for r in unanswerable_results 
            if r.get("failure_analysis", {}).get("failure_type") == "hallucination"
        )
        hallucination_rate = (hallucination_count / len(unanswerable_results) * 100)
        
        report.append(f"- **Unanswerable Questions:** {len(unanswerable_results)}")
        report.append(f"- **Hallucinations Detected:** {hallucination_count} ({hallucination_rate:.1f}%)")
        report.append(f"- **Correct Refusals:** {len(unanswerable_results) - hallucination_count} ({100-hallucination_rate:.1f}%)")
        report.append("")
    
    # =====================================================================
    # SECTION 5: CASE STUDIES
    # =====================================================================
    
    report.append("## 4. Failure Case Studies")
    report.append("")
    report.append("Selected examples of issues detected by evaluation:")
    report.append("")
    
    # Get interesting failure cases
    case_studies = _select_case_studies(results)
    
    for i, case in enumerate(case_studies, 1):
        report.append(f"### Case {i}: {case['failure_type'].title()}")
        report.append("")
        report.append(f"**Question:** {case['question']}")
        report.append("")
        report.append(f"**Expected Answer:** {case['expected_answer']}")
        report.append("")
        report.append(f"**Actual Answer:** {case['actual_answer']}")
        report.append("")
        report.append(f"**Metric Scores:**")
        report.append("")
        
        for metric_name, score in case['metrics'].items():
            if score is not None:
                threshold = case.get('thresholds', {}).get(metric_name, 'N/A')
                status = " PASS" if score >= (threshold if isinstance(threshold, (int, float)) else 0.7) else " FAIL"
                report.append(f"- {metric_name}: {score:.2f} {status}")
        
        report.append("")
        report.append(f"**Analysis:**")
        report.append("")
        report.append(case['failure_analysis']['reason'])
        report.append("")
        report.append(f"**Key Takeaway:** {case['key_takeaway']}")
        report.append("")
    
    # =====================================================================
    # SECTION 6: CONCLUSION
    # =====================================================================
    
    report.append("## 5. Conclusion")
    report.append("")
    report.append("### Key Question: Does Structured LLM Evaluation Catch Issues Manual Testing Doesn't?")
    report.append("")
    
    conclusion = _generate_conclusion(results, summary)
    report.append(conclusion)
    report.append("")
    
    # =====================================================================
    # SECTION 7: RECOMMENDATIONS
    # =====================================================================
    
    report.append("## 6. Recommendations")
    report.append("")
    
    recommendations = _generate_recommendations(results, summary)
    for rec in recommendations:
        report.append(f"- {rec}")
    
    report.append("")
    
    # =====================================================================
    # WRITE TO FILE
    # =====================================================================
    
    report_text = "\n".join(report)

    # Ensure directory exists
    from pathlib import Path

    output_path = Path(output_file)

    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Evaluation findings report written to {output_file}")
    return output_file


def _generate_insights(results: List[Dict], summary: Dict) -> List[Dict[str, str]]:
    """Generate key insights about metric performance"""
    
    insights = []
    
    # Analyze which failure types are most common
    failure_dist = summary.get("failure_distribution", {})
    metrics = summary.get("metrics", {})
    
    if failure_dist:
        most_common = max(failure_dist.items(), key=lambda x: x[1])
        insights.append({
            "title": "Most Common Failure Type",
            "description": f"**{most_common[0].title()}** is the most frequently detected issue ({most_common[1]} cases). "
                          f"This indicates {_interpret_failure_type(most_common[0])} is a significant quality challenge."
        })
    
    # Analyze metric performance
    if metrics:
        hallucination = metrics.get("Hallucination", {})
        faithfulness = metrics.get("Faithfulness", {})
        relevancy = metrics.get("AnswerRelevancy", {})
        
        insights.append({
            "title": "Metric Performance",
            "description": (
                f"**Hallucination metric** (avg: {hallucination.get('avg_score', 0):.2f}): "
                f"{'Strong at catching fabrications' if hallucination.get('avg_score', 0) > 0.3 else 'May allow some hallucinations'}. "
                f"**Faithfulness metric** (avg: {faithfulness.get('avg_score', 0):.2f}): "
                f"{'Strong grounding to source documents' if faithfulness.get('avg_score', 0) > 0.7 else 'Needs improvement'}. "
                f"**Relevance metric** (avg: {relevancy.get('avg_score', 0):.2f}): "
                f"{'Good answer-to-question alignment' if relevancy.get('avg_score', 0) > 0.7 else 'May flag irrelevant answers'}."
            )
        })
    
    # Assess unanswerable question handling
    unanswerable = [r for r in results if r.get("category") == "unanswerable"]
    if unanswerable:
        hallucinations = sum(1 for r in unanswerable if r.get("failure_analysis", {}).get("failure_type") == "hallucination")
        correct_refusals = len(unanswerable) - hallucinations
        
        insights.append({
            "title": "Unanswerable Question Handling",
            "description": (
                f"Model correctly refused to answer {correct_refusals}/{len(unanswerable)} unanswerable questions. "
                f"{hallucinations} cases showed hallucination (invented answers). "
                f"This {'demonstrates strong grounding' if correct_refusals > len(unanswerable) * 0.8 else 'indicates room for improvement in'} "
                f"preventing unfounded claims."
            )
        })
    
    return insights[:3]  # Top 3 insights


def _interpret_failure_type(failure_type: str) -> str:
    """Interpret what a failure type means"""
    interpretations = {
        "hallucination": "the model is making up or inferring information",
        "retrieval_miss": "context retrieval is missing relevant documents",
        "partial_answer": "the model isn't providing complete answers",
        "correct": "the model is performing well"
    }
    return interpretations.get(failure_type.lower(), "there are quality issues detected")


def _select_case_studies(results: List[Dict]) -> List[Dict]:
    """Select 3-5 interesting case studies"""
    
    case_studies = []
    
    # Priority 1: Hallucination cases (most interesting)
    hallucinations = [r for r in results if r.get("failure_analysis", {}).get("failure_type") == "hallucination"]
    case_studies.extend(hallucinations[:2])
    
    # Priority 2: Retrieval misses
    retrieval_misses = [r for r in results if r.get("failure_analysis", {}).get("failure_type") == "retrieval_miss"]
    case_studies.extend(retrieval_misses[:1])
    
    # Priority 3: Partial answers
    partial_answers = [r for r in results if r.get("failure_analysis", {}).get("failure_type") == "partial_answer"]
    case_studies.extend(partial_answers[:1])
    
    # Priority 4: If we need more, take some correct answers for contrast
    if len(case_studies) < 3:
        correct = [r for r in results if r.get("failure_analysis", {}).get("failure_type") == "correct" and r.get("overall_passed")]
        case_studies.extend(correct[:max(0, 3 - len(case_studies))])
    
    # Format case studies
    formatted = []
    for case in case_studies[:5]:
        if not case:
            continue
        
        metrics_dict = {}
        for m_name, m_data in case.get("metrics", {}).items():
            metrics_dict[m_name] = m_data.get("score")
        
        formatted.append({
            "question": case.get("question", "")[:100],
            "expected_answer": case.get("expected_answer", "")[:150],
            "actual_answer": case.get("actual_answer", "")[:150],
            "failure_type": case.get("failure_analysis", {}).get("failure_type", "unknown"),
            "failure_analysis": case.get("failure_analysis", {}),
            "metrics": metrics_dict,
            "thresholds": {"Hallucination": 0.3, "Faithfulness": 0.7, "AnswerRelevancy": 0.7},
            "key_takeaway": _generate_key_takeaway(case)
        })
    
    return formatted


def _generate_key_takeaway(case: Dict) -> str:
    """Generate key takeaway for a case study"""
    
    failure_type = case.get("failure_analysis", {}).get("failure_type", "unknown")
    reason = case.get("failure_analysis", {}).get("reason", "")
    
    if failure_type == "hallucination":
        return "Structured metrics successfully caught an answer that fabricates information not in the source documents."
    elif failure_type == "retrieval_miss":
        return "The RAG system failed to retrieve relevant context, leading to incomplete or incorrect answers."
    elif failure_type == "partial_answer":
        return "While the answer contains relevant information, it doesn't fully satisfy the question requirements."
    else:
        return "This case demonstrates how evaluation metrics validate answer quality across multiple dimensions."


def _generate_conclusion(results: List[Dict], summary: Dict) -> str:
    """Generate conclusion answering the key question"""
    
    total = summary.get("total_tests", 0)
    passed = summary.get("passed_tests", 0)
    hallucinations = sum(1 for r in results if r.get("failure_analysis", {}).get("failure_type") == "hallucination")
    
    # Determine if structured evaluation is valuable
    if hallucinations > total * 0.1:  # >10% hallucinations
        return (
            f"**YES - Structured evaluation is essential.** Despite {passed}/{total} tests passing manual inspection, "
            f"we detected {hallucinations} hallucination cases. Manual 'does this look right?' testing "
            f"would miss these subtle issues where the model sounds confident but fabricates information. "
            f"Metrics provide objective, scalable quality gates that catch problems human reviewers overlook, "
            f"especially when answers sound plausible but lack grounding in source documents."
        )
    elif passed == total:
        return (
            f"**YES - Structured evaluation provides confidence.** All {total} tests passed, confirming the bot maintains "
            f"quality across diverse question types. Without metrics, we'd only have subjective judgment that the bot "
            f"'seems okay.' Evaluation proves it systematically grounds answers in documents, refuses to answer when necessary, "
            f"and provides relevant responses consistently."
        )
    else:
        return (
            f"**YES - Structured evaluation identifies actionable quality gaps.** We found {total - passed} issues "
            f"across {total} tests. Manual testing likely wouldn't identify the root causes—whether failures stem from "
            f"hallucination, retrieval gaps, or answer irrelevance. Metrics pinpoint exactly where the bot fails, "
            f"enabling targeted improvements. This diagnostic power makes testing systematic rather than ad-hoc."
        )


def _generate_recommendations(results: List[Dict], summary: Dict) -> List[str]:
    """Generate recommendations based on findings"""
    
    recommendations = []
    failure_dist = summary.get("failure_distribution", {})
    
    # Based on failure types
    if "hallucination" in failure_dist and failure_dist["hallucination"] > 0:
        recommendations.append(
            "**Strengthen hallucination prevention**: Add explicit 'refuse to speculate' instructions to prompt. "
            "Consider penalizing out-of-context claims more heavily in evaluation thresholds."
        )
    
    if "retrieval_miss" in failure_dist and failure_dist["retrieval_miss"] > 0:
        recommendations.append(
            "**Improve retrieval quality**: Experiment with different embedding models or chunking strategies. "
            "Increase retriever_k to fetch more candidate documents before reranking."
        )
    
    if "partial_answer" in failure_dist and failure_dist["partial_answer"] > 0:
        recommendations.append(
            "**Enhance completeness**: Redesign prompts to encourage exhaustive answers using \"mention all relevant points\" framing."
        )
    
    if len(recommendations) == 0:
        recommendations.append(
            "**Maintain current setup**: The system is performing well. Continue monitoring with evaluation metrics "
            "as documents or question distributions change."
        )
    
    # General recommendations
    recommendations.append(
        "**Establish evaluation gates**: Use metric thresholds to prevent low-quality answers from reaching users. "
        "For production, target: Faithfulness ≥0.85, Hallucination ≤0.15, Relevancy ≥0.80."
    )
    
    recommendations.append(
        "**Schedule regular re-evaluation**: Run evaluation suite monthly as part of continuous quality assurance. "
        "Document metric trends to catch degradation early."
    )
    
    return recommendations[:5]  # Top 5 recommendations
