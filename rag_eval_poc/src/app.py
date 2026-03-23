import streamlit as st
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shutil
import gc

from config import config
from validators import InputValidator, OutputValidator
from demo import RAGBotDemo
from evaluation import UIEvaluator, TestCaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG Intelligence Platform",
    page_icon="",
    layout="wide",
)

# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================

def apply_professional_styling():

    st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system;
    color: #e5e7eb;
}

/* FIX: dark background instead of light */
.main {
    background: #0b1220;
}


/* HERO HEADER */

.hero {
    padding: 30px;
    border-radius: 14px;
    background: linear-gradient(135deg,#1e3a8a,#2563eb);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 34px;
}

.hero p {
    opacity: 0.9;
}


/* KPI CARDS */

.metric-card {
    background: #111827;  /* FIX */
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    color: #e5e7eb;
}

.metric-bar-card {
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;

    background: rgba(255,255,255,0.05);
    color: white;

    border: 1px solid rgba(255,255,255,0.08);
}

.metric-header {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    margin-bottom: 8px;
}

.metric-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    overflow: hidden;
}

.metric-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

.metric-footer {
    margin-top: 6px;
    font-size: 12px;
    opacity: 0.8;
}


/* CHAT WINDOW */

.chat-window {
    background: #111827;  /* FIX */
    color: #e5e7eb;
    padding: 30px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    margin-top: 15px;
}


/* USER MESSAGE */

.user-msg {
    background:#1e293b;  /* FIX */
    border-left:4px solid #2563eb;
    padding:15px;
    border-radius:8px;
    margin-bottom:10px;
    color:#e5e7eb;
}


/* AI MESSAGE */

.ai-msg {
    background:#064e3b;  /* FIX */
    border-left:4px solid #059669;
    padding:15px;
    border-radius:8px;
    margin-bottom:10px;
    color:#d1fae5;
}
/* CHAT ROW */
.chat-row {
    display: flex;
    width: 100%;
    margin-bottom: 10px;
}

/* USER (RIGHT) */
.user-row {
    justify-content: flex-end;
}

.user-msg {
    background:#1e293b;
    border-left:4px solid #2563eb;
    padding:15px;
    border-radius:12px;
    color:#e5e7eb;
    max-width: 65%;
}

/* AI (LEFT) */
.ai-row {
    justify-content: flex-start;
}

.ai-msg {
    background:#064e3b;
    border-left:4px solid #059669;
    padding:15px;
    border-radius:12px;
    color:#d1fae5;
    max-width: 65%;
}

/* SOURCE DOC PANEL */

.source-card {
    background:#111827;  /* FIX */
    border-radius:10px;
    padding:15px;
    border:1px solid #374151;
    margin-bottom:10px;
    color:#e5e7eb;
}


/* SIDEBAR */

section[data-testid="stSidebar"]{
    background:#111827;
}

section[data-testid="stSidebar"] *{
    color:white;
}


/* BUTTON */

.stButton button {
    background:#2563eb;
    border-radius:8px;
    border:none;
}

.stButton button:hover{
    background:#1d4ed8;
}


/* INPUT FIX (IMPORTANT) */

.stTextInput input {
    background-color:#111827 !important;
    color:#ffffff !important;
}

textarea {
    background-color:#111827 !important;
    color:#ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

def initialize_session():

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "bot" not in st.session_state:
        st.session_state.bot = None

    if "bot_initialized" not in st.session_state:
        st.session_state.bot_initialized = False

    if "stats" not in st.session_state:
        st.session_state.stats = {
            "total_questions": 0,
            "total_time": 0,
            "errors": 0
        }

    # Evaluation-related session state
    if "evaluation_test_cases" not in st.session_state:
        st.session_state.evaluation_test_cases = []

    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []

    if "evaluator" not in st.session_state:
        st.session_state.evaluator = None

    if "evaluation_loaded" not in st.session_state:
        st.session_state.evaluation_loaded = False

    # =========================
    # EVALUATION PROFILES (GLOBAL)
    # =========================
    if "eval_profiles" not in st.session_state:
        st.session_state.eval_profiles = {
            "poc": {
                "faithfulness": 0.70,
                "relevancy": 0.75,
                "recall": 0.70,
                "hallucination": 0.25
            },
            "strong": {
                "faithfulness": 0.85,
                "relevancy": 0.85,
                "recall": 0.80,
                "hallucination": 0.15
            },
            "production": {
                "faithfulness": 0.90,
                "relevancy": 0.90,
                "recall": 0.90,
                "hallucination": 0.05
            }
        }

    # =========================
    # DEFAULT PROFILE
    # =========================
    if "eval_profile" not in st.session_state:
        st.session_state.eval_profile = st.session_state.eval_profiles["poc"]

# =========================================================
# HEADER
# =========================================================

def display_header():

    st.markdown("""
    <div class="hero">

    <h1>RAG (Retrieval-Augmented Generation) Platform</h1>

    <p>
    Enterprise Document AI powered by Retrieval-Augmented Generation.
    Ask questions and retrieve knowledge.
    </p>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# KPI BAR
# =========================================================

def display_kpis():

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", st.session_state.stats["total_questions"])

    with col2:
        success = st.session_state.stats["total_questions"] - st.session_state.stats["errors"]
        st.metric("Successful Answers", success)

    with col3:

        avg_time = (
            st.session_state.stats["total_time"] /
            max(st.session_state.stats["total_questions"], 1)
        )

        st.metric("Avg Latency", f"{avg_time:.2f}s")

    with col4:
        st.metric("Errors", st.session_state.stats["errors"])

# =========================================================
# SIDEBAR SETTINGS
# =========================================================

def sidebar_settings():
    st.sidebar.subheader("⚙️ Settings")

    with st.sidebar.expander("LLM Config"):
        temp = st.slider("Temperature", 0.0, 1.0, float(config.TEMPERATURE))
        k = st.slider("Retriever K", 1, 10, config.RETRIEVER_K)
        st.caption(f"Temp={temp}, K={k}")

    with st.sidebar.expander("Chunk Config"):
        chunk_size = st.number_input("Chunk Size", 200, 2000, config.PDF_CHUNK_SIZE)
        overlap = st.number_input("Overlap", 0, 500, config.PDF_CHUNK_OVERLAP)

# =========================================================
# BOT INITIALIZATION
# =========================================================

def init_bot():

    try:
        with st.spinner("Initializing RAG System..."):

            bot = RAGBotDemo(pdf_path="src/data/documents/document.txt")

            if bot.setup():
                st.session_state.bot = bot
                st.session_state.bot_initialized = True
                return True

    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")

    return False

# =========================================================
# CHAT WINDOW
# =========================================================

def display_chat():

    for msg in st.session_state.conversation_history:

        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-row user-row">
                <div class="user-msg">
                    <b>You</b><br>{msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif msg["role"] == "assistant":
            st.markdown(f"""
            <div class="chat-row ai-row">
                <div class="ai-msg">
                    <b>Rag Bot</b><br>{msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sources (keep as is)
            if msg.get("sources"):
                with st.expander("Sources"):
                    for doc in msg["sources"][:3]:
                        st.write(doc.page_content[:200])

# =========================================================
# SOURCE DOCUMENT VIEW
# =========================================================

def display_sources(sources):

    if not sources:
        return

    st.subheader("Retrieved Document Context")

    for doc in sources[:3]:

        st.markdown(
            f"""
            <div class="source-card">

            <b>Source:</b> {doc.metadata.get("source","Document")}

            <hr>

            {doc.page_content[:400]}

            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# EVALUATION UI FUNCTIONS
# =========================================================

def display_metric_card(metric_name: str, score: float, threshold: float, passed: bool):

    # Normalize metric name
    name = metric_name.lower()
    # Use backend-evaluated values ONLY (no recomputation)

    if "hallucination" in name:
        # lower is better
        color = "#ef4444" if score > threshold else "#3b82f6"
    else:
        color = "#10b981" if passed else "#ef4444"

    # Layout
    col1, col2 = st.columns([3, 1])

    with col1:
        if "hallucination" in name:
            st.markdown("**Hallucination (Lower = Better)**")
        else:
            st.markdown(f"**{metric_name}**")
    with col2:
        st.markdown(f"**{'PASS' if passed else 'FAIL'}**")

    # Progress bar
    st.markdown(
        f"""
        <div style="
            width:100%;
            height:8px;
            background:rgba(255,255,255,0.1);
            border-radius:999px;
            overflow:hidden;
        ">
            <div style="
                width:{int(score*100)}%;
                height:100%;
                background:{color};
                border-radius:999px;
                transition:0.4s;
            "></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(f"Score: {score:.2f} | Threshold: {threshold:.2f}")


def display_evaluation_test_cases():
    """Display test case loader and selector"""
    st.subheader("Test Case Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Default Test Cases"):
            try:
                test_manager = TestCaseManager()
                test_manager.load_test_cases()
                st.session_state.evaluation_test_cases = test_manager.test_cases
                st.session_state.evaluation_loaded = True
                st.success(f"✓ Loaded {len(test_manager.test_cases)} test cases")
            except Exception as e:
                st.error(f"✗ Error loading test cases: {str(e)}")
    
    # Show test case breakdown
    if st.session_state.evaluation_test_cases:
        st.write("---")
        col1, col2, col3 = st.columns(3)
        
        categories = {}
        for tc in st.session_state.evaluation_test_cases:
            cat = tc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        with col1:
            st.metric("Total Cases", len(st.session_state.evaluation_test_cases))
        
        for idx, (cat, count) in enumerate(categories.items(), 1):
            if idx == 2:
                with col2:
                    st.metric(f"{cat.title()}", count)
            elif idx == 3:
                with col3:
                    st.metric(f"{cat.title()}", count)


def display_single_test_evaluation():
    """Display single test case evaluation interface"""
    st.subheader("Single Test Evaluation")
    
    if not st.session_state.evaluation_test_cases:
        st.warning("⚠ Load test cases first")
        return
    
    if not st.session_state.bot_initialized:
        st.warning("⚠ Initialize RAG bot first")
        return
    
    # Check if LLM judge is configured
    llm_configured, llm_msg = UIEvaluator._check_llm_for_metrics()
    if not llm_configured:
        st.error(f"⚠ LLM Judge Configuration Required\n\n{llm_msg}")
        return
    
    # Select test case
    test_ids = [f"Q#{tc['id']}: {tc['question'][:50]}..." 
                for tc in st.session_state.evaluation_test_cases]
    selected_idx = st.selectbox("Select Test Case", range(len(test_ids)), 
                                format_func=lambda i: test_ids[i])
    
    selected_test = st.session_state.evaluation_test_cases[selected_idx]
    
    # Display test case details
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Category:** {selected_test.get('category', 'N/A')}")
        st.write(f"**Difficulty:** {selected_test.get('difficulty', 'N/A')}")
    
    with col2:
        st.write(f"**ID:** {selected_test['id']}")
        st.write(f"**Notes:** {selected_test.get('notes', 'N/A')}")
    
    st.write("---")
    st.write(f"**Question:** {selected_test['question']}")
    st.write(f"**Expected Answer:** {selected_test['expected_answer']}")
    
    # Run evaluation
    if st.button("▶ Evaluate This Test", key="single_test_eval"):
        if st.session_state.evaluator is None:
            st.session_state.evaluator = UIEvaluator(st.session_state.bot.qa_chain)
        
        with st.spinner("Evaluating..."):
            result = st.session_state.evaluator.evaluate_single_test(selected_test)
        
        # Display result
        st.write("---")
        st.subheader("Evaluation Results")
        
        if "error" in result and result["error"]:
            st.error(f"✗ Error: {result['error']}")
        else:
            # Show actual answer
            st.write(f"**Actual Answer:** {result['actual_answer']}")
            
            # Show metrics
            st.write("---")
            st.write("**Metric Scores:**")
            
            cols = st.columns(2)
            metrics = result.get("metrics", {})
            
            for idx, (metric_name, metric_data) in enumerate(metrics.items()):
                with cols[idx % 2]:

                    score = metric_data.get("score")
                    passed = metric_data.get("passed", False)

                    if score is not None:
                        display_metric_card(
                            metric_name,
                            score,
                            metric_data.get("threshold", 0),
                            passed
                        )
                    else:
                        st.warning(f"{metric_name}: No score")
            
            # Show retrieved context
            st.write("---")
            with st.expander("Retrieved Context"):
                st.text_area("Context", result.get("context", "No context"), 
                           height=200, disabled=True)


def display_batch_evaluation():
    """Display batch evaluation interface"""
    st.subheader("Batch Evaluation")
    
    if not st.session_state.evaluation_test_cases:
        st.warning("⚠ Load test cases first")
        return
    
    if not st.session_state.bot_initialized:
        st.warning("⚠ Initialize RAG bot first")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + list(set(tc.get("category") for tc in st.session_state.evaluation_test_cases))
        )
    
    with col2:
        if category_filter == "All":
            test_cases_to_run = st.session_state.evaluation_test_cases
        else:
            test_cases_to_run = [tc for tc in st.session_state.evaluation_test_cases 
                                if tc.get("category") == category_filter]
        
        st.metric("Tests to Run", len(test_cases_to_run))
    
    with col3:
        pass
    
    if st.button("▶ Run Batch Evaluation", key="batch_eval"):
        if st.session_state.evaluator is None:
            st.session_state.evaluator = UIEvaluator(st.session_state.bot.qa_chain)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        def progress_callback(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Running: {current}/{total} tests...")
        
        # Run evaluation
        with st.spinner("Running batch evaluation..."):
            batch_results = st.session_state.evaluator.evaluate_batch(
                test_cases_to_run,
                progress_callback=progress_callback
            )
        
        # Store results
        st.session_state.evaluation_results = batch_results
        
        progress_bar.progress(1.0)
        status_text.text("✓ Evaluation complete!")
        
        # Display summary
        st.write("---")
        st.subheader("Batch Results Summary")
        
        summary = st.session_state.evaluator.get_results_summary(batch_results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tests", summary["total_tests"])
        with col2:
            st.metric("Passed", summary["passed_tests"])
        with col3:
            st.metric("Failed", summary["failed_tests"])
        with col4:
            pass_rate = (summary["passed_tests"] / summary["total_tests"] * 100) if summary["total_tests"] > 0 else 0
            st.metric("Pass Rate", f"{pass_rate:.1f}%")

def serialize_chat(history):
    serialized = []

    for msg in history:
        new_msg = {
            "role": msg.get("role"),
            "content": str(msg.get("content"))
        }

        if msg.get("sources"):
            safe_sources = []
            for doc in msg["sources"]:
                try:
                    safe_sources.append({
                        "content": str(doc.page_content)[:300],
                        "metadata": dict(doc.metadata) if hasattr(doc, "metadata") else {}
                    })
                except Exception:
                    continue

            new_msg["sources"] = safe_sources

        serialized.append(new_msg)

    return serialized

def display_evaluation_results():
    st.subheader("Evaluation Overview")

    if not st.session_state.evaluation_results:
        st.info("Run batch evaluation to see results")
        return

    summary = st.session_state.evaluator.get_results_summary(
        st.session_state.evaluation_results
    )

    # ==============================
    # 1. EXECUTIVE SUMMARY
    # ==============================

    col1, col2, col3 = st.columns(3)

    pass_rate = (
        summary["passed_tests"] / summary["total_tests"] * 100
        if summary["total_tests"] > 0 else 0
    )

    with col1:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")

    with col2:
        st.metric("Total Tests", summary["total_tests"])

    with col3:
        status = "Good" if pass_rate > 75 else "Needs Improvement"
        st.metric("System Health", status)

    st.divider()

    # ==============================
    # 2. METRIC PERFORMANCE (CLEAN)
    # ==============================

    st.markdown("### Key Metrics")

    for metric_name, metric_stats in summary.get("metrics", {}).items():
        score = metric_stats["avg_score"]

        # reuse your clean progress UI
        t = st.session_state.eval_profile

        name = metric_name.lower()

        if "hallucination" in name:
            threshold = t["hallucination"]
            passed = score <= threshold
        elif "faithfulness" in name:
            threshold = t["faithfulness"]
            passed = score >= threshold
        elif "relevancy" in name:
            threshold = t["relevancy"]
            passed = score >= threshold
        elif "recall" in name:
            threshold = t["recall"]
            passed = score >= threshold
        else:
            threshold = 0.5
            passed = score >= 0.5

        display_metric_card(metric_name, score, threshold, passed)

    st.divider()

    # ==============================
    # 3. DETAILS (COLLAPSIBLE)
    # ==============================

    with st.expander("View Detailed Results"):

        # Category breakdown
        if summary.get("by_category"):
            df_category = pd.DataFrame([
                {
                    "Category": cat.title(),
                    "Total": stats["count"],
                    "Passed": stats["passed"],
                    "Pass Rate (%)": round(
                        stats["passed"] / stats["count"] * 100, 1
                    )
                }
                for cat, stats in summary["by_category"].items()
            ])

            st.dataframe(df_category, width='stretch')

        st.write("---")

        # Test table
        results_data = []

        for result in st.session_state.evaluation_results:

            if result.get("error"):
                status = "ERROR"
            elif result.get("overall_passed"):
                status = "PASS"
            else:
                status = "FAIL"

            results_data.append({
                "ID": result.get("test_id"),
                "Question": (
                    result.get("question", "N/A")[:80] + "..."
                    if result.get("question") and len(result.get("question")) > 80
                    else result.get("question", "N/A")
                ),
                "Category": result.get("category", "N/A"),
                "Status": status
            })

        if results_data:
            st.dataframe(pd.DataFrame(results_data), width='stretch')
        
        for result in st.session_state.evaluation_results:
            if result.get("error"):
                st.error(f"Test {result.get('test_id')} failed: {result.get('error')}")

def display_clean_dashboard():
    st.subheader("Evaluation Dashboard")

    if not st.session_state.evaluation_results:
        st.info("Run batch evaluation to see results")
        return

    summary = st.session_state.evaluator.get_results_summary(
        st.session_state.evaluation_results
    )

    # ======================
    # 1. EXECUTIVE KPI ROW
    # ======================
    col1, col2, col3, col4 = st.columns(4)

    pass_rate = (
        summary["passed_tests"] / summary["total_tests"] * 100
        if summary["total_tests"] > 0 else 0
    )

    with col1:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")

    with col2:
        st.metric("Tests", summary["total_tests"])

    with col3:
        status = "Healthy" if pass_rate > 75 else "At Risk"
        st.metric("System", status)

    with col4:
        risk = "High" if pass_rate < 60 else "Moderate" if pass_rate < 80 else "Low"
        st.metric("Risk Level", risk)

    st.divider()

    # ======================
    # 2. MAIN CONTENT
    # ======================
    col_left, col_right = st.columns([2, 1])

    # LEFT → METRICS
    with col_left:
        st.markdown("### Key Metrics")

        for metric_name, metric_stats in summary["metrics"].items():

            score = metric_stats["avg_score"]

            # FIX: correct pass logic
            t = st.session_state.eval_profile
            name = metric_name.lower()

            if "hallucination" in name:
                threshold = t["hallucination"]
                passed = score <= threshold
            elif "faithfulness" in name:
                threshold = t["faithfulness"]
                passed = score >= threshold
            elif "relevancy" in name:
                threshold = t["relevancy"]
                passed = score >= threshold
            elif "recall" in name:
                threshold = t["recall"]
                passed = score >= threshold
            else:
                threshold = 0.5
                passed = score >= 0.5

            display_metric_card(metric_name, score, threshold, passed)

    # RIGHT → VISUAL
    with col_right:
        st.markdown("### Score Overview")

        metric_names = list(summary["metrics"].keys())
        scores = [summary["metrics"][m]["avg_score"] for m in metric_names]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metric_names,
            y=scores,
            marker=dict(
                color=scores,
                colorscale="RdYlGn"
            )
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=20),
            yaxis=dict(range=[0,1])
        )

        st.plotly_chart(fig, width='stretch')

    st.divider()

    # ======================
    # 3. DRILLDOWN
    # ======================
    with st.expander("Detailed Results"):

        # Table
        results_data = []
        for result in st.session_state.evaluation_results:

            if result.get("error"):
                status = "ERROR"
            elif result.get("overall_passed"):
                status = "PASS"
            else:
                status = "FAIL"

            results_data.append({
                "ID": result.get("test_id"),
                "Question": (
                    result.get("question", "N/A")[:80] + "..."
                    if result.get("question") and len(result.get("question")) > 80
                    else result.get("question", "N/A")
                ),
                "Category": result.get("category", "N/A"),
                "Status": status
            })

        st.dataframe(pd.DataFrame(results_data), width='stretch')

        for result in st.session_state.evaluation_results:
            if result.get("error"):
                st.error(f"Test {result.get('test_id')} failed: {result.get('error')}")

def display_metrics_dashboard():
    """Display comprehensive metrics dashboard"""
    st.subheader("Metrics Dashboard")
    
    if not st.session_state.evaluation_results:
        st.info("ℹ Run batch evaluation to see dashboard")
        return
    
    if st.session_state.evaluator is None:
        st.session_state.evaluator = UIEvaluator(st.session_state.bot.qa_chain)
    
    summary = st.session_state.evaluator.get_results_summary(st.session_state.evaluation_results)
    
    # Create metric score visualization
    if summary.get("metrics"):
        metric_names = list(summary["metrics"].keys())
        avg_scores = [summary["metrics"][m]["avg_score"] for m in metric_names]
        
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(
            x=metric_names,
            y=avg_scores,
            marker=dict(
                color=avg_scores,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Score")
            )
        ))
        fig_metrics.update_layout(
            title="Average Metric Scores",
            xaxis_title="Metrics",
            yaxis_title="Average Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_metrics, width='stretch')

    col_left, col_right = st.columns([2, 1])

    with col_left:

        st.subheader("Evaluation Overview")

        # KPIs
        col1, col2, col3 = st.columns(3)

        pass_rate = (summary["passed_tests"] / summary["total_tests"] * 100)

        with col1:
            st.metric("Pass Rate", f"{pass_rate:.1f}%")

        with col2:
            st.metric("Tests", summary["total_tests"])

        with col3:
            status = "Healthy" if pass_rate > 75 else "At Risk"
            st.metric("System", status)

        st.divider()

        st.markdown("### Key Metrics")

        for metric_name, metric_stats in summary["metrics"].items():

            score = metric_stats["avg_score"]

            # FIX: correct pass logic
            t = st.session_state.eval_profile
            name = metric_name.lower()

            if "hallucination" in name:
                threshold = t["hallucination"]
                passed = score <= threshold
            elif "faithfulness" in name:
                threshold = t["faithfulness"]
                passed = score >= threshold
            elif "relevancy" in name:
                threshold = t["relevancy"]
                passed = score >= threshold
            elif "recall" in name:
                threshold = t["recall"]
                passed = score >= threshold
            else:
                threshold = 0.5
                passed = score >= 0.5

            display_metric_card(metric_name, score, threshold, passed)
        
    with col_right:

        st.markdown("### Score Distribution")

        metric_names = list(summary["metrics"].keys())
        scores = [summary["metrics"][m]["avg_score"] for m in metric_names]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=scores,
            y=metric_names,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale="RdYlGn"
            )
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis_title="Score",
            yaxis_title=""
        )

        st.plotly_chart(fig, width='stretch')

# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    apply_professional_styling()

    initialize_session()

    display_header()

    # Create tabs for Chat and Evaluation
    tab_chat, tab_analytics, tab_evaluation = st.tabs(["Chat", "Analytics", "Evaluation"])

    with tab_chat:

        col_main, col_side = st.columns([4, 1])

        # -----------------------------------------------------

        with col_side:

            st.title("Control Center")

            st.write("---")

            if st.session_state.bot_initialized:

                st.success("System Online")

            else:

                st.warning("System Offline")

                if st.button("Initialize AI"):

                    if init_bot():
                        st.rerun()

            st.write("---")

            st.subheader("Session Tools")

            if st.button("Clear Conversation"):
                st.session_state.conversation_history = []
                st.rerun()

            st.download_button(
                label="Export Chat",
                data=json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "messages": serialize_chat(st.session_state.conversation_history),
                    "stats": st.session_state.stats
                }, indent=2),
                file_name=f"rag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

            if st.button("Reset Vector Database"):

                db_path = Path("src/chroma_db")

                try:
                    # =========================
                    # 1. DESTROY BOT REFERENCE
                    # =========================
                    if "bot" in st.session_state:
                        try:
                            del st.session_state.bot
                        except:
                            pass

                    st.session_state.bot_initialized = False

                    # =========================
                    # 2. FORCE MEMORY CLEANUP
                    # =========================
                    gc.collect()
                    time.sleep(2)

                    # =========================
                    # 3. WINDOWS-SAFE DELETE
                    # =========================
                    import os

                    if db_path.exists():
                        temp_path = db_path.parent / f"_delete_{int(time.time())}"

                        try:
                            # rename first (bypasses lock)
                            os.rename(db_path, temp_path)

                            # then delete with retry
                            for i in range(6):
                                try:
                                    shutil.rmtree(temp_path)
                                    break
                                except Exception:
                                    time.sleep(1)

                        except Exception as e:
                            st.warning(f"Rename fallback failed: {e}")

                    # =========================
                    # 4. RE-INIT SESSION SAFELY
                    # =========================
                    initialize_session()

                    st.success("✓ Vector database reset successfully")
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to reset database: {str(e)}")

            st.write("---")

            st.subheader("System Info")

            st.write("Model: Groq LLM")
            st.write("Vector DB: Chroma")
            sidebar_settings()
        # -----------------------------------------------------

        with col_main:

            st.subheader("Ask Questions")

            display_chat()

            question = st.text_input(
                "Ask something about your documents",
                placeholder="Example: What is Python used for?"
            )

            ask_clicked = st.button("Ask")

            if ask_clicked:

                if not st.session_state.bot_initialized:
                    st.warning("Initialize the AI system first")
                    return

                if not question:
                    st.warning("Please enter a question")
                    return

                valid, error = InputValidator.validate_question(question)

                if not valid:
                    st.error(error)
                    st.session_state.stats["errors"] += 1
                    return

                #  Add user message
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": question
                })

                start = time.time()

                with st.spinner("Generating answer..."):
                    response = st.session_state.bot.process_question(question)

                elapsed = time.time() - start

                if response and response.get("result"):

                    answer = response.get("result", "")
                    sources = response.get("source_documents", [])

                    is_valid, _ = OutputValidator.validate_answer(answer)
                    if not is_valid:
                        st.warning("Generated answer may be low quality")

                    #  Add assistant message
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                    st.session_state.stats["total_questions"] += 1
                    st.session_state.stats["total_time"] += elapsed

                else:
                    st.error("Failed to generate answer")
                    st.session_state.stats["errors"] += 1

                st.rerun()

    with tab_analytics:

        st.subheader("System Analytics")

        if st.session_state.stats["total_questions"] == 0:
            st.info("No data yet")
        else:
            display_kpis()

            success_rate = (
                (st.session_state.stats["total_questions"] - st.session_state.stats["errors"])
                / st.session_state.stats["total_questions"]
            ) * 100

            st.metric("Success Rate", f"{success_rate:.1f}%")

    # ========================================================
    # EVALUATION TAB
    # ========================================================

    with tab_evaluation:
        st.markdown("### Evaluation Mode")

        selected_mode = st.selectbox(
            "Select Evaluation Standard",
            ["poc", "strong", "production"]
        )

        st.session_state.eval_profile = st.session_state.eval_profiles[selected_mode]

        if not st.session_state.bot_initialized:
            st.warning("⚠ Initialize RAG bot in the Chat tab first!")
        else:

            # Evaluation sub-tabs
            eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs(
                ["Test Cases", "Single Test", "Batch Run", "Results & Dashboard"]
            )

            with eval_tab1:
                display_evaluation_test_cases()

            with eval_tab2:
                display_single_test_evaluation()

            with eval_tab3:
                display_batch_evaluation()

            with eval_tab4:
                display_clean_dashboard()

# =========================================================

if __name__ == "__main__":
    main()
