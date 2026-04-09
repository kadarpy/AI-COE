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
from typing import Dict, List, Any

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

def cleanup_memory():
    """Clean up memory resources to prevent memory leaks"""
    try:
        # Clean up session state resources
        if "bot" in st.session_state and st.session_state.bot is not None:
            # Release evaluator if exists
            if hasattr(st.session_state.bot, 'evaluator'):
                st.session_state.bot.evaluator = None
        
        # Force garbage collection
        gc.collect()
        logger.debug("Memory cleanup completed")
    except Exception as e:
        logger.warning(f"Error during memory cleanup: {e}")


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
    # if "eval_profiles" not in st.session_state:
    #     st.session_state.eval_profiles = {
    #         "poc": {
    #             "faithfulness": 0.70,
    #             "relevancy": 0.75,
    #             "recall": 0.70,
    #             "hallucination": 0.25
    #         },
    #         "strong": {
    #             "faithfulness": 0.85,
    #             "relevancy": 0.85,
    #             "recall": 0.80,
    #             "hallucination": 0.15
    #         },
    #         "production": {
    #             "faithfulness": 0.90,
    #             "relevancy": 0.90,
    #             "recall": 0.90,
    #             "hallucination": 0.05
    #         }
    #     }

    # =========================
    # DEFAULT PROFILE
    # =========================
    # if "eval_profile" not in st.session_state:
    #     st.session_state.eval_profile = st.session_state.eval_profiles["poc"]

# =========================================================
# HEADER
# =========================================================

def display_header():

    st.markdown("""
    <div class="hero">

    <h1>RAG Evaluation Platform</h1>

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

    # =========================
    # LLM CONFIG
    # =========================
    with st.sidebar.expander("LLM Config"):
        temp = st.slider("Temperature", 0.0, 1.0, float(config.TEMPERATURE))
        k = st.slider("Retriever K", 1, 10, config.RETRIEVER_K)
        st.caption(f"Temp={temp}, K={k}")

    # =========================
    # CHUNK CONFIG
    # =========================
    with st.sidebar.expander("Chunk Config"):
        chunk_size = st.number_input("Chunk Size", 200, 2000, config.PDF_CHUNK_SIZE)
        overlap = st.number_input("Overlap", 0, 500, config.PDF_CHUNK_OVERLAP)

    # =========================
    # CONTROL CENTER (NEW)
    # =========================
    with st.sidebar.expander("Control Center", expanded=True):

        if st.session_state.bot_initialized:
            st.success("System Online")
        else:
            st.warning("System Offline")

            if st.button("Initialize AI", key="sidebar_init"):
                if init_bot():
                    st.rerun()

        st.markdown("---")

        st.subheader("Session Tools")

        if st.button("Clear Conversation", key="sidebar_clear"):
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

        if st.button("Reset Vector Database", key="sidebar_reset"):

            db_path = Path(config.CHROMA_DB_DIR)

            try:
                if "bot" in st.session_state:
                    del st.session_state.bot

                st.session_state.bot_initialized = False

                gc.collect()
                time.sleep(2)

                import os

                if db_path.exists():
                    temp_path = db_path.parent / f"_delete_{int(time.time())}"

                    os.rename(db_path, temp_path)

                    for _ in range(6):
                        try:
                            shutil.rmtree(temp_path)
                            break
                        except:
                            time.sleep(1)

                initialize_session()

                st.success("Vector DB reset successfully")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to reset database: {str(e)}")

        st.markdown("---")

        st.subheader("System Info")
        st.write("Model: Groq LLM")
        st.write(f"Vector DB: {config.VECTOR_STORE_TYPE}")

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

def display_metric_card(metric_name: str, score: float):

    col1, col2 = st.columns([3, 1])

    with col1:
        if "hallucination" in metric_name.lower():
            st.markdown("**Hallucination (Lower = Better)**")
        else:
            st.markdown(f"**{metric_name}**")

    with col2:
        st.markdown(f"**{score:.2f}**" if score is not None else "**N/A**")

    safe_score = min(max(score if score is not None else 0, 0), 1)

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
                width:{int(safe_score*100)}%;
                height:100%;
                background:#3b82f6;
                border-radius:999px;
                transition:0.4s;
            "></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# def display_failure_analysis(result: Dict[str, Any]):
#     """Display failure analysis for a single test result"""
    
#     failure_analysis = result.get("failure_analysis", {})
    
#     if not failure_analysis:
#         return
    
#     failure_type = failure_analysis.get("failure_type", "unknown")
#     reason = failure_analysis.get("reason", "")
#     metric_alignment = failure_analysis.get("metric_alignment", "")
#     notes = failure_analysis.get("notes", "")
    
#     # Color code by failure type
#     if failure_type == "hallucination":
#         color = "🔴 HALLUCINATION"
#     elif failure_type == "retrieval_miss":
#         color = "🟠 RETRIEVAL_MISS"
#     elif failure_type == "partial_answer":
#         color = "🟡 PARTIAL_ANSWER"
#     else:
#         color = "🟢 CORRECT"
    
#     with st.expander(f"Failure Analysis: {color}"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.write(f"**Type:** {failure_type.title()}")
#             st.write(f"**Reason:** {reason}")
        
#         with col2:
#             st.write(f"**Metric Alignment:** {metric_alignment.title()}")
#             st.write(f"**Impact:** {notes}")


def display_insights_dashboard(summary: Dict[str, Any], results: List[Dict[str, Any]]):
    """Display insights dashboard showing failure distribution and metric comparisons"""
    
    st.subheader("Insights Dashboard")
    
    
    col_left, col_right = st.columns(2)
    
    # LEFT: Failure Distribution
    with col_left:
        st.markdown("### Distribution")
        st.info("No failure classification in pure DeepEval mode")
        # st.markdown("### Failure Distribution")
        
        # if failure_dist:
        #     # Create pie chart
        #     fig_dist = go.Figure(data=[go.Pie(
        #         labels=list(failure_dist.keys()),
        #         values=list(failure_dist.values()),
        #         marker=dict(colors=['#ef4444', '#f97316', '#eab308', '#22c55e'])
        #     )])
            
        #     fig_dist.update_layout(
        #         height=300,
        #         margin=dict(l=10, r=10, t=20, b=20)
        #     )
            
        #     st.plotly_chart(fig_dist, use_container_width=True)
            
        #     # Statistics
        #     st.markdown("**Breakdown:**")
        #     for failure_type, count in sorted(failure_dist.items(), key=lambda x: x[1], reverse=True):
        #         pct = (count / sum(failure_dist.values()) * 100) if failure_dist else 0
        #         st.write(f"- **{failure_type.title()}**: {count} ({pct:.1f}%)")
        # else:
        #     st.info("No failures detected - all tests passed!")
    
    # RIGHT: Metric Performance Comparison
    with col_right:
        st.markdown("### Metric Performance")
        
        metrics = summary.get("metric_statistics", {})
        
        if metrics:
            metric_names = []
            avg_scores = []
            min_scores = []
            max_scores = []
            
            for metric_name, metric_stats in metrics.items():
                metric_names.append(metric_name.replace("Metric", ""))
                avg_scores.append(metric_stats.get("mean", 0))
                min_scores.append(metric_stats.get("min", 0))
                max_scores.append(metric_stats.get("max", 0))
            
            # Create comparison bar chart
            fig_metrics = go.Figure(data=[
                go.Bar(x=metric_names, y=avg_scores, name='Average', marker_color='#3b82f6'),
                go.Bar(x=metric_names, y=min_scores, name='Min', marker_color='#ef4444', opacity=0.5),
                go.Bar(x=metric_names, y=max_scores, name='Max', marker_color='#10b981', opacity=0.5)
            ])
            
            fig_metrics.update_layout(
                height=300,
                barmode='group',
                margin=dict(l=10, r=10, t=20, b=20),
                yaxis=dict(range=[0, 1]),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Bottom: Key Findings
    st.markdown("---")



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
                st.success(f" Loaded {len(test_manager.test_cases)} test cases")
            except Exception as e:
                st.error(f" Error loading test cases: {str(e)}")
    
    # Show test case breakdown
    if st.session_state.evaluation_test_cases:
        st.write("---")
        col1, col2, col3 = st.columns(3)
        
        categories = {}
        for tc in st.session_state.evaluation_test_cases:
            cat = tc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        # Sort for consistency
        sorted_categories = dict(sorted(categories.items()))

        cols = st.columns(len(sorted_categories))

        for i, (cat, count) in enumerate(sorted_categories.items()):
            with cols[i]:
                st.metric(cat.replace("_", " ").title(), count)


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

    # LLM validation handled internally by DeepEval pipeline
    
    # Select test case
    test_ids = [
                f"[{tc['category'].upper()} | {tc['difficulty'].upper()}] "
                f"Q#{tc['id']}: {tc['question'][:50]}..."
                for tc in st.session_state.evaluation_test_cases
                ]
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
            st.error(f" Error: {result['error']}")
        else:
            # Show actual answer
            st.write(f"**Actual Answer:** {result['actual_answer']}")
            
            # Show metrics
            st.write("---")
            st.write("**DeepEval Metric Scores:**")
            
            cols = st.columns(2)
            metrics = result.get("metrics", {})
            
            for idx, (metric_name, metric_data) in enumerate(metrics.items()):
                with cols[idx % 2]:

                    score = metric_data.get("score")

                    if score is not None:
                        display_metric_card(
                            metric_name,
                            score
                        )
                    else:
                        st.warning(f"{metric_name}: No score")
            
            # =========================================
            # NEW: Display ML Evaluator Metrics
            # =========================================
            ml_metrics = result.get("ml_metrics", {})
            if ml_metrics and "error" not in ml_metrics:
                st.write("---")
                st.write("**ML Evaluator Metrics (CrossEncoder):**")
                st.info("Semantic relevance and context overlap scores computed with sentence-transformers", icon="🤖")
                
                ml_cols = st.columns(3)
                
                # Semantic Relevance
                with ml_cols[0]:
                    semantic_rel = ml_metrics.get("semantic_relevance", 0)
                    display_metric_card(
                        "Semantic Relevance",
                        semantic_rel
                    )
                
                # Context Overlap
                with ml_cols[1]:
                    context_ovlp = ml_metrics.get("context_overlap", 0)
                    display_metric_card(
                        "Context Overlap",
                        context_ovlp
                    )
                
                # Confidence Score
                with ml_cols[2]:
                    confidence = ml_metrics.get("confidence_score", 0)
                    display_metric_card(
                        "Confidence Score",
                        confidence
                    )
            elif ml_metrics and "error" in ml_metrics:
                st.warning(f"ML Metrics unavailable: {ml_metrics.get('error', 'Unknown error')}")
            
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
        status_text.text(" Evaluation complete!")
        
        # Display summary
        st.write("---")
        st.subheader("Batch Results Summary")
        
        summary = st.session_state.evaluator.get_results_summary(batch_results)
        
        col1 = st.columns(1)[0]
        with col1:
            st.metric("Total Tests", summary.get("total_tests", 0))

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

    st.markdown("### Metric Performance")

    for metric_name, metric_stats in summary.get("metric_statistics", {}).items():
        score = metric_stats.get("mean", 0)

        display_metric_card(
            metric_name,
            score
        )

    # =========================================
    # NEW: Display ML Metrics Summary
    # =========================================
    ml_metrics_summary = summary.get("ml_metrics_statistics", {})
    if ml_metrics_summary:
        st.divider()
        st.markdown("### ML Evaluator Metrics (CrossEncoder)")
        st.info("Hybrid evaluation combining LLM judge (DeepEval) with deterministic ML scoring", icon="🤖")
        
        for metric_name, metric_stats in ml_metrics_summary.items():
            score = metric_stats.get("mean", 0)
            display_metric_card(
                metric_name,
                score
            )

    st.divider()

    # ==============================
    # 3. INSIGHTS DASHBOARD (NEW)
    # ==============================
    
    display_insights_dashboard(summary, st.session_state.evaluation_results)
    
    st.divider()

    # ==============================
    # 4. DETAILS (COLLAPSIBLE)
    # ==============================

    with st.expander("View Detailed Results"):

        # Category breakdown
        # if summary.get("by_category"):
        #     df_category = pd.DataFrame([
        #         {
        #             "Category": cat.title(),
        #             "Total": stats["count"],
        #             "Passed": stats["passed"],
        #             "Pass Rate (%)": round(
        #                 stats["passed"] / stats["count"] * 100, 1
        #             )
        #         }
        #         for cat, stats in summary["by_category"].items()
        #     ])

        #     st.dataframe(df_category, width='stretch')

        # st.write("---")

        # Test table
        results_data = []

        for result in st.session_state.evaluation_results:

            status = "OK" if not result.get("error") else "ERROR"

            results_data.append({
                "ID": result.get("test_id"),
                "Question": (
                    result.get("question", "N/A")[:80] + "..."
                    if result.get("question") and len(result.get("question")) > 80
                    else result.get("question", "N/A")
                ),
                "Category": result.get("category", "N/A"),
            })

        if results_data:
            st.dataframe(pd.DataFrame(results_data), width='stretch')
        
        # Show detailed results with failure analysis
        st.write("---")
        st.subheader("Detailed Test Results")
        
        for result in st.session_state.evaluation_results:
            if result.get("error"):
                st.error(f"Test {result.get('test_id')} failed: {result.get('error')}")
            else:
                # Create expandable section for each test
                test_id = result.get("test_id")
                category = result.get("category", "unknown")
                question = result.get("question", "")
                status = "Result"
                
                with st.expander(f"Test #{test_id} ({category}) - {status}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Question:** {question}")
                        st.write(f"**Expected:** {result.get('expected_answer', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Actual:** {result.get('actual_answer', 'N/A')}")
                    
                    st.write("---")
                    
                    # Show metrics
                    st.write("**Metrics:**")
                    metric_cols = st.columns(2)
                    for idx, (metric_name, metric_data) in enumerate(result.get("metrics", {}).items()):
                        with metric_cols[idx % 2]:
                            score = metric_data.get("score")
                            if score is not None:
                                display_metric_card(
                                    metric_name,
                                    score
                                )
                    
                    st.write("---")
                    
                    # Show failure analysis

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
    st.markdown("### Key Metrics")

    for metric_name, metric_stats in summary["metric_statistics"].items():
        score = metric_stats.get("mean", 0)
        display_metric_card(metric_name, score)

    st.divider()

    # ======================
    # 2. MAIN CONTENT
    # ======================
    col_left, col_right = st.columns([2, 1])

    # LEFT → METRICS
    with col_left:
        st.markdown("### Key Metrics")

        for metric_name, metric_stats in summary["metric_statistics"].items():
            score = metric_stats.get("mean", 0)

            display_metric_card(metric_name, score)

    # RIGHT → VISUAL
    with col_right:
        st.markdown("### Score Overview")

        metric_names = list(summary["metric_statistics"].keys())
        scores = [summary["metric_statistics"][m]["mean"] for m in metric_names]

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

            status = "OK" if not result.get("error") else "ERROR"

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
    if summary.get("metric_statistics"):
        metric_names = list(summary["metric_statistics"].keys())
        avg_scores = [summary["metric_statistics"][m]["mean"] for m in metric_names]
        
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

        # pass_rate = (summary["passed_tests"] / summary["total_tests"] * 100)

        # with col1:
        #     st.metric("Pass Rate", f"{pass_rate:.1f}%")

        # with col2:
        #     st.metric("Tests", summary["total_tests"])

        # with col3:
        #     status = "Healthy" if pass_rate > 75 else "At Risk"
        #     st.metric("System", status)

        # st.divider()

        st.markdown("### Key Metrics")

        for metric_name, metric_stats in summary["metric_statistics"].items():

            score = metric_stats.get("mean", 0)


            display_metric_card(
                metric_name,
                score
            )
        
    with col_right:

        st.markdown("### Score Distribution")

        metric_names = list(summary["metric_statistics"].keys())
        scores = [summary["metric_statistics"][m]["mean"] for m in metric_names]

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
    tab_chat, tab_analytics, tab_evaluation, tab_advanced = st.tabs(["Chat", "Analytics", "Evaluation", "Advanced Evaluation"])

    with tab_chat:

        col_main, col_side = st.columns([4, 1])

        # -----------------------------------------------------

        with col_side:

            sidebar_settings()
        # -----------------------------------------------------

        with col_main:

            st.subheader("Ask Questions")

            display_chat()

            question = st.text_input(
                "Ask something about your documents",
                placeholder="Example: What is machine learning used for?"
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

        st.info("Using pure DeepEval (no thresholds, no pass/fail)")

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

    # ========================================================
    # ADVANCED EVALUATION TAB (Phase 2)
    # ========================================================

    with tab_advanced:
        st.markdown("### Advanced Evaluation (Phase 2)")

        if not st.session_state.evaluation_results:
            st.info("ℹ Run batch evaluation to see advanced features")
            return

        if st.session_state.evaluator is None:
            st.session_state.evaluator = UIEvaluator(st.session_state.bot.qa_chain)

        # Create sub-tabs for advanced features
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs(
            ["Trained Models", "Pass/Fail Analysis", "Retrieval Metrics", "MLflow Tracking"]
        )

        with adv_tab1:
            st.subheader("Trained Model Predictions")

            if not hasattr(st.session_state.evaluator, 'trained_evaluator') or not st.session_state.evaluator.trained_evaluator:
                st.warning("⚠ Trained evaluator not initialized. Train models first.")
                st.info("Use `python -m src.mlops.train_evaluator` to train models.")
            elif not st.session_state.evaluator.trained_evaluator.is_available():
                st.warning("⚠ No trained models available. Train models first.")
                st.info("Use `python -m src.mlops.train_evaluator` to train models.")
            else:
                st.success("✓ Trained models available")

                # Display predictions from evaluation results
                trained_predictions = {
                    "relevance": [],
                    "hallucination": [],
                    "faithfulness": []
                }

                for result in st.session_state.evaluation_results:
                    predictions = result.get("trained_model_metrics", {})
                    if predictions and predictions.get("status") != "error":
                        for key in trained_predictions:
                            if key in predictions and predictions[key] is not None:
                                trained_predictions[key].append(predictions[key])

                # Show statistics
                for model_name, scores in trained_predictions.items():
                    if scores:
                        import statistics
                        avg = statistics.mean(scores)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{model_name.title()} (Avg)", f"{avg:.3f}")
                        with col2:
                            st.metric(f"{model_name.title()} (Min)", f"{min(scores):.3f}")
                        with col3:
                            st.metric(f"{model_name.title()} (Max)", f"{max(scores):.3f}")

                        # Histogram
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=scores, nbinsx=10))
                        fig.update_layout(
                            title=f"{model_name.title()} Distribution",
                            xaxis_title="Score",
                            yaxis_title="Frequency",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)

        with adv_tab2:
            st.subheader("Pass/Fail Analysis")

            if not hasattr(st.session_state.evaluator, 'threshold_engine') or not st.session_state.evaluator.threshold_engine:
                st.warning("⚠ Threshold engine not initialized")
            else:
                st.success("✓ Threshold engine active")

                # Show threshold configuration
                with st.expander("Threshold Configuration"):
                    config_col1, config_col2, config_col3 = st.columns(3)
                    with config_col1:
                        st.metric("Relevance Min", config.THRESHOLD_RELEVANCE)
                    with config_col2:
                        st.metric("Faithfulness Min", config.THRESHOLD_FAITHFULNESS)
                    with config_col3:
                        st.metric("Hallucination Max", config.THRESHOLD_HALLUCINATION)

                # Collect pass/fail results
                pass_count = 0
                fail_count = 0
                pass_fail_details = []

                for result in st.session_state.evaluation_results:
                    pf = result.get("pass_fail")
                    if pf:
                        if pf.get("pass"):
                            pass_count += 1
                        else:
                            fail_count += 1
                        pass_fail_details.append({
                            "test_id": result.get("test_id"),
                            "pass": pf.get("pass"),
                            "reason": pf.get("reason")
                        })

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Passed", pass_count)
                with col2:
                    st.metric("Failed", fail_count)
                with col3:
                    pass_rate = (pass_count / (pass_count + fail_count) * 100) if (pass_count + fail_count) > 0 else 0
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")

                # Results table
                if pass_fail_details:
                    st.write("---")
                    st.write("**Detailed Results:**")
                    st_col1, st_col2, st_col3 = st.columns([1, 2, 3])
                    with st_col1:
                        st.write("**ID**")
                    with st_col2:
                        st.write("**Status**")
                    with st_col3:
                        st.write("**Reason**")

                    for detail in pass_fail_details:
                        st_col1, st_col2, st_col3 = st.columns([1, 2, 3])
                        with st_col1:
                            st.write(detail["test_id"])
                        with st_col2:
                            status = "✓ PASS" if detail["pass"] else "✗ FAIL"
                            st.write(status)
                        with st_col3:
                            st.write(detail["reason"])

        with adv_tab3:
            st.subheader("Retrieval Metrics")

            if not hasattr(st.session_state.evaluator, 'retrieval_metrics') or not st.session_state.evaluator.retrieval_metrics:
                st.warning("⚠ Retrieval metrics not initialized")
            else:
                st.success("✓ Retrieval metrics available")

                # Collect metrics
                recall_scores = []
                precision_scores = []
                coverage_scores = []

                for result in st.session_state.evaluation_results:
                    ret_metrics = result.get("retrieval_metrics")
                    if ret_metrics and "error" not in ret_metrics:
                        if "recall_at_k" in ret_metrics and ret_metrics["recall_at_k"] is not None:
                            recall_scores.append(ret_metrics["recall_at_k"])
                        if "precision_at_k" in ret_metrics and ret_metrics["precision_at_k"] is not None:
                            precision_scores.append(ret_metrics["precision_at_k"])
                        if "coverage" in ret_metrics and ret_metrics["coverage"] is not None:
                            coverage_scores.append(ret_metrics["coverage"])

                # Display statistics
                if recall_scores:
                    import statistics

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Recall@K (Avg)", f"{statistics.mean(recall_scores):.3f}")
                        fig_recall = go.Figure()
                        fig_recall.add_trace(go.Histogram(x=recall_scores, nbinsx=10, name="Recall@K"))
                        fig_recall.update_layout(height=250, title="Recall@K Distribution")
                        st.plotly_chart(fig_recall, use_container_width=True)

                    with col2:
                        st.metric("Precision@K (Avg)", f"{statistics.mean(precision_scores):.3f}")
                        fig_prec = go.Figure()
                        fig_prec.add_trace(go.Histogram(x=precision_scores, nbinsx=10, name="Precision@K", marker_color="orange"))
                        fig_prec.update_layout(height=250, title="Precision@K Distribution")
                        st.plotly_chart(fig_prec, use_container_width=True)

                    with col3:
                        st.metric("Coverage (Avg)", f"{statistics.mean(coverage_scores):.3f}")
                        fig_cov = go.Figure()
                        fig_cov.add_trace(go.Histogram(x=coverage_scores, nbinsx=10, name="Coverage", marker_color="green"))
                        fig_cov.update_layout(height=250, title="Coverage Distribution")
                        st.plotly_chart(fig_cov, use_container_width=True)
                else:
                    st.info("No retrieval metrics available")

        with adv_tab4:
            st.subheader("MLflow Tracking")

            if not hasattr(st.session_state.evaluator, 'mlflow_tracker') or not st.session_state.evaluator.mlflow_tracker:
                st.warning("⚠ MLflow tracking not enabled")
                st.info("Enable MLflow in config: ENABLE_MLFLOW=true")
            else:
                st.success("✓ MLflow tracking enabled")

                # Show MLflow info
                mlflow_info = st.session_state.evaluator.mlflow_tracker.get_experiment_info()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Experiment", mlflow_info.get("experiment_name", "N/A"))
                with col2:
                    st.metric("Experiment ID", mlflow_info.get("experiment_id", "N/A"))

                st.write(f"**Tracking URI:** `{mlflow_info.get('tracking_uri', 'N/A')}`")

                st.info("View MLflow UI:\n```\nmlflow ui --backend-store-uri file:mlruns\n```")

# =========================================================

if __name__ == "__main__":
    main()
