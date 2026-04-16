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
from pathlib import Path
import json

LATEST_FILE = Path("evaluation_results_latest.json")
ARCHIVE_FILE = Path(f"runs/eval_{int(time.time())}.json")

def save_results(results):
    LATEST_FILE.parent.mkdir(exist_ok=True)
    ARCHIVE_FILE.parent.mkdir(exist_ok=True)

    # save latest
    with open(LATEST_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # save archive
    with open(ARCHIVE_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

def load_results():
    if LATEST_FILE.exists():
        with open(LATEST_FILE, "r") as f:
            return json.load(f)
    return []

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
        st.session_state.evaluation_results = load_results()

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
        chunk_size = st.number_input("Chunk Size", 200, 2000, config.DOC_CHUNK_SIZE)
        overlap = st.number_input("Overlap", 0, 500, config.DOC_CHUNK_OVERLAP)

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

            db_path = config.CHROMA_DB_DIR

            if db_path.exists():
                for _ in range(5):
                    try:
                        shutil.rmtree(db_path)
                        break
                    except PermissionError:
                        time.sleep(1)

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
        st.write(f"Model: {config.LLM_MODEL}")
        st.write(f"Vector DB: {config.VECTOR_STORE_TYPE}")

# =========================================================
# BOT INITIALIZATION
# =========================================================

def init_bot():

    try:
        with st.spinner("Initializing RAG System"):

            bot = RAGBotDemo()  # let config handle path

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

def display_status_badge(status: str):
    """Display a colored status badge."""
    if status == "PASS":
        color = "green"
        icon = "✅"
    elif status == "FAIL":
        color = "red"
        icon = "❌"
    elif status == "WARNING":
        color = "orange"
        icon = "⚠️"
    else:
        color = "gray"
        icon = "❓"
    
    st.markdown(f"<span style='background-color: {color}; padding: 4px 12px; border-radius: 12px; color: white; font-weight: bold;'>{icon} {status}</span>", unsafe_allow_html=True)

def display_pass_fail_decision(decision: Dict[str, Any]):
    """Display pass/fail decision details."""
    if not decision:
        return
    
    verdict = decision.get("verdict", "UNKNOWN")
    
    # Verdict badge
    col1, col2, col3 = st.columns([1, 2, 3])
    with col1:
        st.markdown("**Decision:**")
    with col2:
        display_status_badge(verdict)
    
    # Thresholds
    passed = decision.get("thresholds_passed", {})
    failed = decision.get("thresholds_failed", [])
    
    if passed or failed:
        st.markdown("**Threshold Analysis:**")
        
        for metric, passed_check in passed.items():
            status = "✅ PASS" if passed_check else "❌ FAIL"
            st.caption(f"{metric.title()}: {status}")
        
        if failed:
            st.markdown("**Violations:**")
            for violation in failed:
                st.error(f"• {violation}", icon="❌")
    
    # Reasoning
    reasoning = decision.get("reasoning", [])
    if reasoning:
        st.markdown("**Reasoning:**")
        for reason in reasoning:
            st.caption(reason)

def display_system_readiness(readiness: str):
    """Display system readiness assessment."""
    readiness_colors = {
        "PRODUCTION_READY": ("green", "✅ Production Ready"),
        "READY_WITH_MINOR_ISSUES": ("blue", "🟦 Ready with Minor Issues"),
        "NEEDS_IMPROVEMENT": ("orange", "⚠️ Needs Improvement"),
        "NOT_READY": ("red", "❌ Not Ready for Production")
    }
    
    color, label = readiness_colors.get(readiness, ("gray", "Unknown"))
    st.markdown(f"<h3 style='color: {color};'>{label}</h3>", unsafe_allow_html=True)

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


def display_failure_analysis(result: Dict[str, Any]):
    """Display failure analysis for a single test result"""
    
    failure_analysis = result.get("failure_analysis", {})
    
    if not failure_analysis:
        return
    
    failure_type = failure_analysis.get("failure_type", "unknown")
    reason = failure_analysis.get("reason", "")
    metric_alignment = failure_analysis.get("metric_alignment", "")
    notes = failure_analysis.get("notes", "")
    
    # Color code by failure type
    if failure_type == "hallucination":
        color = "🔴 HALLUCINATION"
    elif failure_type == "retrieval_miss":
        color = "🟠 RETRIEVAL_MISS"
    elif failure_type == "partial_answer":
        color = "🟡 PARTIAL_ANSWER"
    else:
        color = "🟢 CORRECT"
    
    with st.expander(f"Failure Analysis: {color}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Type:** {failure_type.title()}")
            st.write(f"**Reason:** {reason}")
        
        with col2:
            st.write(f"**Metric Alignment:** {metric_alignment.title()}")
            st.write(f"**Impact:** {notes}")


def display_insights_dashboard(summary: Dict[str, Any], results: List[Dict[str, Any]]):
    """Display insights dashboard showing failure distribution and metric comparisons"""
    
    st.subheader("Insights Dashboard")
    
    
    col_left, col_right = st.columns(2)
    
    # LEFT: Failure Distribution
    with col_left:
        st.markdown("### Distribution")
        st.info("No failure classification in pure DeepEval mode")
        # st.markdown("### Failure Distribution")
        failure_dist = summary.get("failure_distribution", {})
        if failure_dist:
            # Create pie chart
            fig_dist = go.Figure(data=[go.Pie(
                labels=list(failure_dist.keys()),
                values=list(failure_dist.values()),
                marker=dict(colors=['#ef4444', '#f97316', '#eab308', '#22c55e'])
            )])
            
            fig_dist.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=20, b=20)
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Statistics
            st.markdown("**Breakdown:**")
            for failure_type, count in sorted(failure_dist.items(), key=lambda x: x[1], reverse=True):
                pct = (count / sum(failure_dist.values()) * 100) if failure_dist else 0
                st.write(f"- **{failure_type.title()}**: {count} ({pct:.1f}%)")
        else:
            st.info("Failure classification not enabled. Showing metric-based evaluation only.")
    
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
            f"{tc['category'].upper()} • {tc['difficulty'].upper()}  |  Q{tc['id']}: {tc['question']}"
            for tc in st.session_state.evaluation_test_cases
        ]
    selected_idx = st.selectbox("Select Test Case", range(len(test_ids)), 
                                format_func=lambda i: test_ids[i])
    
    selected_test = st.session_state.evaluation_test_cases[selected_idx]
    
    # Display test case details
    st.write("")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Category:** {selected_test.get('category', 'N/A')}")
        st.write(f"**Difficulty:** {selected_test.get('difficulty', 'N/A')}")
    
    with col2:
        st.write(f"**ID:** {selected_test['id']}")
        # st.write(f"**Notes:** {selected_test.get('notes', '')}")
    
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
        st.subheader("🎯 `Evaluation Results`")

        if "error" in result and result["error"]:
            st.error(f"❌ Error: {result['error']}")
        else:
            # ✅ PRODUCTION UI: Show final score FIRST and PROMINENTLY
            status = result.get("result", "?").upper()
            final_score = result.get("final_score", 0.0)

            # Status indicator with color
            if status == "PASS":
                status_color = "🟢"
                status_bg = "green"
            elif status == "FAIL":
                status_color = "🔴"
                status_bg = "red"
            else:  # WARNING
                status_color = "🟡"
                status_bg = "orange"

            # Prominent score display (separated answer vs retrieval)
            category = selected_test.get("category", "").lower()
            
            # Extract separated scores from new architecture
            answer_score = result.get("answer_score", final_score)
            retrieval_score = result.get("retrieval_score")

            # For UNANSWERABLE: don't show retrieval metrics (not applicable)
            if category == "unanswerable":
                col_status, col_answer = st.columns(2)

                with col_status:
                    st.metric(f"{status_color} Status", status)

                with col_answer:
                    st.metric("💡 Answer Score", f"{answer_score:.3f}")
            else:
                # For ANSWERABLE/PARTIAL: show both answer and retrieval scores
                col_status, col_answer, col_retrieval = st.columns(3)

                with col_status:
                    st.metric(f"{status_color} Status", status)

                with col_answer:
                    st.metric("💡 Answer Score", f"{answer_score:.3f}")

                with col_retrieval:
                    if retrieval_score is not None:
                        st.metric("📡 Retrieval Score", f"{retrieval_score:.3f}")
                    else:
                        st.metric("📡 Retrieval Score", "N/A")

            # Show actual answer
            st.write("---")
            st.write(f"**📝 Actual Answer:** {result['actual_answer']}")

            # ✅ SHOW DIAGNOSIS (NEW: Clean problem identification)
            diagnosis = result.get("diagnosis")
            if diagnosis:
                st.info(f"🧠 **Diagnosis:** {diagnosis}")

            # ✅ PRODUCTION UI: Show interpretation reasoning
            st.write("---")
            st.subheader("📋 `Evaluation Reasoning`")

            reasoning = result.get("reasoning", [])
            for reason in reasoning:
                st.write(f"`{reason}`")

            # Show is_refusal status
            is_refusal = result.get("is_refusal", False)
            if selected_test.get("category", "").lower() == "unanswerable":
                st.write(f"🔍 Refusal Detection: {'✅ Detected (LLM-based)' if is_refusal else '❌ Not detected'}")

            # ✅ PRODUCTION UI: Show score breakdown
            st.write("---")
            st.subheader("📈 Score Breakdown (Separated Architecture)")
            
            st.info(
                "**🧠 Key:** Answer Score and Retrieval Score are **INDEPENDENT** — "
                "not multiplied together. Each measures a different system component. "
                "They are reported separately for precise diagnostics."
            )

            col_raw, col_factor = st.columns(2)

            with col_raw:
                st.write("**Answer Score (LLM Quality):**")
                st.write(f"*No retrieval penalty — measures only answer quality*")
                metrics = result.get("metrics", {})
                for metric_name, metric_data in metrics.items():
                    # Skip ContextualRecall - it's a retrieval metric, not answer quality
                    if metric_name.lower() == "contextual_recall":
                        continue
                    score = metric_data.get("score")
                    applied = metric_data.get("applied", True)
                    if score is not None:
                        if applied:
                            st.write(f"  • {metric_name}: {score:.3f}")
                        else:
                            st.write(f"  • {metric_name}: — (not applicable)")
                    else:
                        st.write(f"  • {metric_name}: ✗ (error)")

            with col_factor:
                st.write("**Retrieval Score (RAG Quality):**")
                st.write(f"*Independent of answer quality*")
                retrieval_metrics = result.get("retrieval_metrics", {})
                metrics = result.get("metrics", {})
                
                # Show consolidated retrieval metrics
                if retrieval_metrics.get("computed", False):
                    recall = retrieval_metrics.get("recall_at_k")
                    precision = retrieval_metrics.get("precision_at_k")
                    hit_rate = retrieval_metrics.get("hit_rate_at_k")
                    if recall is not None:
                        st.write(f"  • Recall: {recall:.3f}")
                    if precision is not None:
                        st.write(f"  • Precision: {precision:.3f}")
                    if hit_rate is not None:
                        st.write(f"  • Hit Rate: {hit_rate:.3f}")
                
                # Also show ContextualRecall (from DeepEval) as part of retrieval
                if metrics.get("contextual_recall", {}).get("applied"):
                    contextual_score = metrics.get("contextual_recall", {}).get("score")
                    if contextual_score is not None:
                        st.write(f"  • Contextual Recall: {contextual_score:.3f}")
                
                if not retrieval_metrics.get("computed", False) and not metrics.get("contextual_recall", {}).get("applied"):
                    st.write("*No ground truth for retrieval metrics*")

            # ✅ PRODUCTION UI: Show raw metrics as secondary details
            st.write("---")
            st.subheader("📊 Detailed Metrics")

            col1, col2 = st.columns(2)
            metric_count = 0
            for metric_name, metric_data in metrics.items():
                with col1 if metric_count % 2 == 0 else col2:
                    score = metric_data.get("score")
                    applied = metric_data.get("applied", True)

                    if score is not None and applied:
                        # Normalize score for display
                        if metric_name.lower() == "hallucination":
                            # Invert hallucination for display (higher is worse)
                            display_score = 1 - score
                            st.write(f"**{metric_name}** (inverted): {display_score:.3f}")
                        else:
                            st.write(f"**{metric_name}**: {score:.3f}")
                    elif applied:
                        st.warning(f"**{metric_name}**: Error (no score)")
                    metric_count += 1

            # Show retrieved context
            st.write("---")
            with st.expander("📄 Retrieved Context Details"):
                context = result.get("retrieval_context", [])
                if context:
                    st.write(f"Retrieved {len(context)} document(s):")
                    for i, doc in enumerate(context, 1):
                        st.write(f"\n**Document {i}:**")
                        st.text_area(f"doc_{i}", doc, height=100, disabled=True, key=f"context_{i}")
                else:
                    st.info("No context retrieved for this query")


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
        save_results(batch_results)

        if st.button("Clear Saved Results"):
            LATEST_FILE.unlink(missing_ok=True)
            st.session_state.evaluation_results = []
            st.success("Results cleared")
        
        progress_bar.progress(1.0)
        status_text.text(" Evaluation complete!")
        
        # Display summary
        st.write("---")
        st.subheader("Batch Results Summary")
        
        results = st.session_state.evaluation_results

        if not results:
            return {}

        import statistics

        metric_names = ["Hallucination", "Faithfulness", "AnswerRelevancy", "ContextualRecall"]

        metric_statistics = {}

        for metric in metric_names:
            scores = [
                r["metrics"].get(metric, {}).get("score")
                for r in results
                if r.get("metrics") and r["metrics"].get(metric, {}).get("score") is not None
            ]

            if scores:
                metric_statistics[metric] = {
                    "mean": statistics.mean(scores),
                    "min": min(scores),
                    "max": max(scores)
                }

        summary = {
            "total_tests": len(results),
            "metric_statistics": metric_statistics
        }
        
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

    results = st.session_state.evaluation_results

    if not results:
        return {}

    import statistics

    metric_names = ["Hallucination", "Faithfulness", "AnswerRelevancy", "ContextualRecall"]

    metric_statistics = {}

    for metric in metric_names:
        scores = [
            r["metrics"].get(metric, {}).get("score")
            for r in results
            if r.get("metrics") and r["metrics"].get(metric, {}).get("score") is not None
        ]

        if scores:
            metric_statistics[metric] = {
                "mean": statistics.mean(scores),
                "min": min(scores),
                "max": max(scores)
            }

    summary = {
        "total_tests": len(results),
        "metric_statistics": metric_statistics
    }

    # ==============================
    # 0. SYSTEM READINESS ASSESSMENT (NEW)
    # ==============================
    
    st.markdown("### 🎯 System Readiness & Decision Status")
    
    # Calculate pass/fail statistics
    pass_fail_pass = sum(1 for r in results if r.get("overall_status") == "PASS")
    pass_fail_fail = sum(1 for r in results if r.get("overall_status") == "FAIL")
    pass_fail_warning = sum(1 for r in results if r.get("overall_status") == "WARNING")
    total = len(results)
    
    # Determine system readiness
    pass_rate = pass_fail_pass / total if total > 0 else 0
    if pass_rate >= 0.95:
        readiness = "PRODUCTION_READY"
    elif pass_rate >= 0.85:
        readiness = "READY_WITH_MINOR_ISSUES"
    elif pass_rate >= 0.70:
        readiness = "NEEDS_IMPROVEMENT"
    else:
        readiness = "NOT_READY"
    
    # Display system readiness
    display_system_readiness(readiness)
    
    # Pass/Fail summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ PASS", pass_fail_pass)
    with col2:
        st.metric("⚠️ WARNING", pass_fail_warning)
    with col3:
        st.metric("❌ FAIL", pass_fail_fail)
    with col4:
        st.metric("Pass Rate", f"{pass_rate*100:.1f}%")
    
    st.divider()

    # ==============================
    # 1. EXECUTIVE SUMMARY / METRIC PERFORMANCE
    # ==============================

    st.markdown("### Metric Performance")

    for metric_name, metric_stats in summary.get("metric_statistics", {}).items():
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

            status = result.get("overall_status", result.get("result", "UNKNOWN"))

            results_data.append({
                "ID": result.get("test_id"),
                "Question": (
                    result.get("question", "N/A")[:60] + "..."
                    if result.get("question") and len(result.get("question")) > 60
                    else result.get("question", "N/A")
                ),
                "Category": result.get("category", "N/A").upper(),
                "Status": status,
                "Score": f"{result.get('final_score', 0):.2f}"
            })

        if results_data:
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, width='stretch', use_container_width=True)
        
        # Show detailed results with failure analysis
        st.write("---")
        st.subheader("Detailed Test Results & Pass/Fail Decisions")
        
        for result in st.session_state.evaluation_results:
            if result.get("error"):
                st.error(f"Test {result.get('test_id')} failed: {result.get('error')}")
            else:
                # Create expandable section for each test
                test_id = result.get("test_id")
                category = result.get("category", "unknown")
                question = result.get("question", "")
                status = result.get("overall_status", result.get("result", "UNKNOWN"))
                score = result.get("final_score", 0.0)
                
                with st.expander(f"Test #{test_id} ({category.upper()}) - [{status}] Score: {score:.2f}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Question:** {question}")
                        st.write(f"**Expected:** {result.get('expected_answer', 'N/A')[:200]}")
                    
                    with col2:
                        st.write(f"**Actual:** {result.get('actual_answer', 'N/A')[:200]}")
                    
                    # Pass/Fail Decision
                    st.divider()
                    pass_fail = result.get("pass_fail", {})
                    if pass_fail:
                        st.write("**Pass/Fail Decision:**")
                        display_pass_fail_decision(pass_fail)
                    
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
                    display_failure_analysis(result)

def display_clean_dashboard():
    st.subheader("Evaluation Dashboard")

    if not st.session_state.evaluation_results:
        st.info("Run batch evaluation to see results")
        return

    results = st.session_state.evaluation_results

    if not results:
        st.info("Run batch evaluation to see results")
        return

    # rebuild summary manually
    import statistics

    metric_names = ["Hallucination", "Faithfulness", "AnswerRelevancy", "ContextualRecall"]

    metric_statistics = {}

    for metric in metric_names:
        scores = [
            r["metrics"].get(metric, {}).get("score")
            for r in results
            if r.get("metrics") and r["metrics"].get(metric, {}).get("score") is not None
        ]

        if scores:
            metric_statistics[metric] = {
                "mean": statistics.mean(scores),
                "min": min(scores),
                "max": max(scores)
            }

    summary = {
        "total_tests": len(results),
        "metric_statistics": metric_statistics
    }

    # ======================
    # 1. EXECUTIVE KPI ROW
    # ======================
    # st.markdown("### Key Metrics")

    # for metric_name, metric_stats in summary["metric_statistics"].items():
    #     score = metric_stats.get("mean", 0)
    #     display_metric_card(metric_name, score)

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
                    result.get("question", "N/A")
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
    
    results = st.session_state.evaluation_results

    if not results:
        return {}

    import statistics

    metric_names = ["Hallucination", "Faithfulness", "AnswerRelevancy", "ContextualRecall"]

    metric_statistics = {}

    for metric in metric_names:
        scores = [
            r["metrics"].get(metric, {}).get("score")
            for r in results
            if r.get("metrics") and r["metrics"].get(metric, {}).get("score") is not None
        ]

        if scores:
            metric_statistics[metric] = {
                "mean": statistics.mean(scores),
                "min": min(scores),
                "max": max(scores)
            }

    summary = {
        "total_tests": len(results),
        "metric_statistics": metric_statistics
    }
    
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

        pass_rate = 0  # or remove completely

        with col1:
            st.metric("Evaluation Coverage", f"{len(st.session_state.evaluation_results)} tests")

        with col2:
            st.metric("Tests", summary["total_tests"])

        with col3:
            status = "Active"
            st.metric("System", status)

        st.divider()

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
    tab_chat, tab_analytics, tab_evaluation = st.tabs(["Chat", "Analytics", "Evaluation"])

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

        st.info("Production Evaluation Mode (DeepEval + Decision Engine)")

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
