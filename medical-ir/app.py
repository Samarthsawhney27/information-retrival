"""
app.py
------
MedIR — Biomedical Information Retrieval System
Streamlit application

Run with:
    streamlit run app.py
"""

import sys
import os
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from data_loader import load_dataset, compute_dataset_stats
from preprocessing import preprocess_documents, analyse_query
from ir_engine import MedicalSearchEngine
from evaluation import evaluate_system, compare_systems
from visualization import (
    plot_model_comparison,
    plot_precision_at_k,
    plot_recall_at_k,
    plot_precision_recall_curve,
    plot_latency,
    plot_overlap_heatmap,
    plot_grade_distribution,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedIR — Biomedical Information Retrieval",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header gradient */
.hero-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #3b82f6 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.25);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.8);
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.75rem;
    backdrop-filter: blur(4px);
}

/* Result cards */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.875rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s ease, transform 0.15s ease;
    position: relative;
}
.result-card:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.12);
    transform: translateY(-1px);
}
.result-rank {
    position: absolute;
    top: 1rem;
    right: 1.25rem;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
}
.result-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.25rem;
    padding-right: 2.5rem;
}
.result-docid {
    font-size: 0.78rem;
    color: #64748b;
    font-family: monospace;
    background: #f1f5f9;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 0.6rem;
}
.result-score {
    font-size: 0.85rem;
    color: #2563eb;
    font-weight: 500;
    margin-bottom: 0.6rem;
}
.result-snippet {
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.6;
}
.rrf-ranks {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}
.rrf-badge {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    color: #475569;
    font-weight: 500;
}

/* Query analysis box */
.query-analysis {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}
.qa-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #2563eb;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.25rem;
}
.qa-value {
    font-size: 0.9rem;
    color: #1e293b;
    font-weight: 400;
}
.term-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.15rem;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2563eb;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #2563eb22, transparent);
    border-radius: 2px;
    margin-left: 0.5rem;
}

/* Sidebar */
.sidebar-section {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* About page */
.about-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.about-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #2563eb;
    margin-bottom: 0.75rem;
    border-bottom: 2px solid #eff6ff;
    padding-bottom: 0.5rem;
}
.pipeline-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.8;
    overflow-x: auto;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f8fafc;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    font-weight: 500;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: #2563eb;
    color: white;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    border-radius: 4px;
}

/* Alert / info boxes */
.info-box {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 0.875rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.88rem;
    color: #1e40af;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_raw_dataset():
    """Load and preprocess the NFCorpus dataset (cached)."""
    docs, queries, qrels = load_dataset(
        doc_splits=("train", "dev", "test"),
        eval_split="test",
        query_type="titles",
        qrel_type="3-2-1",
    )
    processed_docs = preprocess_documents(docs, stem=True)
    stats = compute_dataset_stats(docs, queries, qrels)
    return processed_docs, queries, qrels, stats


@st.cache_resource(show_spinner=False)
def _build_search_engine(bm25_k1: float = 1.5, bm25_b: float = 0.75):
    """Build and cache all search indices."""
    processed_docs, queries, qrels, stats = _load_raw_dataset()
    engine = MedicalSearchEngine(
        processed_docs, bm25_k1=bm25_k1, bm25_b=bm25_b
    )
    engine.build_tfidf_index()
    engine.build_bm25_index()
    engine.build_dense_index()
    return engine


@st.cache_data(show_spinner=False)
def _run_evaluation(_engine_id: str):
    """Run system evaluation over test queries (cached by engine identity)."""
    processed_docs, queries, qrels, stats = _load_raw_dataset()
    engine = _build_search_engine()

    systems = {
        "TF-IDF":     engine.search_tfidf,
        "BM25":       engine.search_bm25,
        "Semantic":   engine.search_dense,
        "Hybrid RRF": engine.search_rrf,
    }

    with st.spinner("🔬 Running evaluation over NFCorpus test queries…"):
        results = compare_systems(systems, queries, qrels, top_k=100)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 MedIR")
        st.markdown("*Biomedical IR System*")
        st.divider()

        st.markdown("### ⚙️ Search Settings")
        method = st.selectbox(
            "Retrieval Method",
            ["Hybrid RRF", "BM25", "TF-IDF", "Semantic Search"],
            index=0,
            key="method_select",
        )
        top_k = st.select_slider(
            "Top K Results",
            options=[5, 10, 20],
            value=10,
            key="top_k_select",
        )

        st.divider()
        st.markdown("### 🔧 BM25 Parameters")
        bm25_k1 = st.slider("k1 (term frequency saturation)", 0.5, 3.0, 1.5, 0.1)
        bm25_b = st.slider("b (length normalisation)", 0.0, 1.0, 0.75, 0.05)

        st.divider()
        st.markdown("### 📌 Example Queries")
        examples = [
            "vitamin D deficiency",
            "dietary fiber diabetes",
            "cardiovascular health omega-3",
            "magnesium hypertension",
            "cancer prevention diet",
            "gut microbiome inflammation",
        ]
        for ex in examples:
            if st.button(f"→ {ex}", key=f"ex_{ex}", use_container_width=True):
                st.session_state["search_query"] = ex

        st.divider()
        st.caption("NFCorpus · Boteva et al. (ECIR 2016)")
        st.caption("Model: all-MiniLM-L6-v2 (pretrained)")

    return method, top_k, bm25_k1, bm25_b


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: FORMAT SNIPPET
# ══════════════════════════════════════════════════════════════════════════════

def _snippet(text: str, max_chars: int = 240) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def render_search_tab(engine, method, top_k):
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">MedIR</div>
        <div class="hero-subtitle">Biomedical Information Retrieval System</div>
        <div class="hero-badge">🧬 NFCorpus · TF-IDF · BM25 · Sentence Transformers · RRF</div>
    </div>
    """, unsafe_allow_html=True)

    # Search bar
    query = st.text_input(
        label="Search query",
        placeholder="What medical or nutritional information are you looking for?",
        value=st.session_state.get("search_query", ""),
        key="search_input",
        label_visibility="collapsed",
    )

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        search_clicked = st.button("🔎 Search", use_container_width=True, type="primary")

    if query:
        st.session_state["search_query"] = query

    if not query:
        # Landing state
        st.markdown("---")
        st.markdown("#### 💡 Try these example queries")
        c1, c2 = st.columns(2)
        examples_left = [
            "What are the effects of vitamin D deficiency?",
            "How does dietary fiber affect diabetes?",
        ]
        examples_right = [
            "What foods are associated with cardiovascular health?",
            "What is the relationship between magnesium and hypertension?",
        ]
        for ex in examples_left:
            if c1.button(ex, key=f"land_{ex}"):
                st.session_state["search_query"] = ex
                st.rerun()
        for ex in examples_right:
            if c2.button(ex, key=f"land_{ex}"):
                st.session_state["search_query"] = ex
                st.rerun()
        return

    # ── Query analysis ────────────────────────────────────────────────────────
    q_analysis = analyse_query(query)

    detected_html = ""
    if q_analysis["detected_terms"]:
        for term, expansion in q_analysis["detected_terms"]:
            detected_html += f'<span class="term-badge">{term} → {expansion}</span> '
    else:
        detected_html = "<span style='color:#64748b;font-size:0.85rem'>None detected</span>"

    st.markdown(f"""
    <div class="query-analysis">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div>
                <div class="qa-label">Original Query</div>
                <div class="qa-value">{q_analysis['original_query']}</div>
            </div>
            <div>
                <div class="qa-label">Processed Query</div>
                <div class="qa-value" style="font-family:monospace;font-size:0.82rem">{q_analysis['processed_query'][:120]}{'…' if len(q_analysis['processed_query']) > 120 else ''}</div>
            </div>
            <div>
                <div class="qa-label">Medical Terms Detected</div>
                <div>{detected_html}</div>
            </div>
            <div>
                <div class="qa-label">Expanded Query</div>
                <div class="qa-value">{q_analysis['expanded_query'][:120]}{'…' if len(q_analysis['expanded_query']) > 120 else ''}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Run search ────────────────────────────────────────────────────────────
    method_map = {
        "TF-IDF":       engine.search_tfidf,
        "BM25":         engine.search_bm25,
        "Semantic Search": engine.search_dense,
        "Hybrid RRF":   engine.search_rrf,
    }
    search_fn = method_map[method]

    t0 = time.perf_counter()
    results = search_fn(query, top_k=top_k)
    elapsed = time.perf_counter() - t0

    if not results:
        st.warning("No results found. Try a different query.")
        return

    # Results header
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f'<div class="section-header">📄 Results — {method}</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div style="text-align:right;padding-top:1.5rem;color:#64748b;font-size:0.82rem">
            {len(results)} results · {elapsed*1000:.1f} ms
        </div>""", unsafe_allow_html=True)

    # ── Result cards ──────────────────────────────────────────────────────────
    is_rrf = method == "Hybrid RRF"
    for r in results:
        with st.container():
            # Score display
            if is_rrf:
                score_label = f"RRF Score: {r.get('rrf_score', r['score']):.5f}"
            else:
                score_label = f"Score: {r['score']:.4f}"

            # RRF rank breakdown
            rrf_html = ""
            if is_rrf:
                tr = r.get("tfidf_rank", "—")
                br = r.get("bm25_rank", "—")
                dr = r.get("dense_rank", "—")
                rrf_html = (
                    '<div class="rrf-ranks">'
                    f'<span class="rrf-badge">📊 TF-IDF Rank: {tr if tr else "—"}</span>'
                    f'<span class="rrf-badge">🔤 BM25 Rank: {br if br else "—"}</span>'
                    f'<span class="rrf-badge">🧠 Dense Rank: {dr if dr else "—"}</span>'
                    '</div>'
                )

            card_html = (
                '<div class="result-card">'
                f'<div class="result-rank">#{r["rank"]}</div>'
                f'<div class="result-title">{r["title"]}</div>'
                f'<div class="result-docid">{r["doc_id"]}</div>'
                f'<div class="result-score">{score_label}</div>'
                f'{rrf_html}'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            with st.expander(f"📖 Full document — {r['doc_id']}"):
                st.markdown(f"**Title:** {r['title']}")
                st.markdown(f"**Document ID:** `{r['doc_id']}`")
                st.markdown(f"**{score_label}**")
                st.divider()
                st.write(r["text"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def render_evaluation_tab(engine, stats):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f, #2563eb);
                border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;
                box-shadow:0 4px 20px rgba(37,99,235,0.2);">
        <h2 style="color:white;margin:0;font-size:1.8rem;">📊 System Evaluation</h2>
        <p style="color:rgba(255,255,255,0.8);margin:0.25rem 0 0 0;">
            NFCorpus test set · titles queries · 3-2-1 graded relevance judgements
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset statistics ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Dataset Statistics</div>', unsafe_allow_html=True)

    grade_dist = stats.get("grade_distribution", {})
    cols = st.columns(6)
    stat_items = [
        ("Documents", f"{stats['num_documents']:,}"),
        ("Queries", f"{stats['num_queries']:,}"),
        ("Qrel Pairs", f"{stats['num_relevance_judgments']:,}"),
        ("Vocabulary", f"{stats['vocabulary_size']:,}"),
        ("Avg Doc Len", f"{stats['avg_doc_length']}"),
        ("Avg Query Len", f"{stats['avg_query_length']}"),
    ]
    for col, (label, value) in zip(cols, stat_items):
        with col:
            st.metric(label=label, value=value)

    # Grade distribution
    col_chart, col_info = st.columns([2, 3])
    with col_chart:
        if grade_dist:
            fig_grade = plot_grade_distribution(grade_dist)
            st.plotly_chart(fig_grade, use_container_width=True)
    with col_info:
        st.markdown("""
        <div class="about-section" style="margin-top:1rem">
            <div class="about-title">Relevance Grading Scheme (3-2-1)</div>
            <p style="font-size:0.88rem;color:#374151">
                NFCorpus uses a 3-level graded relevance scheme:
            </p>
            <ul style="font-size:0.88rem;color:#374151;line-height:2">
                <li><strong>Grade 3</strong> — Direct link (highly relevant)</li>
                <li><strong>Grade 2</strong> — Indirect link (relevant)</li>
                <li><strong>Grade 1</strong> — Marginal relevance</li>
                <li><strong>Grade 0</strong> — Not relevant (absent from qrel file)</li>
            </ul>
            <div class="info-box">
                Binary metrics (P@K, R@K, MAP) use threshold ≥ 1 as "relevant".<br>
                nDCG preserves full graded scores for a richer signal.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Run evaluation ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Evaluation Results</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Evaluation runs over all NFCorpus test queries that have relevance judgements.
        Results are computed from actual NFCorpus data — no values are hardcoded.
    </div>
    """, unsafe_allow_html=True)

    run_eval = st.button("▶ Run Full Evaluation", type="primary", key="run_eval_btn")

    if run_eval or st.session_state.get("eval_results") is not None:
        if run_eval or st.session_state.get("eval_results") is None:
            with st.spinner("⏳ Evaluating all four retrieval systems…"):
                eval_results = _run_evaluation(id(engine))
            st.session_state["eval_results"] = eval_results
        else:
            eval_results = st.session_state["eval_results"]

        # ── Performance table ─────────────────────────────────────────────────
        st.markdown("#### 📋 Performance Summary")
        table_data = []
        for model, res in eval_results.items():
            row = {
                "Model":         model,
                "Precision@1":   round(res["precision_at_k"].get(1, 0), 4),
                "Precision@3":   round(res["precision_at_k"].get(3, 0), 4),
                "Precision@5":   round(res["precision_at_k"].get(5, 0), 4),
                "Precision@10":  round(res["precision_at_k"].get(10, 0), 4),
                "Recall@10":     round(res["recall_at_k"].get(10, 0), 4),
                "MAP":           round(res["map"], 4),
                "nDCG@10":       round(res["ndcg_10"], 4),
                "Queries Eval.": res["num_queries_evaluated"],
            }
            table_data.append(row)

        df_table = pd.DataFrame(table_data).set_index("Model")

        # Highlight best values
        def highlight_max(s):
            is_max = s == s.max()
            return ["background-color:#dbeafe;font-weight:600;color:#1d4ed8" if v else "" for v in is_max]

        numeric_cols = [c for c in df_table.columns if c != "Queries Eval."]
        styled = df_table.style.apply(highlight_max, subset=numeric_cols, axis=0)
        st.dataframe(styled, use_container_width=True, height=200)

        st.divider()

        # ── Graph 1: Model comparison ─────────────────────────────────────────
        st.markdown("#### 📊 Graph 1 — Model Comparison")
        fig1 = plot_model_comparison(eval_results)
        st.plotly_chart(fig1, use_container_width=True)

        # ── Graph 2 & 3: P@K and R@K ─────────────────────────────────────────
        col_g2, col_g3 = st.columns(2)
        with col_g2:
            st.markdown("#### 📈 Graph 2 — Precision@K")
            fig2 = plot_precision_at_k(eval_results)
            st.plotly_chart(fig2, use_container_width=True)
        with col_g3:
            st.markdown("#### 📈 Graph 3 — Recall@K")
            fig3 = plot_recall_at_k(eval_results)
            st.plotly_chart(fig3, use_container_width=True)

        # ── Graph 4: P-R curve ────────────────────────────────────────────────
        st.markdown("#### 📉 Graph 4 — Precision-Recall Curves")
        fig4 = plot_precision_recall_curve(eval_results)
        st.plotly_chart(fig4, use_container_width=True)

        # ── Graph 5: Latency ──────────────────────────────────────────────────
        st.markdown("#### ⏱ Graph 5 — Retrieval Latency")
        with st.spinner("Measuring latency…"):
            latencies = engine.measure_latency(
                "vitamin D deficiency inflammation", top_k=10, n_runs=5
            )
        fig5 = plot_latency(latencies)
        st.plotly_chart(fig5, use_container_width=True)

        # ── Graph 6: Overlap heatmap ──────────────────────────────────────────
        st.markdown("#### 🗺 Graph 6 — Retrieval Overlap (Jaccard)")
        test_q = "vitamin D deficiency inflammation"
        overlap_results = {
            "TF-IDF":      engine.search_tfidf(test_q, top_k=10),
            "BM25":        engine.search_bm25(test_q, top_k=10),
            "Semantic":    engine.search_dense(test_q, top_k=10),
            "Hybrid RRF":  engine.search_rrf(test_q, top_k=10),
        }
        fig6 = plot_overlap_heatmap(overlap_results)
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            💡 <strong>Interpretation:</strong> Low Jaccard similarity between TF-IDF/BM25 and Semantic
            shows they retrieve different documents — this diversity is why RRF fusion improves
            overall retrieval quality.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def render_about_tab():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f, #2563eb);
                border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;
                box-shadow:0 4px 20px rgba(37,99,235,0.2);">
        <h2 style="color:white;margin:0;font-size:1.8rem;">ℹ️ About MedIR</h2>
        <p style="color:rgba(255,255,255,0.8);margin:0.25rem 0 0 0;">
            Biomedical Information Retrieval · Academic Research Prototype
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Problem statement
    st.markdown("""
    <div class="about-section">
        <div class="about-title">🎯 Problem Statement</div>
        <p style="font-size:0.9rem;color:#374151;line-height:1.7">
            Medical and nutritional literature is vast and dense, making efficient
            information retrieval a critical challenge for researchers and health
            practitioners. Standard keyword search often fails to surface semantically
            related content because biomedical text contains specialised terminology,
            abbreviations, and domain-specific phrasing.
        </p>
        <p style="font-size:0.9rem;color:#374151;line-height:1.7">
            MedIR addresses this challenge by combining <strong>classical IR methods</strong>
            (TF-IDF, BM25) with <strong>neural semantic search</strong> (Sentence Transformers),
            and fusing them via <strong>Reciprocal Rank Fusion</strong> to improve retrieval
            quality beyond any single method.  The system is evaluated on the
            <strong>NFCorpus</strong> benchmark (Boteva et al., ECIR 2016).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # System architecture
    st.markdown("""
    <div class="about-section">
        <div class="about-title">🏗️ System Architecture</div>
        <div class="pipeline-box">
User Query
    │
    ▼
Text Preprocessing (lowercase · tokenise · stopword removal · stemming)
    │
    ▼
Medical Term Expansion (abbreviation → full term)
    │
    ├─────────────────┬──────────────────┐
    ▼                 ▼                  ▼
TF-IDF            BM25          Sentence Transformer
(sklearn)     (rank-bm25)     (all-MiniLM-L6-v2)
cosine sim      BM25Okapi       cosine similarity
    │                 │                  │
    └─────────────────┴──────────────────┘
                      │
                      ▼
         Reciprocal Rank Fusion (k=60)
                      │
                      ▼
              Ranked Results
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Algorithms
    st.markdown('<div class="section-header">⚙️ Algorithms</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="about-section">
            <div class="about-title">TF-IDF</div>
            <p style="font-size:0.88rem;color:#374151;line-height:1.7">
                Term Frequency–Inverse Document Frequency weights terms by how often they
                appear in a document (TF) relative to how rare they are across the whole
                corpus (IDF). Documents are represented as sparse vectors, and queries
                are matched via <strong>cosine similarity</strong>.
            </p>
            <code style="font-size:0.82rem;background:#f1f5f9;padding:0.25rem 0.5rem;border-radius:4px">
                tfidf(t,d) = tf(t,d) × log(N / df(t))
            </code>
        </div>

        <div class="about-section">
            <div class="about-title">BM25 (Okapi BM25)</div>
            <p style="font-size:0.88rem;color:#374151;line-height:1.7">
                An improved probabilistic ranking function that adds
                <strong>term frequency saturation</strong> (via k₁) and
                <strong>document length normalisation</strong> (via b).
                It is the de-facto standard for classical IR.
            </p>
            <code style="font-size:0.82rem;background:#f1f5f9;padding:0.25rem 0.5rem;border-radius:4px">
                BM25 = IDF × tf·(k₁+1) / (tf + k₁·(1-b+b·dl/avgdl))
            </code>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="about-section">
            <div class="about-title">Sentence Transformers (Dense Retrieval)</div>
            <p style="font-size:0.88rem;color:#374151;line-height:1.7">
                The pretrained <strong>all-MiniLM-L6-v2</strong> model encodes text into
                384-dimensional dense vectors that capture semantic meaning.
                Documents are encoded once and cached; at query time the query embedding
                is compared against all document embeddings via <strong>cosine similarity</strong>.
                <em>No training is performed — the model is used off-the-shelf.</em>
            </p>
        </div>

        <div class="about-section">
            <div class="about-title">Reciprocal Rank Fusion (RRF)</div>
            <p style="font-size:0.88rem;color:#374151;line-height:1.7">
                RRF combines multiple ranked lists without requiring score normalisation.
                Each document receives a score based on its rank in each list:
            </p>
            <code style="font-size:0.82rem;background:#f1f5f9;padding:0.25rem 0.5rem;border-radius:4px;display:block">
                RRF(d) = Σᵢ 1 / (k + rankᵢ(d))
            </code>
            <p style="font-size:0.85rem;color:#374151;margin-top:0.5rem">
                where k=60 is a smoothing constant that reduces the impact of very high
                ranks and makes the fusion robust to outliers.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Evaluation metrics
    st.markdown('<div class="section-header">📏 Evaluation Metrics</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div class="about-section">
            <div class="about-title">Precision@K & Recall@K</div>
            <p style="font-size:0.88rem;color:#374151;line-height:1.7">
                <strong>Precision@K</strong> measures the fraction of the top-K retrieved
                documents that are relevant.<br>
                <strong>Recall@K</strong> measures the fraction of all relevant documents
                that appear in the top-K results.
            </p>
            <code style="font-size:0.82rem;background:#f1f5f9;padding:0.25rem 0.5rem;border-radius:4px;display:block">
                P@K = |Rel ∩ TopK| / K<br>R@K = |Rel ∩ TopK| / |Rel|
            </code>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="about-section">
            <div class="about-title">MAP & nDCG</div>
            <p style="font-size:0.88rem;color:#374151;line-height:1.7">
                <strong>MAP</strong> (Mean Average Precision) averages the precision at
                each relevant document's rank across all queries — rewarding systems that
                rank relevant documents higher.<br><br>
                <strong>nDCG</strong> (normalised Discounted Cumulative Gain) uses
                <em>graded</em> relevance, discounting gains at lower ranks via
                log₂(rank+1). nDCG = DCG / IDCG where IDCG is the ideal ordering.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Dataset reference
    st.markdown("""
    <div class="about-section">
        <div class="about-title">📖 Dataset — NFCorpus</div>
        <p style="font-size:0.88rem;color:#374151;line-height:1.7">
            NFCorpus (NutritionFacts Corpus) is a full-text biomedical information retrieval
            benchmark. It consists of medical abstracts (PubMed) paired with
            NutritionFacts.org video queries, with graded relevance judgements.
        </p>
        <p style="font-size:0.85rem;color:#64748b">
            Boteva, V., Gholipour, D., Sokolov, A., &amp; Riezler, S. (2016).
            <em>A Full-Text Learning to Rank Dataset for Medical Information Retrieval.</em>
            ECIR 2016, Springer.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    method, top_k, bm25_k1, bm25_b = render_sidebar()

    # Loading state
    with st.spinner("🔄 Loading NFCorpus dataset and building indices…"):
        _, _, _, stats = _load_raw_dataset()
        engine = _build_search_engine(bm25_k1=bm25_k1, bm25_b=bm25_b)

    # Tabs
    tab_search, tab_eval, tab_about = st.tabs([
        "🔎 Search", "📊 Evaluation", "ℹ️ About"
    ])

    with tab_search:
        render_search_tab(engine, method, top_k)

    with tab_eval:
        render_evaluation_tab(engine, stats)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
