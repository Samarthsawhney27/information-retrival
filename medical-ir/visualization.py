"""
visualization.py
----------------
Plotly chart builders for the MedIR Streamlit application.
All charts are built from actual evaluation data — nothing is fabricated.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Colour palette ───────────────────────────────────────────────────────────
MODEL_COLORS = {
    "TF-IDF":      "#3B82F6",   # blue
    "BM25":        "#10B981",   # emerald
    "Semantic":    "#8B5CF6",   # violet
    "Hybrid RRF":  "#F59E0B",   # amber
}

CHART_LAYOUT = dict(
    font_family="Inter, -apple-system, sans-serif",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,1)",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E2E8F0",
        borderwidth=1,
    ),
)


# ── 1. Model comparison bar chart ────────────────────────────────────────────

def plot_model_comparison(eval_results: dict) -> go.Figure:
    """
    Grouped bar chart comparing Precision@10, Recall@10, MAP, nDCG@10
    across all four models.
    """
    models = list(eval_results.keys())
    metrics = {
        "Precision@10": [eval_results[m]["precision_at_k"].get(10, 0) for m in models],
        "Recall@10":    [eval_results[m]["recall_at_k"].get(10, 0) for m in models],
        "MAP":          [eval_results[m]["map"] for m in models],
        "nDCG@10":      [eval_results[m]["ndcg_10"] for m in models],
    }

    metric_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]

    fig = go.Figure()
    for (metric, values), color in zip(metrics.items(), metric_colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=values,
            marker_color=color,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color="#374151"),
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Model Comparison — Retrieval Metrics", font_size=16, x=0.5),
        barmode="group",
        yaxis=dict(
            title="Score",
            range=[0, max(max(v) for v in metrics.values()) * 1.25],
            gridcolor="#E2E8F0",
            zeroline=False,
        ),
        xaxis=dict(title="Retrieval Method"),
        height=420,
    )
    return fig


# ── 2. Precision@K line chart ────────────────────────────────────────────────

def plot_precision_at_k(eval_results: dict, k_values=(1, 3, 5, 10)) -> go.Figure:
    """Line chart: Precision@K for each model."""
    fig = go.Figure()
    for model, results in eval_results.items():
        p_values = [results["precision_at_k"].get(k, 0) for k in k_values]
        fig.add_trace(go.Scatter(
            x=list(k_values),
            y=p_values,
            name=model,
            mode="lines+markers",
            line=dict(color=MODEL_COLORS.get(model, "#6B7280"), width=2.5),
            marker=dict(size=8, symbol="circle"),
            hovertemplate=f"<b>{model}</b><br>K=%{{x}}<br>Precision=%{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Precision@K", font_size=16, x=0.5),
        xaxis=dict(title="K", tickmode="array", tickvals=list(k_values), gridcolor="#E2E8F0"),
        yaxis=dict(title="Precision", gridcolor="#E2E8F0", zeroline=False),
        height=380,
    )
    return fig


# ── 3. Recall@K line chart ───────────────────────────────────────────────────

def plot_recall_at_k(eval_results: dict, k_values=(1, 3, 5, 10)) -> go.Figure:
    """Line chart: Recall@K for each model."""
    fig = go.Figure()
    for model, results in eval_results.items():
        r_values = [results["recall_at_k"].get(k, 0) for k in k_values]
        fig.add_trace(go.Scatter(
            x=list(k_values),
            y=r_values,
            name=model,
            mode="lines+markers",
            line=dict(color=MODEL_COLORS.get(model, "#6B7280"), width=2.5),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate=f"<b>{model}</b><br>K=%{{x}}<br>Recall=%{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Recall@K", font_size=16, x=0.5),
        xaxis=dict(title="K", tickmode="array", tickvals=list(k_values), gridcolor="#E2E8F0"),
        yaxis=dict(title="Recall", gridcolor="#E2E8F0", zeroline=False),
        height=380,
    )
    return fig


# ── 4. Precision-Recall curve ────────────────────────────────────────────────

def plot_precision_recall_curve(eval_results: dict) -> go.Figure:
    """
    Interpolated Precision-Recall curves for all models.
    Points come from actual NFCorpus evaluation — not fabricated.
    """
    fig = go.Figure()
    for model, results in eval_results.items():
        curve = results.get("avg_pr_curve", [])
        if not curve:
            continue
        recalls = [pt[0] for pt in curve]
        precs = [pt[1] for pt in curve]
        fig.add_trace(go.Scatter(
            x=recalls,
            y=precs,
            name=model,
            mode="lines+markers",
            line=dict(color=MODEL_COLORS.get(model, "#6B7280"), width=2.5),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor=MODEL_COLORS.get(model, "#6B7280").replace(")", ",0.05)").replace("rgb", "rgba").replace("#3B82F6", "rgba(59,130,246,0.05)").replace("#10B981", "rgba(16,185,129,0.05)").replace("#8B5CF6", "rgba(139,92,246,0.05)").replace("#F59E0B", "rgba(245,158,11,0.05)"),
            hovertemplate=f"<b>{model}</b><br>Recall=%{{x:.2f}}<br>Precision=%{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Precision-Recall Curves (Interpolated)", font_size=16, x=0.5),
        xaxis=dict(title="Recall", range=[0, 1.05], gridcolor="#E2E8F0"),
        yaxis=dict(title="Precision", range=[0, 1.05], gridcolor="#E2E8F0", zeroline=False),
        height=420,
    )
    return fig


# ── 5. Retrieval latency bar chart ───────────────────────────────────────────

def plot_latency(latencies: dict) -> go.Figure:
    """
    Horizontal bar chart showing retrieval latency in milliseconds.
    """
    models = list(latencies.keys())
    times_ms = [latencies[m] * 1000 for m in models]
    colors = [MODEL_COLORS.get(m, "#6B7280") for m in models]

    fig = go.Figure(go.Bar(
        x=times_ms,
        y=models,
        orientation="h",
        marker_color=colors,
        text=[f"{t:.1f} ms" for t in times_ms],
        textposition="outside",
        textfont=dict(size=12, color="#374151"),
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Retrieval Latency (avg over 5 runs)", font_size=16, x=0.5),
        xaxis=dict(title="Latency (ms)", gridcolor="#E2E8F0"),
        yaxis=dict(title="Method"),
        height=320,
    )
    return fig


# ── 6. Retrieval overlap heatmap ─────────────────────────────────────────────

def plot_overlap_heatmap(search_results: dict) -> go.Figure:
    """
    Heatmap of Jaccard similarity between top-10 retrieval sets from each model.

    Parameters
    ----------
    search_results : dict
        {model_name: [result_dicts]}  — one set of results per model
    """
    models = list(search_results.keys())
    n = len(models)
    matrix = np.zeros((n, n))

    sets = {m: set(r["doc_id"] for r in search_results[m]) for m in models}

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                matrix[i][j] = 1.0
            else:
                a, b = sets[m1], sets[m2]
                union = len(a | b)
                matrix[i][j] = len(a & b) / union if union else 0.0

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=models,
        y=models,
        colorscale="Blues",
        zmin=0,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=14, color="black"),
        hovertemplate="<b>%{y}</b> ∩ <b>%{x}</b><br>Jaccard = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Jaccard"),
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Retrieval Overlap Heatmap (Jaccard Similarity)", font_size=16, x=0.5),
        height=360,
    )
    return fig


# ── Dataset statistics donut chart ───────────────────────────────────────────

def plot_grade_distribution(grade_dist: dict) -> go.Figure:
    """Donut chart showing relevance grade distribution."""
    grade_labels = {
        3: "Highly Relevant (3)",
        2: "Relevant (2)",
        1: "Marginal (1)",
    }
    labels = [grade_labels.get(k, str(k)) for k in sorted(grade_dist.keys())]
    values = [grade_dist[k] for k in sorted(grade_dist.keys())]
    colors = ["#10B981", "#3B82F6", "#F59E0B"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Relevance Grade Distribution", font_size=14, x=0.5),
        height=320,
        showlegend=False,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig
