"""
evaluation.py
-------------
Evaluation metrics for the MedIR Information Retrieval system.

Metrics implemented:
  - Precision@K  (K = 1, 3, 5, 10)
  - Recall@K     (K = 1, 3, 5, 10)
  - MAP          (Mean Average Precision)
  - nDCG@K       (Normalised Discounted Cumulative Gain)
  - Precision-Recall curve data

NFCorpus uses graded relevance (3-2-1 scale):
  3 — direct link (highly relevant)
  2 — indirect link (relevant)
  1 — marginally relevant
  0 — not relevant (not in qrel file)

For binary-threshold metrics (Precision@K, Recall@K, MAP):
  We treat documents with relevance >= REL_THRESHOLD as relevant.
  REL_THRESHOLD = 1  (marginal relevance and above counts)

For nDCG we preserve the graded scores as-is.

Nothing in this module is hardcoded — all values are computed from
the actual NFCorpus test queries and qrels.
"""

import math
from collections import defaultdict
import numpy as np

# Minimum relevance grade to be considered "relevant" for binary metrics
REL_THRESHOLD = 1


# ── Per-query metric helpers ─────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant: dict, k: int) -> float:
    """
    Precision@K for a single query.

    Parameters
    ----------
    retrieved : list of doc_ids (ordered, highest score first)
    relevant  : {doc_id: relevance_grade}  — from qrels
    k         : cutoff
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if relevant.get(d, 0) >= REL_THRESHOLD)
    return hits / k


def recall_at_k(retrieved: list, relevant: dict, k: int) -> float:
    """
    Recall@K for a single query.
    """
    total_relevant = sum(1 for v in relevant.values() if v >= REL_THRESHOLD)
    if total_relevant == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if relevant.get(d, 0) >= REL_THRESHOLD)
    return hits / total_relevant


def average_precision(retrieved: list, relevant: dict) -> float:
    """
    Average Precision for a single query.
    """
    total_relevant = sum(1 for v in relevant.values() if v >= REL_THRESHOLD)
    if total_relevant == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if relevant.get(doc_id, 0) >= REL_THRESHOLD:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_relevant


def dcg_at_k(retrieved: list, relevant: dict, k: int) -> float:
    """
    DCG@K using graded relevance.
    DCG = Σ (2^rel - 1) / log2(rank + 1)
    """
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevant.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: list, relevant: dict, k: int) -> float:
    """
    nDCG@K using graded relevance.
    nDCG = DCG / IDCG   (IDCG is the ideal DCG)
    """
    # Ideal ranking: sort relevant docs by grade descending
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


# ── Precision-Recall curve ───────────────────────────────────────────────────

def precision_recall_curve(retrieved: list, relevant: dict, num_points: int = 11):
    """
    Compute interpolated Precision-Recall curve points for a single query.

    Returns
    -------
    list of (recall, precision) pairs at standard recall levels 0.0, 0.1, …, 1.0
    """
    total_relevant = sum(1 for v in relevant.values() if v >= REL_THRESHOLD)
    if total_relevant == 0:
        return [(r / 10, 0.0) for r in range(11)]

    # Raw curve
    hits = 0
    raw_points = []
    for rank, doc_id in enumerate(retrieved, start=1):
        if relevant.get(doc_id, 0) >= REL_THRESHOLD:
            hits += 1
            prec = hits / rank
            rec = hits / total_relevant
            raw_points.append((rec, prec))

    if not raw_points:
        return [(r / 10, 0.0) for r in range(11)]

    # Interpolated at standard recall levels
    recall_levels = [i / 10 for i in range(11)]
    interpolated = []
    for rec_level in recall_levels:
        # Max precision at recall >= rec_level
        max_prec = max(
            (p for r, p in raw_points if r >= rec_level), default=0.0
        )
        interpolated.append((rec_level, max_prec))
    return interpolated


# ── System-level evaluation ──────────────────────────────────────────────────

def evaluate_system(
    search_fn,
    queries: dict,
    qrels: dict,
    top_k: int = 10,
    k_values: tuple = (1, 3, 5, 10),
) -> dict:
    """
    Evaluate an IR system over all test queries.

    Parameters
    ----------
    search_fn : callable
        Function with signature search_fn(query_text, top_k=int) → list of result dicts.
        Each result dict must have at least a 'doc_id' key.
    queries : dict
        {query_id: {"query": str}}
    qrels : dict
        {query_id: {doc_id: relevance_grade}}
    top_k : int
        Max documents to retrieve per query.
    k_values : tuple
        K values for Precision@K and Recall@K.

    Returns
    -------
    dict
        Aggregated metrics plus per-query details.
    """
    # Only evaluate queries that have relevance judgements
    eval_queries = {qid: q for qid, q in queries.items() if qid in qrels}

    per_query_results = {}
    p_at_k = defaultdict(list)
    r_at_k = defaultdict(list)
    ap_scores = []
    ndcg_scores = []
    pr_curve_data = defaultdict(list)  # {recall_level: [precision, ...]}

    for qid, q_info in eval_queries.items():
        query_text = q_info["query"]
        relevant = qrels[qid]

        try:
            results = search_fn(query_text, top_k=max(top_k, 100))
        except Exception:
            continue

        retrieved = [r["doc_id"] for r in results]

        # Precision@K and Recall@K
        for k in k_values:
            p_at_k[k].append(precision_at_k(retrieved, relevant, k))
            r_at_k[k].append(recall_at_k(retrieved, relevant, k))

        # Average Precision
        ap = average_precision(retrieved, relevant)
        ap_scores.append(ap)

        # nDCG@10
        ndcg = ndcg_at_k(retrieved, relevant, k=10)
        ndcg_scores.append(ndcg)

        # P-R curve
        curve_points = precision_recall_curve(retrieved, relevant)
        for rec, prec in curve_points:
            pr_curve_data[rec].append(prec)

        per_query_results[qid] = {
            "ap": ap,
            "ndcg_10": ndcg,
            "p_at_k": {k: precision_at_k(retrieved, relevant, k) for k in k_values},
            "r_at_k": {k: recall_at_k(retrieved, relevant, k) for k in k_values},
        }

    # Aggregate
    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    aggregated_p = {k: mean(v) for k, v in p_at_k.items()}
    aggregated_r = {k: mean(v) for k, v in r_at_k.items()}

    # Interpolated P-R curve (mean over queries)
    avg_pr_curve = [
        (rec, mean(pr_curve_data[rec]))
        for rec in sorted(pr_curve_data.keys())
    ]

    return {
        "num_queries_evaluated": len(per_query_results),
        "precision_at_k": aggregated_p,
        "recall_at_k": aggregated_r,
        "map": mean(ap_scores),
        "ndcg_10": mean(ndcg_scores),
        "avg_pr_curve": avg_pr_curve,
        "per_query": per_query_results,
    }


def compare_systems(
    systems: dict,
    queries: dict,
    qrels: dict,
    top_k: int = 10,
) -> dict:
    """
    Evaluate multiple IR systems and collect results for comparison.

    Parameters
    ----------
    systems : dict
        {system_name: search_fn}
    queries : dict
    qrels : dict

    Returns
    -------
    dict
        {system_name: evaluation_result_dict}
    """
    return {
        name: evaluate_system(fn, queries, qrels, top_k=top_k)
        for name, fn in systems.items()
    }
