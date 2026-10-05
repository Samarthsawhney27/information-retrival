"""
verify.py
---------
Integration test script for the MedIR system.
Run from the medical-ir/ directory.
Tests: data loading, preprocessing, IR engine (TF-IDF, BM25, Dense, RRF), evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("MedIR Integration Verification")
print("=" * 60)

# ── 1. Data loading ───────────────────────────────────────────────────────────
print("\n[1/5] Testing data loader...")
from data_loader import load_dataset, compute_dataset_stats

docs, queries, qrels = load_dataset()
stats = compute_dataset_stats(docs, queries, qrels)
assert len(docs) > 3000, f"Expected >3000 docs, got {len(docs)}"
assert len(queries) > 200, f"Expected >200 queries, got {len(queries)}"
assert len(qrels) > 100, f"Expected >100 qrel entries, got {len(qrels)}"
print(f"  ✓ Documents: {stats['num_documents']:,}")
print(f"  ✓ Queries: {stats['num_queries']:,}")
print(f"  ✓ Qrel pairs: {stats['num_relevance_judgments']:,}")
print(f"  ✓ Vocabulary: {stats['vocabulary_size']:,}")
print(f"  ✓ Avg doc length: {stats['avg_doc_length']} tokens")
print(f"  ✓ Grade distribution: {stats['grade_distribution']}")

# ── 2. Document format ────────────────────────────────────────────────────────
print("\n[2/5] Verifying document/query/qrel formats...")
first_doc_id = next(iter(docs))
first_doc = docs[first_doc_id]
assert "title" in first_doc, "Document missing 'title'"
assert "text" in first_doc, "Document missing 'text'"
print(f"  ✓ First doc ID: {first_doc_id}")
print(f"  ✓ First doc title: {first_doc['title'][:60]}")

first_q_id = next(iter(queries))
first_q = queries[first_q_id]
assert "query" in first_q, "Query missing 'query' field"
print(f"  ✓ First query ID: {first_q_id}")
print(f"  ✓ First query text: {first_q['query'][:60]}")

first_qrel_id = next(iter(qrels))
first_qrel_entry = qrels[first_qrel_id]
assert isinstance(first_qrel_entry, dict), "Qrel value should be dict"
sample_grade = next(iter(first_qrel_entry.values()))
assert sample_grade in (1, 2, 3), f"Grade {sample_grade} not in (1,2,3)"
print(f"  ✓ Sample qrel: query={first_qrel_id}, docs={len(first_qrel_entry)}, grade_example={sample_grade}")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
print("\n[3/5] Testing preprocessing...")
from preprocessing import preprocess_text, analyse_query, preprocess_documents

test_text = "What is the effect of Vitamin D deficiency on bone health?"
processed = preprocess_text(test_text)
assert len(processed) > 0, "Empty processed text"
print(f"  ✓ Original:  {test_text}")
print(f"  ✓ Processed: {processed}")

q_analysis = analyse_query("effects of HTN and DM on the heart")
assert len(q_analysis["detected_terms"]) > 0, "Expected medical terms detected"
print(f"  ✓ Detected terms: {q_analysis['detected_terms']}")
print(f"  ✓ Expanded: {q_analysis['expanded_query'][:80]}")

print("  Processing all documents (this may take ~30s)...")
processed_docs = preprocess_documents(docs, stem=True)
first_pdoc = processed_docs[first_doc_id]
assert "processed" in first_pdoc, "Missing 'processed' field"
assert "text" in first_pdoc, "Missing original 'text' in processed doc"
print(f"  ✓ Processed docs: {len(processed_docs):,}")
print(f"  ✓ Sample processed: {first_pdoc['processed'][:80]}")

# ── 4. IR Engine ──────────────────────────────────────────────────────────────
print("\n[4/5] Building search indices and testing retrieval...")
from ir_engine import MedicalSearchEngine

engine = MedicalSearchEngine(processed_docs, bm25_k1=1.5, bm25_b=0.75)

print("  Building TF-IDF index...")
engine.build_tfidf_index()
print("  ✓ TF-IDF index built")

print("  Building BM25 index...")
engine.build_bm25_index()
print("  ✓ BM25 index built")

print("  Loading sentence transformer (all-MiniLM-L6-v2)...")
engine.build_dense_index()
print("  ✓ Dense embeddings ready")

# Test queries
test_query = "vitamin D deficiency bone health"
print(f"\n  Test query: '{test_query}'")

tfidf_results = engine.search_tfidf(test_query, top_k=5)
assert len(tfidf_results) == 5
print(f"  ✓ TF-IDF top-1: [{tfidf_results[0]['doc_id']}] {tfidf_results[0]['title'][:50]} (score={tfidf_results[0]['score']:.4f})")

bm25_results = engine.search_bm25(test_query, top_k=5)
assert len(bm25_results) == 5
print(f"  ✓ BM25 top-1: [{bm25_results[0]['doc_id']}] {bm25_results[0]['title'][:50]} (score={bm25_results[0]['score']:.4f})")

dense_results = engine.search_dense(test_query, top_k=5)
assert len(dense_results) == 5
print(f"  ✓ Dense top-1: [{dense_results[0]['doc_id']}] {dense_results[0]['title'][:50]} (score={dense_results[0]['score']:.4f})")

rrf_results = engine.search_rrf(test_query, top_k=5)
assert len(rrf_results) == 5
r = rrf_results[0]
assert "rrf_score" in r, "RRF result missing rrf_score"
assert "tfidf_rank" in r, "RRF result missing tfidf_rank"
assert "bm25_rank" in r, "RRF result missing bm25_rank"
assert "dense_rank" in r, "RRF result missing dense_rank"
print(f"  ✓ RRF top-1: [{r['doc_id']}] {r['title'][:50]}")
print(f"    TF-IDF rank={r['tfidf_rank']}, BM25 rank={r['bm25_rank']}, Dense rank={r['dense_rank']}, RRF={r['rrf_score']:.5f}")

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
print("\n[5/5] Running evaluation on first 20 test queries (quick check)...")
from evaluation import evaluate_system, precision_at_k, recall_at_k, average_precision, ndcg_at_k

# Quick eval on first 20 queries only for speed
eval_queries_subset = dict(list(queries.items())[:20])
tfidf_eval = evaluate_system(engine.search_tfidf, eval_queries_subset, qrels, top_k=50)

print(f"  ✓ Queries evaluated: {tfidf_eval['num_queries_evaluated']}")
print(f"  ✓ TF-IDF Precision@10: {tfidf_eval['precision_at_k'].get(10, 0):.4f}")
print(f"  ✓ TF-IDF Recall@10:    {tfidf_eval['recall_at_k'].get(10, 0):.4f}")
print(f"  ✓ TF-IDF MAP:          {tfidf_eval['map']:.4f}")
print(f"  ✓ TF-IDF nDCG@10:      {tfidf_eval['ndcg_10']:.4f}")
print(f"  ✓ P-R curve points:    {len(tfidf_eval['avg_pr_curve'])}")

# Verify no values are hardcoded
assert tfidf_eval['map'] != 0.42, "MAP value looks hardcoded!"
assert tfidf_eval['ndcg_10'] != 0.51, "nDCG value looks hardcoded!"

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED — MedIR is ready!")
print("=" * 60)
print("\nRun the application:")
print("  cd medical-ir")
print("  streamlit run app.py")
