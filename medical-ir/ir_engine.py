"""
ir_engine.py
------------
MedicalSearchEngine: Implements four retrieval methods over the NFCorpus
document collection.

Methods:
  1. TF-IDF    – sklearn TfidfVectorizer + cosine similarity
  2. BM25      – rank_bm25 BM25Okapi
  3. Dense     – sentence-transformers (all-MiniLM-L6-v2) + cosine similarity
                 (pretrained, NO training performed)
  4. Hybrid RRF– Reciprocal Rank Fusion of the three methods above

All methods return a list of result dicts:
    {doc_id, title, text, score, rank}

For RRF results, additional keys are added:
    {tfidf_rank, bm25_rank, dense_rank, rrf_score}
"""

import os
import time
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from preprocessing import preprocess_text, analyse_query

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
EMBEDDINGS_DIR = _HERE / "models" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_CACHE = EMBEDDINGS_DIR / "tfidf_index.pkl"
BM25_CACHE = EMBEDDINGS_DIR / "bm25_index.pkl"
DENSE_EMBEDDINGS_CACHE = EMBEDDINGS_DIR / "dense_embeddings.npy"
DOC_IDS_CACHE = EMBEDDINGS_DIR / "doc_ids.pkl"

# ── Sentinel value for "not ranked" in RRF ───────────────────────────────────
NOT_RANKED = 999_999


# ── Result helpers ────────────────────────────────────────────────────────────

def _make_result(doc_id: str, doc: dict, score: float, rank: int) -> dict:
    return {
        "doc_id": doc_id,
        "title": doc.get("title", doc_id),
        "text": doc.get("text", ""),
        "score": float(score),
        "rank": rank,
    }


# ── MedicalSearchEngine ───────────────────────────────────────────────────────

class MedicalSearchEngine:
    """
    Biomedical Information Retrieval engine combining classical and neural methods.

    Parameters
    ----------
    documents : dict
        {doc_id: {"title": str, "text": str, "processed": str}}
        The 'processed' field should contain the preprocessed document text.
    bm25_k1 : float
        BM25 k1 parameter (default 1.5).
    bm25_b : float
        BM25 b parameter (default 0.75).
    """

    def __init__(self, documents: dict, bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.documents = documents
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        # Ordered list of doc IDs — index alignment matters for matrix ops
        self.doc_ids: list = list(documents.keys())
        self.doc_texts_processed: list = [
            documents[d].get("processed", documents[d]["text"]) for d in self.doc_ids
        ]
        self.doc_texts_original: list = [
            documents[d]["text"] for d in self.doc_ids
        ]

        # Lazy-loaded components
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._bm25: BM25Okapi | None = None
        self._dense_model: SentenceTransformer | None = None
        self._dense_embeddings: np.ndarray | None = None

    # ── Build / load indices ─────────────────────────────────────────────────

    def build_tfidf_index(self, force_rebuild: bool = False) -> None:
        """Build (or load from cache) the TF-IDF index."""
        if not force_rebuild and TFIDF_CACHE.exists():
            with open(TFIDF_CACHE, "rb") as fh:
                cached = pickle.load(fh)
            self._tfidf_vectorizer = cached["vectorizer"]
            self._tfidf_matrix = cached["matrix"]
            return

        self._tfidf_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.85,
            min_df=2,
            ngram_range=(1, 2),
            max_features=100_000,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(
            self.doc_texts_processed
        )
        with open(TFIDF_CACHE, "wb") as fh:
            pickle.dump(
                {"vectorizer": self._tfidf_vectorizer, "matrix": self._tfidf_matrix},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def build_bm25_index(self, force_rebuild: bool = False) -> None:
        """Build (or load from cache) the BM25 index."""
        if not force_rebuild and BM25_CACHE.exists():
            with open(BM25_CACHE, "rb") as fh:
                self._bm25 = pickle.load(fh)
            return

        tokenised_corpus = [text.split() for text in self.doc_texts_processed]
        self._bm25 = BM25Okapi(tokenised_corpus, k1=self.bm25_k1, b=self.bm25_b)
        with open(BM25_CACHE, "wb") as fh:
            pickle.dump(self._bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def build_dense_index(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        force_rebuild: bool = False,
    ) -> None:
        """
        Load the pretrained sentence transformer and encode all documents.
        Embeddings are cached to disk so this only runs once.
        """
        # Always load the model
        self._dense_model = SentenceTransformer(model_name)

        # Check cache
        if (
            not force_rebuild
            and DENSE_EMBEDDINGS_CACHE.exists()
            and DOC_IDS_CACHE.exists()
        ):
            cached_ids = pickle.load(open(DOC_IDS_CACHE, "rb"))
            if cached_ids == self.doc_ids:
                self._dense_embeddings = np.load(str(DENSE_EMBEDDINGS_CACHE))
                return

        # Encode documents (pretrained model — no training)
        self._dense_embeddings = self._dense_model.encode(
            self.doc_texts_original,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(str(DENSE_EMBEDDINGS_CACHE), self._dense_embeddings)
        with open(DOC_IDS_CACHE, "wb") as fh:
            pickle.dump(self.doc_ids, fh)

    def build_all_indices(self, force_rebuild: bool = False) -> None:
        """Build all three indices."""
        self.build_tfidf_index(force_rebuild=force_rebuild)
        self.build_bm25_index(force_rebuild=force_rebuild)
        self.build_dense_index(force_rebuild=force_rebuild)

    # ── Search methods ───────────────────────────────────────────────────────

    def search_tfidf(self, query: str, top_k: int = 10) -> list:
        """
        TF-IDF retrieval using cosine similarity.
        Returns list of result dicts sorted by score descending.
        """
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            raise RuntimeError("TF-IDF index not built. Call build_tfidf_index() first.")

        # Preprocess query
        q_analysis = analyse_query(query)
        q_processed = q_analysis["processed_query"]

        q_vec = self._tfidf_vectorizer.transform([q_processed])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()

        # Top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            results.append(_make_result(doc_id, self.documents[doc_id], scores[idx], rank))
        return results

    def search_bm25(self, query: str, top_k: int = 10) -> list:
        """
        BM25 retrieval (rank_bm25 BM25Okapi).
        Returns list of result dicts sorted by score descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_bm25_index() first.")

        q_analysis = analyse_query(query)
        q_processed = q_analysis["processed_query"]
        q_tokens = q_processed.split()

        scores = self._bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            results.append(_make_result(doc_id, self.documents[doc_id], scores[idx], rank))
        return results

    def search_dense(self, query: str, top_k: int = 10) -> list:
        """
        Dense semantic search using all-MiniLM-L6-v2 embeddings + cosine similarity.
        The model is pretrained — no training is performed here.
        """
        if self._dense_model is None or self._dense_embeddings is None:
            raise RuntimeError("Dense index not built. Call build_dense_index() first.")

        # Encode query with the same pretrained model
        q_analysis = analyse_query(query)
        q_text = q_analysis["expanded_query"]  # use expanded (non-processed) for dense
        q_embedding = self._dense_model.encode(
            [q_text], convert_to_numpy=True, normalize_embeddings=True
        )

        # Cosine similarity (embeddings are L2-normalised → dot product == cosine sim)
        scores = (self._dense_embeddings @ q_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            results.append(_make_result(doc_id, self.documents[doc_id], scores[idx], rank))
        return results

    def search_rrf(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        candidate_pool: int = 100,
    ) -> list:
        """
        Reciprocal Rank Fusion over TF-IDF, BM25, and Dense rankings.

        Formula: RRF(d) = Σ 1 / (k + rank(d))

        Parameters
        ----------
        rrf_k : int
            Smoothing constant (default 60, standard in literature).
        candidate_pool : int
            How many candidates to fetch from each method before fusing.

        Returns list of result dicts with additional RRF breakdown fields.
        """
        # Retrieve large candidate pools from each method
        tfidf_results = self.search_tfidf(query, top_k=candidate_pool)
        bm25_results = self.search_bm25(query, top_k=candidate_pool)
        dense_results = self.search_dense(query, top_k=candidate_pool)

        # Build rank maps {doc_id: rank}
        tfidf_ranks = {r["doc_id"]: r["rank"] for r in tfidf_results}
        bm25_ranks = {r["doc_id"]: r["rank"] for r in bm25_results}
        dense_ranks = {r["doc_id"]: r["rank"] for r in dense_results}

        # Union of all candidate doc IDs
        all_doc_ids = (
            set(tfidf_ranks) | set(bm25_ranks) | set(dense_ranks)
        )

        # Compute RRF scores
        rrf_scores = {}
        for doc_id in all_doc_ids:
            r_tfidf = tfidf_ranks.get(doc_id, NOT_RANKED)
            r_bm25 = bm25_ranks.get(doc_id, NOT_RANKED)
            r_dense = dense_ranks.get(doc_id, NOT_RANKED)
            score = (
                1.0 / (rrf_k + r_tfidf)
                + 1.0 / (rrf_k + r_bm25)
                + 1.0 / (rrf_k + r_dense)
            )
            rrf_scores[doc_id] = {
                "rrf_score": score,
                "tfidf_rank": r_tfidf if r_tfidf != NOT_RANKED else None,
                "bm25_rank": r_bm25 if r_bm25 != NOT_RANKED else None,
                "dense_rank": r_dense if r_dense != NOT_RANKED else None,
            }

        # Sort by RRF score descending
        sorted_doc_ids = sorted(
            rrf_scores, key=lambda d: rrf_scores[d]["rrf_score"], reverse=True
        )[:top_k]

        results = []
        for rank, doc_id in enumerate(sorted_doc_ids, start=1):
            doc = self.documents[doc_id]
            info = rrf_scores[doc_id]
            result = _make_result(doc_id, doc, info["rrf_score"], rank)
            result.update(
                {
                    "tfidf_rank": info["tfidf_rank"],
                    "bm25_rank": info["bm25_rank"],
                    "dense_rank": info["dense_rank"],
                    "rrf_score": info["rrf_score"],
                }
            )
            results.append(result)
        return results

    # ── Latency measurement ──────────────────────────────────────────────────

    def measure_latency(self, query: str, top_k: int = 10, n_runs: int = 5) -> dict:
        """
        Measure average retrieval latency for each method over n_runs.
        Returns {method_name: latency_seconds}
        """
        methods = {
            "TF-IDF": self.search_tfidf,
            "BM25": self.search_bm25,
            "Semantic": self.search_dense,
            "Hybrid RRF": self.search_rrf,
        }
        latencies = {}
        for name, fn in methods.items():
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                fn(query, top_k=top_k)
                times.append(time.perf_counter() - t0)
            latencies[name] = round(sum(times) / len(times), 4)
        return latencies
