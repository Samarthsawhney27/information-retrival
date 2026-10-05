"""
data_loader.py
--------------
Loads the NFCorpus dataset from its original file format.

NFCorpus file formats (verified from README.txt and direct inspection):

Documents  ({split}.docs):
    Tab-separated, one document per line: ID<TAB>TEXT
    IDs are of the form MED-XXXXX or PLAIN-XXXXX

Queries ({split}.{type}.queries):
    Tab-separated, one query per line: ID<TAB>TEXT
    IDs are of the form PLAIN-XXXXX

Qrels ({split}.{2-1-0|3-2-1}.qrel):
    TREC format (space-separated): QUERY_ID 0 DOC_ID RELEVANCE_LEVEL
    3-2-1 qrel: relevance levels 3 (direct), 2 (indirect), 1 (marginal)
    2-1-0 qrel: relevance levels 2 (direct), 1 (indirect)
    Per README, 2-1-0 was used for experiments; 3-2-1 included for reference.

We use:
  - Documents: all three splits combined (train + dev + test) as the document collection
  - Evaluation queries: test.titles.queries (clean, user-like queries)
  - Evaluation qrels: test.3-2-1.qrel (graded, preserves full relevance signal)
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# ── Path configuration ──────────────────────────────────────────────────────
_HERE = Path(__file__).parent
NFCORPUS_DIR = _HERE.parent / "nfcorpus"


# ── Low-level readers ────────────────────────────────────────────────────────

def _read_docs_file(filepath: Path) -> dict:
    """
    Parse a {split}.docs file.
    Format: ID<TAB>TEXT   (one document per line)
    Returns: {doc_id: {"title": "", "text": full_text}}

    NOTE: The docs files contain processed text (lowercased, tokenised, stop-words
    removed).  There is no separate title field in the processed files.  The raw
    title+abstract can be found in nfcorpus/raw/doc_dump.txt but that file is very
    large and many documents there lack clean abstracts.  For simplicity we treat
    the first ~10 tokens of the processed text as a pseudo-title for display.
    """
    docs = {}
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            tab_idx = line.find("\t")
            if tab_idx == -1:
                continue
            doc_id = line[:tab_idx].strip()
            text = line[tab_idx + 1:].strip()
            # Build a pseudo-title from the first ~8 significant words
            tokens = text.split()
            pseudo_title = " ".join(tokens[:8]).title() if tokens else doc_id
            docs[doc_id] = {
                "title": pseudo_title,
                "text": text,
            }
    return docs


def _read_queries_file(filepath: Path) -> dict:
    """
    Parse a {split}.{type}.queries file.
    Format: ID<TAB>TEXT   (one query per line)
    Returns: {query_id: {"query": query_text}}
    """
    queries = {}
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            tab_idx = line.find("\t")
            if tab_idx == -1:
                continue
            query_id = line[:tab_idx].strip()
            query_text = line[tab_idx + 1:].strip()
            queries[query_id] = {"query": query_text}
    return queries


def _read_qrel_file(filepath: Path) -> dict:
    """
    Parse a {split}.{type}.qrel file.
    Format (TREC): QUERY_ID 0 DOC_ID RELEVANCE_LEVEL
    Returns: {query_id: {doc_id: relevance_grade}}
    Relevance grades are kept as integers.
    """
    qrels = defaultdict(dict)
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            query_id, _, doc_id, rel = parts[0], parts[1], parts[2], parts[3]
            qrels[query_id][doc_id] = int(rel)
    return dict(qrels)


# ── Raw doc dump reader (for title extraction) ───────────────────────────────

def _load_raw_titles() -> dict:
    """
    Parse raw/doc_dump.txt to extract doc titles.
    Format: ID<TAB>URL<TAB>TITLE<TAB>ABSTRACT
    Returns: {doc_id: title_string}
    """
    raw_path = NFCORPUS_DIR / "raw" / "doc_dump.txt"
    titles = {}
    if not raw_path.exists():
        return titles
    with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                doc_id = parts[0].strip()
                title = parts[2].strip()
                if title:
                    titles[doc_id] = title
    return titles


# ── Public API ───────────────────────────────────────────────────────────────

def load_documents(splits=("train", "dev", "test"), enrich_titles=True) -> dict:
    """
    Load documents from the specified NFCorpus splits.

    Parameters
    ----------
    splits : tuple of str
        Which splits to load.  By default all three splits are combined so that
        the full document collection is available for retrieval.
    enrich_titles : bool
        If True, attempt to pull real document titles from raw/doc_dump.txt.

    Returns
    -------
    dict
        {doc_id: {"title": str, "text": str}}
    """
    documents = {}
    for split in splits:
        path = NFCORPUS_DIR / f"{split}.docs"
        if not path.exists():
            raise FileNotFoundError(f"NFCorpus docs file not found: {path}")
        documents.update(_read_docs_file(path))

    if enrich_titles:
        raw_titles = _load_raw_titles()
        for doc_id, title in raw_titles.items():
            if doc_id in documents:
                documents[doc_id]["title"] = title

    return documents


def load_queries(split="test", query_type="titles") -> dict:
    """
    Load queries from a specific NFCorpus split and query type.

    Parameters
    ----------
    split : str
        One of "train", "dev", "test".
    query_type : str
        One of "titles", "all", "vid-titles", "vid-desc", "nontopic-titles".
        Defaults to "titles" (clean, user-like title queries — best for evaluation).

    Returns
    -------
    dict
        {query_id: {"query": str}}
    """
    path = NFCORPUS_DIR / f"{split}.{query_type}.queries"
    if not path.exists():
        raise FileNotFoundError(f"NFCorpus queries file not found: {path}")
    return _read_queries_file(path)


def load_qrels(split="test", qrel_type="3-2-1") -> dict:
    """
    Load relevance judgements from a specific NFCorpus split.

    Parameters
    ----------
    split : str
        One of "train", "dev", "test".
    qrel_type : str
        "3-2-1" — graded relevance: 3 (direct), 2 (indirect), 1 (marginal).
        "2-1-0" — collapsed: 2 (direct), 1 (indirect).  Used in original experiments.
        We default to "3-2-1" to preserve the full graded signal for nDCG.

    Returns
    -------
    dict
        {query_id: {doc_id: relevance_grade}}
    """
    path = NFCORPUS_DIR / f"{split}.{qrel_type}.qrel"
    if not path.exists():
        raise FileNotFoundError(f"NFCorpus qrel file not found: {path}")
    return _read_qrel_file(path)


def load_dataset(
    doc_splits=("train", "dev", "test"),
    eval_split="test",
    query_type="titles",
    qrel_type="3-2-1",
) -> tuple:
    """
    Convenience wrapper that loads the full dataset in one call.

    Returns
    -------
    (documents, queries, qrels) — three dicts as described above.
    """
    documents = load_documents(splits=doc_splits)
    queries = load_queries(split=eval_split, query_type=query_type)
    qrels = load_qrels(split=eval_split, qrel_type=qrel_type)
    return documents, queries, qrels


# ── Dataset statistics ───────────────────────────────────────────────────────

def compute_dataset_stats(documents: dict, queries: dict, qrels: dict) -> dict:
    """
    Compute descriptive statistics about the loaded dataset.
    All values are derived from the actual data — nothing is hardcoded.
    """
    # Vocabulary (unique tokens across all documents)
    all_tokens = []
    doc_lengths = []
    for doc in documents.values():
        tokens = doc["text"].split()
        all_tokens.extend(tokens)
        doc_lengths.append(len(tokens))

    vocabulary = set(all_tokens)

    # Query lengths
    query_lengths = [len(q["query"].split()) for q in queries.values()]

    # Relevance judgements
    total_rels = sum(len(v) for v in qrels.values())

    # Grade distribution
    grade_dist = defaultdict(int)
    for doc_grades in qrels.values():
        for grade in doc_grades.values():
            grade_dist[grade] += 1

    return {
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_relevance_judgments": total_rels,
        "vocabulary_size": len(vocabulary),
        "avg_doc_length": round(sum(doc_lengths) / len(doc_lengths), 1) if doc_lengths else 0,
        "avg_query_length": round(sum(query_lengths) / len(query_lengths), 1) if query_lengths else 0,
        "grade_distribution": dict(grade_dist),
    }


if __name__ == "__main__":
    # Quick sanity-check
    print("Loading NFCorpus dataset …")
    docs, queries, qrels = load_dataset()
    stats = compute_dataset_stats(docs, queries, qrels)
    print(f"Documents : {stats['num_documents']:,}")
    print(f"Queries   : {stats['num_queries']:,}")
    print(f"Qrel pairs: {stats['num_relevance_judgments']:,}")
    print(f"Vocabulary: {stats['vocabulary_size']:,}")
    print(f"Avg doc len  : {stats['avg_doc_length']} tokens")
    print(f"Avg query len: {stats['avg_query_length']} tokens")
    print(f"Grade dist: {stats['grade_distribution']}")
    # Sample
    first_doc_id = next(iter(docs))
    print(f"\nSample doc [{first_doc_id}]: {docs[first_doc_id]['title']}")
    first_q_id = next(iter(queries))
    print(f"Sample query [{first_q_id}]: {queries[first_q_id]['query'][:80]}")
