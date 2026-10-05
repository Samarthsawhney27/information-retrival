"""
preprocessing.py
----------------
Text preprocessing for the MedIR Information Retrieval system.

Provides:
  - tokenisation
  - lowercasing
  - punctuation normalisation
  - stopword removal (uses NFCorpus's own stopwords.large list)
  - Porter stemming
  - medical term / abbreviation expansion
  - query analysis helper

Original document text is NEVER modified — preprocessing operates on a copy.
"""

import re
import string
from pathlib import Path
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_NFCORPUS_DIR = _HERE.parent / "nfcorpus"
_STOPWORDS_FILE = _NFCORPUS_DIR / "raw" / "stopwords.large"

# ── Load NFCorpus stopwords ──────────────────────────────────────────────────

def _load_nfcorpus_stopwords() -> set:
    """Load the NFCorpus stopwords.large list (preferred over generic lists)."""
    stopwords = set()
    path = _STOPWORDS_FILE
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    # Fallback: NLTK English stopwords
    if not stopwords:
        from nltk.corpus import stopwords as nltk_sw
        stopwords = set(nltk_sw.words("english"))
    return stopwords


STOPWORDS: set = _load_nfcorpus_stopwords()
_stemmer = PorterStemmer()

# ── Medical term expansion dictionary ────────────────────────────────────────
# Maps abbreviations/common lay-terms → medical equivalents.
# Used to enrich user queries before retrieval.
MEDICAL_TERMS: dict = {
    # Abbreviations
    "htn":          "hypertension",
    "bp":           "blood pressure",
    "dm":           "diabetes mellitus",
    "dm2":          "type 2 diabetes mellitus",
    "t2dm":         "type 2 diabetes mellitus",
    "mi":           "myocardial infarction",
    "cvd":          "cardiovascular disease",
    "chd":          "coronary heart disease",
    "af":           "atrial fibrillation",
    "copd":         "chronic obstructive pulmonary disease",
    "ckd":          "chronic kidney disease",
    "ibs":          "irritable bowel syndrome",
    "ibd":          "inflammatory bowel disease",
    "bmi":          "body mass index",
    "ldl":          "low-density lipoprotein cholesterol",
    "hdl":          "high-density lipoprotein cholesterol",
    "vit":          "vitamin",
    "vit d":        "vitamin d",
    "vit c":        "vitamin c",
    "vit b12":      "vitamin b12",
    "dha":          "docosahexaenoic acid omega-3",
    "epa":          "eicosapentaenoic acid omega-3",
    "gi":           "gastrointestinal",
    "cad":          "coronary artery disease",
    "hr":           "heart rate",
    "rct":          "randomized controlled trial",
    # Lay terms
    "heart attack":         "myocardial infarction",
    "high blood pressure":  "hypertension",
    "low blood pressure":   "hypotension",
    "sugar":                "glucose",
    "bad cholesterol":      "ldl low-density lipoprotein",
    "good cholesterol":     "hdl high-density lipoprotein",
    "gut health":           "gastrointestinal microbiome",
    "fish oil":             "omega-3 fatty acid",
    "vitamin d":            "cholecalciferol vitamin d deficiency",
    "vitamin c":            "ascorbic acid",
    "vitamin b12":          "cobalamin",
    "blood sugar":          "blood glucose glycemia",
    "weight loss":          "obesity body weight reduction",
    "cancer":               "malignancy neoplasm",
    "stroke":               "cerebrovascular accident",
    "alzheimer":            "alzheimer disease dementia",
    "alzheimers":           "alzheimer disease dementia",
}


# ── Core preprocessing ───────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    # Replace hyphens with space to separate compound words
    text = re.sub(r"[-/]", " ", text)
    # Remove remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list:
    """Simple whitespace tokeniser (corpus text is already tokenised)."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Remove tokens that appear in the NFCorpus stopword list."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stem_tokens(tokens: list) -> list:
    """Apply Porter stemming to each token."""
    return [_stemmer.stem(t) for t in tokens]


def preprocess_text(text: str, stem: bool = True) -> str:
    """
    Full preprocessing pipeline for a document or query.

    Parameters
    ----------
    text : str
        Raw text.
    stem : bool
        Whether to apply Porter stemming.

    Returns
    -------
    str
        Preprocessed text (space-joined tokens).
    """
    text = normalise_text(text)
    tokens = tokenise(text)
    tokens = remove_stopwords(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


def preprocess_documents(documents: dict, stem: bool = True) -> dict:
    """
    Preprocess all document texts.  The original 'text' field is preserved.

    Returns a new dict: {doc_id: {"title": ..., "text": original, "processed": preprocessed}}
    """
    result = {}
    for doc_id, doc in documents.items():
        processed = preprocess_text(doc["text"], stem=stem)
        result[doc_id] = {
            "title": doc["title"],
            "text": doc["text"],
            "processed": processed,
        }
    return result


# ── Medical term expansion ───────────────────────────────────────────────────

def expand_query(query: str) -> dict:
    """
    Detect and expand medical abbreviations/terms in a user query.

    Returns
    -------
    dict with keys:
        original_query   : str  — the raw query as typed
        detected_terms   : list of (matched_term, expansion) tuples
        expanded_query   : str  — query with expansions appended
    """
    query_lower = query.lower()
    detected = []
    extra_terms = []

    # Check multi-word phrases first (longest match wins)
    sorted_terms = sorted(MEDICAL_TERMS.keys(), key=len, reverse=True)
    for term in sorted_terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, query_lower):
            expansion = MEDICAL_TERMS[term]
            detected.append((term, expansion))
            # Only add expansion text not already in the query
            for word in expansion.split():
                if word not in query_lower:
                    extra_terms.append(word)

    # Remove duplicates while preserving order
    seen = set()
    unique_extra = []
    for t in extra_terms:
        if t not in seen:
            seen.add(t)
            unique_extra.append(t)

    expanded = query
    if unique_extra:
        expanded = query + " " + " ".join(unique_extra)

    return {
        "original_query": query,
        "detected_terms": detected,
        "expanded_query": expanded,
    }


def analyse_query(query: str) -> dict:
    """
    Full query analysis: expand → preprocess.

    Returns
    -------
    dict with keys:
        original_query, detected_terms, expanded_query, processed_query
    """
    expansion = expand_query(query)
    processed = preprocess_text(expansion["expanded_query"], stem=True)
    return {
        **expansion,
        "processed_query": processed,
    }


if __name__ == "__main__":
    test_queries = [
        "What are the effects of vitamin D deficiency?",
        "How does dietary fiber affect DM2?",
        "Heart attack prevention with omega-3",
        "HTN and magnesium relationship",
    ]
    for q in test_queries:
        result = analyse_query(q)
        print("Original :", result["original_query"])
        print("Detected :", result["detected_terms"])
        print("Expanded :", result["expanded_query"])
        print("Processed:", result["processed_query"])
        print()
