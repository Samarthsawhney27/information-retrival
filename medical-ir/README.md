# MedIR — Biomedical Information Retrieval System

> A university Information Retrieval / Machine Learning project built on the **NFCorpus** biomedical dataset.

## Project Overview

MedIR is an academic research prototype that compares four document retrieval strategies over the NFCorpus biomedical document collection:

| Method | Description |
|---|---|
| **TF-IDF** | Classical vector space model with cosine similarity (scikit-learn) |
| **BM25** | Probabilistic ranking with term frequency saturation (rank-bm25) |
| **Semantic Search** | Dense retrieval with `all-MiniLM-L6-v2` sentence embeddings |
| **Hybrid RRF** | Reciprocal Rank Fusion of all three methods |

---

## Dataset — NFCorpus

NFCorpus (Boteva et al., ECIR 2016) is a full-text biomedical information retrieval dataset derived from NutritionFacts.org and PubMed abstracts.

**The project uses the original NFCorpus file format directly:**

```
nfcorpus/
├── {train,dev,test}.docs         # Tab-separated: DOC_ID<TAB>TEXT
├── {train,dev,test}.*.queries    # Tab-separated: QUERY_ID<TAB>TEXT
└── {train,dev,test}.{3-2-1,2-1-0}.qrel  # TREC format: QUERY_ID 0 DOC_ID GRADE
```

- **Documents:** ~3,600 PubMed abstracts per split (train/dev/test combined for retrieval)
- **Queries:** Titles queries (clean, user-like) used for evaluation
- **Qrels:** 3-2-1 graded relevance (3=direct, 2=indirect, 1=marginal)

---

## Features

- 🔎 **TF-IDF retrieval** — sublinear TF, bigrams, cosine similarity
- 🔤 **BM25 retrieval** — BM25Okapi with configurable k1/b parameters
- 🧠 **Semantic retrieval** — pretrained sentence transformer (no training)
- 💊 **Medical query expansion** — abbreviation/term dictionary
- 🔀 **RRF hybrid retrieval** — fuses all three rankings (k=60)
- 📊 **Precision@K** (K=1,3,5,10)
- 📊 **Recall@K** (K=1,3,5,10)
- 📊 **MAP** (Mean Average Precision)
- 📊 **nDCG@10** (graded relevance)
- 📉 **Precision-Recall curves** (interpolated, real data)
- ⏱ **Retrieval latency** comparison
- 🗺 **Retrieval overlap** heatmap (Jaccard similarity)
- 🖥 **Streamlit interface** — professional academic prototype

---

## System Architecture

```
User Query
     │
     ▼
Text Preprocessing
(lowercase · tokenise · stopword removal · stemming)
     │
     ▼
Medical Term Expansion
(HTN → hypertension, DM → diabetes mellitus, …)
     │
     ├─────────────────┬──────────────────┐
     ▼                 ▼                  ▼
 TF-IDF            BM25          Sentence Transformer
(sklearn)      (rank-bm25)      (all-MiniLM-L6-v2)
cosine sim       BM25Okapi       cosine similarity
     │                 │                  │
     └─────────────────┴──────────────────┘
                       │
                       ▼
          Reciprocal Rank Fusion (k=60)
              RRF(d) = Σ 1/(k + rankᵢ(d))
                       │
                       ▼
               Ranked Results
```

---

## Project Structure

```
medical-ir/
├── app.py             # Streamlit UI — 3 tabs: Search, Evaluation, About
├── data_loader.py     # NFCorpus parser (docs, queries, qrels)
├── preprocessing.py   # Text cleaning, stemming, medical term expansion
├── ir_engine.py       # MedicalSearchEngine (TF-IDF, BM25, Dense, RRF)
├── evaluation.py      # P@K, R@K, MAP, nDCG, P-R curve computation
├── visualization.py   # Plotly chart builders
├── requirements.txt
├── README.md
├── nfcorpus/          # Original dataset — DO NOT MODIFY
└── models/
    └── embeddings/    # Cached TF-IDF index, BM25 index, dense embeddings
```

---

## Installation

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The sentence-transformer model (`all-MiniLM-L6-v2`) is downloaded automatically on first run (~90 MB). Document embeddings are cached to `models/embeddings/` after the first run.

---

## Running the Application

```bash
cd medical-ir
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

**First run** will:
1. Load and preprocess all NFCorpus documents
2. Build the TF-IDF index
3. Build the BM25 index
4. Download and cache sentence transformer embeddings

Subsequent runs load everything from cache — startup is fast.

---

## Deployment (Streamlit Community Cloud)

1. Push the project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository.
4. Set the main file path to `medical-ir/app.py`.
5. Add a `packages.txt` if any system packages are needed.

> **Note:** The NFCorpus dataset must be included in the repository (it is small enough at ~30 MB).

---

## Evaluation Methodology

| Metric | Threshold / Formula |
|---|---|
| Precision@K | ≥1 relevance grade as "relevant"; P@K = \|Rel ∩ TopK\| / K |
| Recall@K | R@K = \|Rel ∩ TopK\| / \|Rel\| |
| MAP | Mean over queries of Average Precision |
| nDCG@10 | Full graded relevance; DCG / IDCG |

All metrics are computed from actual NFCorpus test queries and qrels — no values are hardcoded.

---

## Citation

```bibtex
@inproceedings{boteva16full,
    title={A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
    author={Vera Boteva and Demian Gholipour and Artem Sokolov and Stefan Riezler},
    booktitle={Proceedings of the European Conference on Information Retrieval (ECIR)},
    location={Padova, Italy},
    publisher={Springer},
    year={2016},
}
```
