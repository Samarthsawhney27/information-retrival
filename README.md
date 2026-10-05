# 🧬 MedIR — Biomedical Information Retrieval System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?style=for-the-badge)](https://www.sbert.net/)
[![Dataset: NFCorpus](https://img.shields.io/badge/Dataset-NFCorpus-green?style=for-the-badge)](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

MedIR is an advanced academic Information Retrieval (IR) and NLP application that compares, evaluates, and fuses classical and modern dense document retrieval algorithms over the **NFCorpus** biomedical collection.

---

## 📸 Overview & Demo

MedIR features a full-featured, interactive **Streamlit Web Application** designed to evaluate biomedical query processing, semantic document retrieval, and ranking fusion in real time.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 User Biomedical Query                  │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │          Preprocessing & Medical Term Expansion        │
                  │   (Tokenisation · Stopwords · Stemming · Medical Dict) │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
        ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
        │     TF-IDF Model      │ │      BM25 Model       │ │    Dense Transformer  │
        │ (sklearn TfidfVector) │ │ (rank-bm25 BM25Okapi) │ │  (all-MiniLM-L6-v2)   │
        └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │            Reciprocal Rank Fusion (RRF k=60)           │
                  │              RRF(d) = Σ 1 / (k + rank_m(d))            │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │            Streamlit Interface & Metrics               │
                  │   (Precision@K · Recall@K · MAP · nDCG · P-R Curves)   │
                  └────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

* 🔎 **Multi-Model Retrieval Engine**:
  * **TF-IDF**: Sublinear TF-weighted term frequency-inverse document frequency vector space model with cosine similarity.
  * **BM25**: Probabilistic Okapi BM25 implementation ($k_1=1.5, b=0.75$) via `rank_bm25`.
  * **Dense Semantic Search**: Dense embedding retrieval utilizing pretrained `all-MiniLM-L6-v2` sentence transformers.
  * **Hybrid Reciprocal Rank Fusion (RRF)**: Fuses top rankings from TF-IDF, BM25, and Dense semantic models using $RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$ ($k=60$).
* 💊 **Biomedical Query Expansion**: Automatically expands medical abbreviations and clinical terms (e.g., `HTN` $\rightarrow$ `hypertension`, `DM` $\rightarrow$ `diabetes mellitus`).
* 📊 **Comprehensive Evaluation Framework**:
  * **Precision@K** ($K \in \{1, 3, 5, 10\}$)
  * **Recall@K** ($K \in \{1, 3, 5, 10\}$)
  * **Mean Average Precision (MAP)**
  * **Normalized Discounted Cumulative Gain (nDCG@10)** with 3-2-1 graded relevance signals
  * **Interpolated Precision-Recall Curves** (built from real NFCorpus test data)
  * **Retrieval Latency Benchmarking**
  * **Algorithm Overlap Heatmap** (Jaccard similarity between model top-$K$ result sets)
* 🖥 **Interactive Streamlit Web Dashboard**: 3 dedicated workspace tabs (Search Engine, Model Evaluation, Architecture & Dataset details).

---

## 📚 Dataset — NFCorpus

**NFCorpus** (*Boteva et al., ECIR 2016*) is a full-text biomedical information retrieval collection built from **NutritionFacts.org** articles and **PubMed** medical abstracts.

### Data Structure:
```
nfcorpus/
├── {train,dev,test}.docs                 # Tab-separated: DOC_ID <TAB> TEXT
├── {train,dev,test}.*.queries            # Tab-separated: QUERY_ID <TAB> TEXT
└── {train,dev,test}.{3-2-1,2-1-0}.qrel   # TREC format: QUERY_ID 0 DOC_ID GRADE
```

* **Documents**: ~3,600 PubMed abstracts across train/dev/test splits (combined for full collection search).
* **Queries**: User-style biomedical queries (`test.titles.queries`).
* **Qrels**: Graded relevance assessments (3 = Direct relevance, 2 = Indirect relevance, 1 = Marginal relevance).

---

## 🛠 Project Structure

```
.
├── .gitignore
├── README.md                           # Main repository documentation
├── medical-ir/                         # Python application source
│   ├── app.py                          # Streamlit UI Entrypoint (3 Tabs)
│   ├── ir_engine.py                    # MedicalSearchEngine (TF-IDF, BM25, Dense, RRF)
│   ├── data_loader.py                  # NFCorpus dataset parser (Docs, Queries, Qrels)
│   ├── preprocessing.py                # Text normalization, stemming, medical expansion
│   ├── evaluation.py                   # IR Metrics (P@K, R@K, MAP, nDCG@10, P-R Curves)
│   ├── visualization.py                # Plotly chart builders
│   ├── verify.py                       # System metrics verification script
│   ├── requirements.txt                # Python package dependencies
│   ├── README.md                       # Module documentation
│   └── models/embeddings/              # Cached runtime vector indices & embeddings
└── nfcorpus/                           # Full-text biomedical dataset
```

---

## ⚡ Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Samarthsawhney27/information-retrival.git
cd information-retrival
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On macOS/Linux
# .venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r medical-ir/requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run medical-ir/app.py
```
*The web interface will automatically open in your browser at `http://localhost:8501`.*

> **Note on First Run:** On initial startup, the system downloads the `all-MiniLM-L6-v2` transformer model (~90MB) and builds document vector indices. Generated indices are cached in `medical-ir/models/embeddings/` for instantaneous subsequent starts.

---

## 📈 Evaluation Metrics Summary

| Metric | Description | Formula / Threshold |
| :--- | :--- | :--- |
| **Precision@K** | Fraction of top-$K$ retrieved docs that are relevant | $\frac{|\text{Rel} \cap \text{TopK}|}{K}$ |
| **Recall@K** | Fraction of total relevant docs retrieved in top-$K$ | $\frac{|\text{Rel} \cap \text{TopK}|}{|\text{Rel}|}$ |
| **MAP** | Mean of Average Precision across evaluation queries | $\frac{1}{|Q|} \sum_{q \in Q} \text{AP}(q)$ |
| **nDCG@10** | Normalized Discounted Cumulative Gain accounting for graded relevance | $\frac{\text{DCG}_{10}}{\text{IDCG}_{10}}$ |

---

## 📖 Citation

If you use the NFCorpus dataset or reference this work, please cite the original paper:

```bibtex
@inproceedings{boteva16full,
    title={A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
    author={Vera Boteva and Demian Gholipour and Artem Sokolov and Stefan Riezler},
    booktitle={Proceedings of the European Conference on Information Retrieval (ECIR)},
    location={Padova, Italy},
    publisher={Springer},
    year={2016}
}
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
