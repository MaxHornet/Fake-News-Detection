# HXDistil-XGB

### Hybrid Transformer–Gradient-Boosting Fake News Detection with Retrieval-Augmented Evidence Verification

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-98.31%25-brightgreen?style=for-the-badge" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/F1--Score-98.36%25-brightgreen?style=for-the-badge" alt="F1 Score"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-HuggingFace-orange?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow?style=for-the-badge" alt="Status"/>
</p>

---

**HXDistil-XGB** is an explainable fake-news detection and evidence-verification system that combines:

| Component | Technology |
|---|---|
| Contextual Embeddings | `DistilBERT` (frozen) |
| Lexical Features | TF-IDF (3,000 features) |
| Stylistic Features | 7 handcrafted metadata features |
| Classifier | XGBoost |
| Claim Extraction | Sentence segmentation + NER |
| Evidence Retrieval | Google News RSS |
| Semantic Re-ranking | `all-MiniLM-L6-v2` |
| NLI Verification | `DeBERTa-v3-base-mnli-fever-anli` |
| Explainability | SHAP (`TreeExplainer`) |

> The objective is to move beyond a simple **Real / Fake** prediction by providing an evidence-grounded explanation of *why* an article is classified in a particular way.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Contributions](#2-key-contributions)
3. [Classification Performance](#3-classification-performance)
4. [Baseline Comparison](#4-baseline-comparison)
5. [Evidence Verification System](#5-evidence-verification-system)
6. [Retrieval Engine](#6-retrieval-engine)
7. [Source Credibility](#7-source-credibility)
8. [NLI Verification](#8-nli-verification)
9. [Adaptive Fusion](#9-adaptive-fusion)
10. [Explainability](#10-explainability)
11. [Dataset](#11-dataset)
12. [Dataset Leakage Investigation](#12-dataset-leakage-investigation)
13. [Retrieval Evaluation](#13-retrieval-evaluation)
14. [Repository Structure](#14-repository-structure)
15. [Main Files](#15-main-files)
16. [Installation](#16-installation)
17. [Model Files](#17-model-files)
18. [Running the System](#18-running-the-system)
19. [Example Output](#19-example-output)
20. [Important Design Principle](#20-important-design-principle)
21. [Limitations](#21-limitations)
22. [Research Reproducibility](#22-research-reproducibility)
23. [Computational Characteristics](#23-computational-characteristics)
24. [Future Work](#24-future-work)
25. [Research Paper](#25-research-paper)
26. [Disclaimer](#26-disclaimer)
27. [Citation](#27-citation)
28. [Project Status](#28-project-status)
29. [Author](#29-author)

---

## 1. Overview

Traditional fake-news classifiers generally produce a binary prediction:

```
Article → Real / Fake
```

**HXDistil-XGB** extends this into a multi-stage verification pipeline:

```
                    ┌─────────────────────┐
                    │    News Article     │
                    │ URL / Text / PDF    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Article Scraper   │
                    │   & Preprocessing   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌─────────────────────┐          ┌─────────────────────┐
    │ HXDistil-XGB Branch │          │   Evidence Branch   │
    └──────────┬──────────┘          └──────────┬──────────┘
               │                                │
       ┌───────┼────────┐              ┌────────▼─────────┐
       │       │        │              │ Claim Extraction │
       ▼       ▼        ▼              └────────┬─────────┘
    DistilBERT TF-IDF Metadata                  │
       │       │        │                       ▼
       └───────┼────────┘              ┌──────────────────┐
               ▼                       │ Google News RSS  │
       Feature Fusion                  └────────┬─────────┘
               │                                │
               ▼                                ▼
          XGBoost                      URL Decoding
               │                                │
               │                                ▼
               │                       Semantic Ranking
               │                                │
               │                                ▼
               │                     Article / Paragraph
               │                       Evidence Ranking
               │                                │
               │                                ▼
               │                       DeBERTa NLI
               │                                │
               └──────────────┬─────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Adaptive Evidence  │
                    │ Fusion & Scoring   │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Explainable Result │
                    │ Dashboard / JSON   │
                    └────────────────────┘
```

The classification and evidence-verification branches are designed to provide **complementary signals** rather than treating retrieval as a replacement for the content classifier.

---

## 2. Key Contributions

The project contains two major components.

### 2.1 HXDistil-XGB Classification Model

The classifier combines three feature families:

#### 🔵 Semantic Features
A frozen `distilbert-base-uncased` model produces contextual embeddings of **768 dimensions**.

#### 🟡 Lexical Features
TF-IDF features are extracted from the article text — **3,000 features** in the current implementation.

#### 🟢 Metadata / Stylistic Features
Seven lightweight features are extracted:

| Feature | Description |
|---|---|
| Text length | Total character/word count |
| Uppercase ratio | Ratio of uppercase characters |
| Exclamation marks | Count of `!` |
| Question marks | Count of `?` |
| URL count | Number of hyperlinks |
| VADER sentiment | Compound sentiment score |
| Flesch reading ease | Readability score |

These are concatenated into a single feature vector:

```
768 (DistilBERT) + 3,000 (TF-IDF) + 7 (Metadata) = 3,775 features
```

The resulting vector is classified using **XGBoost**.

---

## 3. Classification Performance

On the held-out WELFake test split (14,427 articles), HXDistil-XGB achieved:

| Metric | Score |
|---|---|
| **Accuracy** | **98.31%** |
| **Precision** | **97.83%** |
| **Recall** | **98.91%** |
| **F1-score** | **98.36%** |

> Results are reported in the accompanying research paper.

---

## 4. Baseline Comparison

HXDistil-XGB was evaluated against several feature/model configurations under identical experimental settings:

| Model | Accuracy |
|---|---|
| Logistic Regression | 91.90% |
| DistilBERT + XGBoost | 93.83% |
| DistilBERT + Metadata | 94.91% |
| TF-IDF + XGBoost | 96.58% |
| **HXDistil-XGB (ours)** | **98.31%** ✅ |

The hybrid model provides the strongest performance among all evaluated configurations.

---

## 5. Evidence Verification System

Classification alone cannot determine whether every factual statement in an article is correct. Therefore, HXDistil-XGB includes a **claim-level evidence verification pipeline**:

```
Article
   ↓
Sentence Segmentation
   ↓
Claim Detection
   ↓
Top-K Factual Claims
   ↓
Evidence Retrieval
   ↓
Semantic Re-ranking
   ↓
NLI Verification
   ↓
Evidence Fusion
```

The system extracts factual claims using:

- Sentence segmentation
- Named-entity information
- Fact-oriented verbs
- Numerical information
- Claim scoring
- Boilerplate filtering
- Duplicate removal

> The current prototype limits the number of claims processed per article.

---

## 6. Retrieval Engine

The original retrieval prototype used **NewsAPI**, which produced several problems:

- Poor search relevance
- Unrelated articles
- Weak evidence retrieval
- Limited coverage

The retrieval stage was redesigned around **Google News RSS**. The current pipeline:

```
Claim
  ↓
Query Generation
  ↓
Google News RSS
  ↓
Google URL Decoding
  ↓
Candidate Articles
  ↓
Semantic Article Ranking
  ↓
Article Downloading
  ↓
Paragraph Extraction
  ↓
Paragraph Semantic Ranking
  ↓
Evidence Document
```

The retrieval engine uses the following libraries:

| Library | Role |
|---|---|
| `feedparser` | Google News RSS parsing |
| `googlenewsdecoder` | Google URL decoding |
| `sentence-transformers` (`all-MiniLM-L6-v2`) | Semantic ranking |
| `newspaper3k` | Article downloading |
| `trafilatura` | Fallback content extraction |

Implementation: [`retrieval_engine.py`](retrieval_engine.py)

---

## 7. Source Credibility

The retrieval engine assigns configurable credibility weights to recognized publishers:

| Publisher | Credibility Weight |
|---|---|
| Reuters | 1.00 |
| Associated Press | 1.00 |
| BBC | 0.98 |
| Bloomberg | 0.96 |
| CNBC | 0.95 |
| TechCrunch | 0.92 |
| Business Insider | 0.90 |
| The Verge | 0.88 |
| Wired | 0.88 |
| Forbes | 0.84 |
| Fortune | 0.84 |

These weights are used as **one component** of the evidence reliability calculation rather than as an independent truth label.

---

## 8. NLI Verification

Retrieved evidence is passed to a pretrained DeBERTa NLI model:

```
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

The model evaluates the relationship between a **Claim ↔ Retrieved Evidence** pair and produces one of three labels:

| Label | Meaning |
|---|---|
| ✅ Entailment | Evidence supports the claim |
| ➖ Neutral | Evidence is unrelated or inconclusive |
| ❌ Contradiction | Evidence contradicts the claim |

The system aggregates these claim-level signals across all retrieved evidence.

---

## 9. Adaptive Fusion

The final system does not rely exclusively on the HXDistil-XGB classifier. Instead, it combines:

```
HXDistil-XGB Confidence
        +
  Claim Support
        +
Contradiction Signals
        +
 Evidence Reliability
        +
  Source Credibility
        +
 Evidence Coverage
        +
  Source Diversity
        ↓
  Adaptive Fusion
        ↓
  Final Verdict
```

> The exact fusion weights are treated as implementation-level design choices rather than learned parameters in the current version.

---

## 10. Explainability

HXDistil-XGB uses **SHAP** to explain the contribution of the fused feature space via `TreeExplainer`. SHAP analysis can inspect the contribution of:

- `DistilBERT` features
- TF-IDF features
- Metadata features

The final system additionally exposes:

| Output Field | Description |
|---|---|
| Final Verdict | Evidence-aware classification |
| HXDistil-XGB Prediction | Raw classifier output |
| Classifier Confidence | Prediction probability |
| Extracted Claims | Top-K factual claims |
| Claim-Level NLI Results | Entailment / Neutral / Contradiction per claim |
| Evidence Sources | Retrieved article URLs |
| Evidence Quality | Aggregated evidence score |
| Source Reliability | Credibility-weighted score |
| Fusion Score | Combined evidence-aware score |

---

## 11. Dataset

The project investigated multiple fake-news datasets:

| Dataset | Notes |
|---|---|
| **WELFake** | Primary benchmark — 72,134 labeled records |
| **LIAR** | Used alongside WELFake for training |
| **ISOT** | Evaluated; substantial overlap with WELFake discovered |

The principal classification evaluation uses a held-out WELFake test split with an **80 / 20 stratified train/test split**.

---

## 12. Dataset Leakage Investigation

An important contribution of this project was discovering **substantial overlap** between WELFake and ISOT:

| Statistic | Count |
|---|---|
| ISOT total rows | 44,898 |
| ISOT unique articles | 39,105 |
| WELFake total rows | 72,134 |
| WELFake unique articles | 63,678 |
| Articles shared by both datasets | **39,105** |
| Unique ISOT articles absent from WELFake | **0** |

> ⚠️ The initially observed near-perfect ISOT performance was identified as **dataset leakage** rather than evidence of genuine cross-dataset generalisation. This finding is explicitly documented rather than presenting the leaked evaluation as a valid independent benchmark.

---

## 13. Retrieval Evaluation

The project includes a dedicated retrieval evaluation procedure, independent of the classification pipeline:

| Metric | Description |
|---|---|
| Retrieval Success Rate | % of claims with ≥1 retrieved article |
| Average RSS Results | Mean articles from RSS feed |
| Average Decoded Results | Mean successfully decoded URLs |
| Average Downloaded Articles | Mean articles successfully scraped |
| Average Evidence Items | Mean evidence paragraphs per claim |
| Pseudo-MRR | Approximate Mean Reciprocal Rank |
| Average Semantic Similarity | Mean cosine similarity of top evidence |
| Trusted Source Ratio | Fraction from high-credibility publishers |

This distinction is critical:

```
Good Classifier  ≠  Good Evidence Retrieval
Good Retrieval   ≠  Correct Final Verdict
```

The project evaluates all components **separately**.

---

## 14. Repository Structure

```
HXDistil-XGB/
│
├── README.md
│
├── fakenews_prototype6.ipynb          ← Main end-to-end prototype
│
├── Prototype2_Train_on_WELFAKE+LIAR.ipynb
├── Prototype2_Evaluation_on_ISOT.ipynb
│
├── retrieval_engine.py                ← Reusable evidence retrieval module
├── webscrape.py                       ← Reusable article-processing module
│
├── models/
│   ├── HXDistil_XGB_v2.joblib
│   └── tfidf.joblib
│
├── evaluation/
│   └── retrieval_evaluation.ipynb
│
├── results/
│   ├── fake_news_results.csv
│   └── fake_news_results.json
│
├── figures/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   └── shap_analysis.png
│
└── docs/
    ├── progress_report.pdf
    └── supplementary_technical_report.md
```

> Large datasets and model binaries should preferably be hosted separately rather than committed directly to GitHub.

---

## 15. Main Files

### `fakenews_prototype6.ipynb`
Main end-to-end prototype. Integrates:
- Article scraping & preprocessing
- HXDistil-XGB inference
- Claim extraction
- Google News retrieval
- Evidence verification
- Adaptive fusion
- Result generation

The notebook imports the reusable scraper and retrieval modules rather than duplicating the complete retrieval implementation.

---

### `retrieval_engine.py`
Reusable evidence retrieval module. Responsibilities:

```
Google News RSS  →  URL Decoding  →  Candidate Ranking
  →  Article Downloading  →  Paragraph Extraction  →  Evidence Ranking
```

---

### `webscrape.py`
Reusable article-processing module. Supports:
- URL article extraction
- Pasted article text
- PDF extraction
- Title extraction
- Article categorisation
- Summary generation

---

## 16. Installation

The project was developed primarily in **Google Colab**. Install the required Python packages:

```bash
pip install newspaper3k beautifulsoup4 pandas nltk spacy gensim scikit-learn lxml_html_clean
pip install transformers sentence-transformers xgboost joblib shap textstat vaderSentiment feedparser
pip install googlenewsdecoder trafilatura sentencepiece pypdf
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

## 17. Model Files

The inference pipeline expects the following trained artifacts:

| File | Description |
|---|---|
| `HXDistil_XGB_v2.joblib` | Trained XGBoost classifier |
| `tfidf.joblib` | Fitted TF-IDF vectorizer |

The notebook loads these artifacts at inference time rather than retraining the model for every run.

---

## 18. Running the System

### Step 1 — Prepare the environment
Install the dependencies and spaCy model (see [Installation](#16-installation)).

### Step 2 — Place model files
Place the following in the configured model directory:
```
HXDistil_XGB_v2.joblib
tfidf.joblib
```

### Step 3 — Place modules beside the notebook
```
webscrape.py
retrieval_engine.py
```
Both must be accessible from the notebook working directory.

### Step 4 — Run the main notebook
Open `fakenews_prototype6.ipynb` and execute the cells sequentially.

### Step 5 — Provide an article
The system accepts three input types:

| Input Type | Description |
|---|---|
| 🔗 URL | Direct article link |
| 📄 Pasted text | Raw article body |
| 📑 PDF | Uploaded document |

### Step 6 — Inspect the result
The system produces:

| Output | Description |
|---|---|
| HXDistil-XGB Prediction | `REAL` / `FAKE` |
| Confidence | Classifier probability |
| Extracted Claims | Top-K factual statements |
| Evidence Sources | Retrieved article URLs |
| Claim Verification | NLI labels per claim |
| Evidence Quality | Aggregated evidence score |
| Source Reliability | Credibility-weighted score |
| Fusion Score | Combined evidence-aware score |
| **Final Verdict** | Evidence-grounded conclusion |
| Explanation | SHAP-based feature attribution |

---

## 19. Example Output

```
════════════════════════════════════════════════
           HXDistil-XGB Result Dashboard
════════════════════════════════════════════════

  HXDistil Prediction : REAL
  HXDistil Confidence : 79.41%

────────────────────────────────────────────────
  Evidence Signals
────────────────────────────────────────────────
  Claim Support Score  : 71.01
  Evidence Quality     : 53.76
  Evidence Reliability : 56.20
  Trusted Source Ratio : 25.00
  Source Diversity     : 66.67
  Fusion Confidence    : 66.64%

════════════════════════════════════════════════
  ✅  Final Verdict    : VERIFIED
════════════════════════════════════════════════
```

The dashboard exposes both the original classifier result and the evidence-based verification signals instead of hiding the intermediate reasoning.

---

## 20. Important Design Principle

The system deliberately separates **Classification** from **Fact Verification**:

| Branch | Question Answered |
|---|---|
| 🤖 HXDistil-XGB | *"Does this article resemble real/fake patterns learned from training data?"* |
| 🔍 Evidence Branch | *"Are the important factual claims supported or contradicted by independently retrieved evidence?"* |

The final system **combines** these signals. This is important because a classifier can incorrectly label a genuine article, while strong external evidence can provide information unavailable to a static training corpus.

---

## 21. Limitations

### Dataset Limitations
The principal benchmark is based on WELFake. Substantial overlap with ISOT means the leaked ISOT result is not treated as independent generalisation evidence.

### Retrieval Limitations
- Google News RSS provides **candidate discovery**, not guaranteed factual truth.
- Publisher pages may block automated downloading, requiring fallback extraction methods.

### NLI Limitations
NLI predictions depend on:
- Quality of retrieved evidence
- Wording of the claim
- Evidence context
- Pretrained NLI model limitations

### Fusion Limitations
The current fusion weights are **manually designed** implementation parameters rather than learned from a dedicated end-to-end verification dataset.

### Real-World Misinformation
> ⚠️ The system should not be treated as an autonomous authority on truth. A final evidence-grounded verdict should always be interpreted together with the retrieved sources and claim-level evidence.

---

## 22. Research Reproducibility

The project keeps all major stages modular:

```
Training
    ↓
Saved Model
    ↓
Article Scraping
    ↓
Feature Extraction
    ↓
HXDistil-XGB
    ↓
Claim Extraction
    ↓
Evidence Retrieval
    ↓
NLI Verification
    ↓
Adaptive Fusion
    ↓
Explainable Result
```

The supplementary technical material documents the experiments, implementation decisions and architecture details derived from the project notebooks.

---

## 23. Computational Characteristics

The classification branch is designed to remain **relatively lightweight** compared with a fully fine-tuned transformer classifier.

| Branch | Components | Cost |
|---|---|---|
| Classification | DistilBERT embedding, TF-IDF transform, XGBoost inference | 🟢 Lightweight |
| Evidence | Network retrieval, article downloading, MiniLM ranking, paragraph processing, DeBERTa NLI | 🔴 Expensive |

> Evidence verification can be treated as a **separate or selective stage** when computational budget is limited.

---

## 24. Future Work

Planned research directions:

- [ ] Larger and more diverse datasets
- [ ] Independently collected test datasets
- [ ] Improved claim extraction
- [ ] Stronger retrieval models
- [ ] Learned fusion weights
- [ ] Retrieval calibration
- [ ] Additional source-quality modelling
- [ ] Statistical significance testing
- [ ] Latency benchmarking
- [ ] Larger-scale real-world evaluation
- [ ] Multilingual fake-news verification
- [ ] Improved evidence coverage
- [ ] Human fact-checker comparison

---

## 25. Research Paper

The project is accompanied by a research manuscript:

> **HXDistil-XGB: Hybrid Feature-Fusion Fake News Detection with Evidence Verification**

The manuscript describes the proposed architecture, experimental evaluation, ablation studies, explainability analysis and retrieval-augmented verification framework.

**Reported WELFake Results:**

| Metric | Score |
|---|---|
| Accuracy | 98.31% |
| Precision | 97.83% |
| Recall | 98.91% |
| F1-score | 98.36% |

---

## 26. Disclaimer

> ⚠️ This project is a **research prototype** and should not be considered a replacement for professional fact-checkers or authoritative sources.

- A prediction of `FAKE` does **not** by itself prove that an article is false.
- A prediction of `REAL` does **not** guarantee that every factual claim in an article is correct.

The evidence-verification layer is intended to provide **additional context** and source-attributed evidence to support human interpretation.

---

## 27. Citation

If you use this project in academic work, please cite the associated research paper:

```bibtex
@article{hxdistilxgb,
  title   = {HXDistil-XGB: Hybrid Feature-Fusion Fake News Detection
             with Evidence Verification},
  author  = {Dayyan Waseem},
  journal = {Applied Soft Computing},
  year    = {2026}
}
```

---

## 28. Project Status

| Feature | Status |
|---|---|
| Dataset preprocessing | ✅ Complete |
| Dataset overlap investigation | ✅ Complete |
| DistilBERT feature extraction | ✅ Complete |
| TF-IDF feature extraction | ✅ Complete |
| Metadata feature extraction | ✅ Complete |
| HXDistil-XGB training | ✅ Complete |
| Baseline comparison | ✅ Complete |
| SHAP explainability | ✅ Complete |
| Article scraping | ✅ Complete |
| Claim extraction | ✅ Complete |
| Google News RSS retrieval | ✅ Complete |
| Google URL decoding | ✅ Complete |
| Semantic evidence ranking | ✅ Complete |
| Paragraph-level evidence ranking | ✅ Complete |
| DeBERTa NLI verification | ✅ Complete |
| Source credibility scoring | ✅ Complete |
| Evidence fusion | ✅ Complete |
| Explainable dashboard | ✅ Complete |
| Retrieval evaluation | ✅ Complete |
| JSON/CSV result export | ✅ Complete |
| Research documentation | ✅ Complete |

---

## 29. Author

Developed as a research project on **Explainable Fake News Detection and Evidence-Based Fact Verification**.

**Primary Technologies:**

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/DistilBERT-0A66C2?style=flat-square"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-8A2BE2?style=flat-square"/>
</p>

| Technology | Role |
|---|---|
| Python | Primary language |
| PyTorch | Deep learning backend |
| Hugging Face Transformers | `DistilBERT`, `DeBERTa` models |
| Sentence-Transformers | Semantic ranking (`all-MiniLM-L6-v2`) |
| XGBoost | Classification |
| TF-IDF (scikit-learn) | Lexical features |
| spaCy | NLP / claim extraction |
| SHAP | Explainability |
| Google News RSS | Evidence retrieval |
| `newspaper3k` / `trafilatura` | Article scraping |

---

<p align="center">
  <sub>HXDistil-XGB — Research Prototype · Dayyan Waseem · 2026</sub>
</p>
