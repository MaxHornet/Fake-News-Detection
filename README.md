# HXDistil-XGB
## Hybrid Transformer–Gradient-Boosting Fake News Detection with Retrieval-Augmented Evidence Verification

HXDistil-XGB is an explainable fake-news detection and evidence-verification system that combines:

- DistilBERT contextual semantic embeddings
- TF-IDF lexical features
- Handcrafted linguistic and metadata features
- XGBoost classification
- Automatic factual claim extraction
- Google News RSS-based evidence retrieval
- Semantic evidence re-ranking
- DeBERTa-based Natural Language Inference (NLI)
- Source credibility weighting
- Adaptive evidence fusion
- SHAP-based explainability

The objective is to move beyond a simple **Real/Fake** prediction by providing an evidence-grounded explanation of why an article is classified in a particular way.

---

# 1. Overview

Traditional fake-news classifiers generally produce a binary prediction:

```text
Article → Real / Fake

HXDistil-XGB extends this into a multi-stage verification pipeline:

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
    │ HXDistil-XGB Branch │          │ Evidence Branch     │
    └──────────┬──────────┘          └──────────┬──────────┘
               │                                │
       ┌───────┼────────┐              ┌────────▼─────────┐
       │       │        │              │ Claim Extraction │
       ▼       ▼        ▼              └────────┬─────────┘
    DistilBERT TF-IDF Metadata                    │
       │       │        │                        ▼
       └───────┼────────┘              ┌──────────────────┐
               ▼                       │ Google News RSS  │
       Feature Fusion                  └────────┬─────────┘
               │                                │
               ▼                                ▼
          XGBoost                     URL Decoding
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

The classification and evidence-verification branches are designed to provide complementary signals rather than treating retrieval as a replacement for the content classifier.

2. Key Contributions

The project contains two major components.

2.1 HXDistil-XGB Classification Model

The classifier combines three feature families:

Semantic Features

A frozen:

distilbert-base-uncased

model is used to obtain contextual embeddings.

Dimension:

768
Lexical Features

TF-IDF features are extracted from the article text.

Current implementation:

3000 TF-IDF features
Metadata / Stylistic Features

Seven lightweight features are extracted:

Text length
Uppercase ratio
Exclamation mark count
Question mark count
URL count
VADER sentiment compound score
Flesch reading ease

These are concatenated into a single representation:

768 + 3000 + 7 = 3775 features

The resulting vector is classified using XGBoost.

3. Classification Performance

On the held-out WELFake test split, HXDistil-XGB achieved:

Metric	Score
Accuracy	98.31%
Precision	97.83%
Recall	98.91%
F1-score	98.36%

The held-out test set contains:

14,427 articles

These results are reported in the project research paper.

4. Baseline Comparison

The project evaluates HXDistil-XGB against several feature/model configurations using the same experimental setting.

Model	Accuracy
Logistic Regression	91.90%
DistilBERT + XGBoost	93.83%
DistilBERT + Metadata	94.91%
TF-IDF + XGBoost	96.58%
HXDistil-XGB	98.31%

The hybrid model provides the strongest performance among the evaluated configurations.

5. Evidence Verification System

Classification alone cannot determine whether every factual statement in an article is correct.

Therefore, HXDistil-XGB includes a claim-level evidence verification pipeline.

For each article:

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

The system extracts factual claims using:

sentence segmentation
named-entity information
fact-oriented verbs
numerical information
claim scoring
boilerplate filtering
duplicate removal

The current prototype limits the number of claims processed per article.

6. Retrieval Engine

The original retrieval prototype used NewsAPI.

During development, this approach produced several problems:

poor search relevance
unrelated articles
weak evidence retrieval
limited coverage

The retrieval stage was therefore redesigned around Google News RSS.

The current retrieval pipeline is:

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

The retrieval engine uses:

Google News RSS
googlenewsdecoder
Sentence-Transformers
all-MiniLM-L6-v2
newspaper3k
trafilatura

The implementation is provided in:

retrieval_engine.py

The module explicitly implements RSS retrieval, URL decoding, semantic ranking, downloading, paragraph extraction and evidence ranking.

7. Source Credibility

The retrieval engine assigns configurable credibility weights to recognized publishers.

Examples include:

Reuters             1.00
Associated Press    1.00
BBC                 0.98
Bloomberg           0.96
CNBC                0.95
TechCrunch          0.92
Business Insider    0.90
The Verge           0.88
Wired               0.88
Forbes              0.84
Fortune             0.84

These weights are used as one component of the evidence reliability calculation rather than as an independent truth label.

8. NLI Verification

Retrieved evidence is passed to a pretrained DeBERTa NLI model:

MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

The model evaluates the relationship between:

Claim ↔ Retrieved Evidence

and produces:

Entailment
Neutral
Contradiction

The system aggregates these claim-level signals across the retrieved evidence.

9. Adaptive Fusion

The final system does not rely exclusively on the HXDistil-XGB classifier.

Instead, it combines:

HXDistil-XGB confidence
claim support
contradiction signals
evidence reliability
source credibility
evidence coverage
source diversity

The result is a final evidence-aware verdict.

Conceptually:

HXDistil-XGB
      +
Claim Verification
      +
Evidence Reliability
      +
Source Quality
      +
Evidence Coverage
      ↓
Adaptive Fusion
      ↓
Final Verdict

The exact fusion weights are treated as an implementation-level design choice rather than learned parameters in the current version.

10. Explainability

HXDistil-XGB uses SHAP to explain the contribution of the fused feature space.

SHAP analysis is performed using:

TreeExplainer

The explanation can be used to inspect the contribution of:

DistilBERT features
TF-IDF features
metadata features

The research implementation evaluates SHAP explanations on a sample of test instances.

The final system additionally exposes:

final verdict
HXDistil-XGB prediction
classifier confidence
extracted claims
claim-level NLI results
evidence sources
evidence quality
source reliability
fusion score
11. Dataset

The project investigated multiple fake-news datasets including:

WELFake
LIAR
ISOT

The training pipeline uses WELFake together with LIAR data for model development, while the principal reported classification evaluation uses a held-out WELFake test split.

The WELFake dataset contains:

72,134 labeled records

A stratified:

80 / 20

train/test split is used.

12. Dataset Leakage Investigation

An important part of this project was discovering substantial overlap between WELFake and ISOT.

The investigation found:

ISOT total rows:                  44,898
ISOT unique articles:             39,105
WELFake total rows:               72,134
WELFake unique articles:          63,678
Articles shared by both:          39,105
Unique ISOT articles absent
from WELFake:                     0

Therefore, the initially observed near-perfect ISOT performance was identified as dataset leakage rather than evidence of genuine cross-dataset generalisation.

This finding is explicitly documented rather than presenting the leaked evaluation as a valid independent benchmark.

13. Retrieval Evaluation

The project also includes a retrieval evaluation procedure.

Example evaluation metrics include:

Retrieval Success Rate
Average RSS Results
Average Decoded Results
Average Downloaded Articles
Average Evidence Items
Pseudo-MRR
Average Semantic Similarity
Trusted Source Ratio

These metrics evaluate the retrieval pipeline independently from the final fake-news classification accuracy.

This distinction is important because:

Good classifier
≠
Good evidence retrieval

and:

Good retrieval
≠
Correct final verdict

The project therefore evaluates the components separately.

14. Repository Structure

A recommended repository structure is:

HXDistil-XGB/
│
├── README.md
│
├── fakenews_prototype6.ipynb
│
├── Prototype2_Train_on_WELFAKE+LIAR.ipynb
├── Prototype2_Evaluation_on_ISOT.ipynb
│
├── retrieval_engine.py
├── webscrape.py
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

Large datasets and model binaries should preferably be hosted separately rather than committed directly to GitHub.

15. Main Files
fakenews_prototype6.ipynb

Main end-to-end prototype.

It integrates:

article scraping
preprocessing
HXDistil-XGB inference
claim extraction
Google News retrieval
evidence verification
adaptive fusion
result generation

The notebook imports the reusable scraper and retrieval modules rather than duplicating the complete retrieval implementation.

retrieval_engine.py

Reusable evidence retrieval module.

Main responsibilities:

Google News RSS
↓
URL decoding
↓
Candidate ranking
↓
Article downloading
↓
Paragraph extraction
↓
Evidence ranking

webscrape.py

Reusable article-processing module.

It supports:

URL article extraction
pasted article text
PDF extraction
title extraction
article categorisation
summary generation

16. Installation

The project was developed primarily in Google Colab.

Install the required Python packages:

pip install newspaper3k beautifulsoup4 pandas nltk spacy gensim scikit-learn lxml_html_clean
pip install transformers sentence-transformers xgboost joblib shap textstat vaderSentiment feedparser
pip install googlenewsdecoder trafilatura sentencepiece pypdf

Install the spaCy English model:

python -m spacy download en_core_web_sm

The project notebooks use these packages for the classification, NLP, scraping, retrieval and explainability pipeline.

17. Model Files

The inference pipeline expects the trained artifacts:

HXDistil_XGB_v2.joblib
tfidf.joblib

The notebook loads these artifacts rather than retraining the model for every inference run.

The trained model and TF-IDF vectorizer were saved specifically to allow subsequent inference without repeating the training process.

18. Running the System
Step 1 — Prepare the environment

Install the dependencies and spaCy model.

Step 2 — Place model files

Place:

HXDistil_XGB_v2.joblib
tfidf.joblib

in the configured model directory.

Step 3 — Place modules beside the notebook
webscrape.py
retrieval_engine.py

should be accessible from the notebook.

Step 4 — Run the main notebook

Open:

fakenews_prototype6.ipynb

and execute the cells sequentially.

Step 5 — Provide an article

The system accepts article inputs through the scraping/inference pipeline.

Supported input types include:

URL
Pasted article text
PDF
Step 6 — Inspect the result

The system produces:

HXDistil-XGB Prediction
Confidence
Extracted Claims
Evidence Sources
Claim Verification
Evidence Quality
Source Reliability
Fusion Score
Final Verdict
Explanation
19. Example Output

A typical result contains:

HXDistil Prediction: REAL
HXDistil Confidence: 79.41%

Claim Support Score: 71.01
Evidence Quality: 53.76
Evidence Reliability: 56.20
Trusted Source Ratio: 25.00
Source Diversity: 66.67
Fusion Confidence: 66.64%

Final Verdict: VERIFIED

The dashboard exposes both the original classifier result and the evidence-based verification signals instead of hiding the intermediate reasoning.

20. Important Design Principle

The system deliberately separates:

Classification

from:

Fact Verification

The HXDistil-XGB model answers:

Does this article resemble the real/fake patterns learned from the training data?

The evidence branch answers:

Are the important factual claims in this article supported or contradicted by independently retrieved evidence?

The final system combines these signals.

This is important because a classifier can incorrectly label a genuine article, while strong external evidence can provide information unavailable to a static training corpus.

21. Limitations

The current implementation has several limitations.

Dataset limitations

The principal benchmark evaluation is based on WELFake, and the investigation found substantial overlap between WELFake and ISOT. Therefore, the leaked ISOT result is not treated as independent generalisation evidence.

Retrieval limitations

Google News RSS provides candidate discovery rather than guaranteed factual truth.

Publisher pages may also block automated downloading, requiring fallback extraction methods.

NLI limitations

NLI predictions depend on:

quality of retrieved evidence
wording of the claim
evidence context
pretrained NLI model limitations
Fusion limitations

The current fusion weights are manually designed implementation parameters rather than learned from a dedicated end-to-end verification dataset.

Real-world misinformation

The system should not be treated as an autonomous authority on truth. A final evidence-grounded verdict should be interpreted together with the retrieved sources and claim-level evidence.

22. Research Reproducibility

The project keeps the major stages modular:

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

The supplementary technical material documents the experiments, implementation decisions and architecture details derived from the project notebooks.

23. Computational Characteristics

The classification branch is designed to remain relatively lightweight compared with a fully fine-tuned transformer classifier.

The major computational components are:

DistilBERT embedding
TF-IDF transformation
XGBoost inference

The evidence branch is more expensive because it additionally requires:

Network retrieval
Article downloading
MiniLM semantic ranking
Paragraph processing
DeBERTa NLI inference

Therefore, evidence verification can be treated as a separate or selective stage when required.

24. Future Work

Planned research directions include:

larger and more diverse datasets
independently collected test datasets
improved claim extraction
stronger retrieval models
learned fusion weights
retrieval calibration
additional source-quality modelling
statistical significance testing
latency benchmarking
larger-scale real-world evaluation
multilingual fake-news verification
improved evidence coverage
human fact-checker comparison
25. Research Paper

The project is accompanied by a research manuscript:

HXDistil-XGB: Hybrid Feature-Fusion Fake News Detection with Evidence Verification

The manuscript describes the proposed architecture, experimental evaluation, ablation studies, explainability analysis and retrieval-augmented verification framework.

The current paper reports the WELFake results of:

Accuracy : 98.31%
Precision: 97.83%
Recall   : 98.91%
F1       : 98.36%

26. Disclaimer

This project is a research prototype.

The system should not be considered a replacement for professional fact-checkers or authoritative sources.

A prediction of:

FAKE

does not by itself prove that an article is false.

Similarly:

REAL

does not guarantee that every factual claim in an article is correct.

The evidence-verification layer is intended to provide additional context and source-attributed evidence to support human interpretation.

27. Citation

If you use this project in academic work, please cite the associated research paper:

@article{hxdistilxgb,
  title   = {HXDistil-XGB: Hybrid Feature-Fusion Fake News Detection
             with Evidence Verification},
  author  = {Dayyan Waseem},
  journal = {Applied Soft Computing},
  year    = {2026}
}


28. Project Status

Current implementation:

[✓] Dataset preprocessing
[✓] Dataset overlap investigation
[✓] DistilBERT feature extraction
[✓] TF-IDF feature extraction
[✓] Metadata feature extraction
[✓] HXDistil-XGB training
[✓] Baseline comparison
[✓] SHAP explainability
[✓] Article scraping
[✓] Claim extraction
[✓] Google News RSS retrieval
[✓] Google URL decoding
[✓] Semantic evidence ranking
[✓] Paragraph-level evidence ranking
[✓] DeBERTa NLI verification
[✓] Source credibility scoring
[✓] Evidence fusion
[✓] Explainable dashboard
[✓] Retrieval evaluation
[✓] JSON/CSV result export
[✓] Research documentation
29. Author

Developed as a research project on:

Explainable Fake News Detection and Evidence-Based Fact Verification

Primary technologies:

Python
PyTorch
Hugging Face Transformers
DistilBERT
XGBoost
TF-IDF
Sentence-Transformers
DeBERTa
spaCy
SHAP
Google News RSS
newspaper3k
trafilatura
scikit-learn
