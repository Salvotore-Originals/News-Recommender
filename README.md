# News Recommendation System

A reproducible hybrid news-recommendation pipeline built using the **MIND** and **EB-NeRD** datasets. The system combines lexical retrieval, semantic similarity, and behavioral personalization to rank candidate news articles by estimated click relevance.

## Overview

News recommendation is a ranking problem in which a user is presented with an impression containing multiple candidate articles. The goal is to place articles that the user is most likely to click near the top of the ranked list.

This project implements an end-to-end recommendation pipeline based on three complementary signals:

* **Lexical relevance:** BM25 retrieval using article text and user click history.
* **Semantic relevance:** Sentence Transformer embeddings using `all-MiniLM-L6-v2`.
* **Behavioral relevance:** User-history and category-preference features.

The project emphasizes **reproducibility, temporal evaluation, leakage prevention, modularity, and competition-ready submission generation**.

---

## System Architecture

```text
                         ┌──────────────────┐
                         │  Raw MIND /      │
                         │  EB-NeRD Data    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Data Cleaning &  │
                         │ Temporal Splits  │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌──────────┐ ┌───────────┐ ┌────────────┐
              │  BM25    │ │ Semantic  │ │ Behavioral │
              │ Retrieval│ │ Embeddings│ │ Features   │
              └────┬─────┘ └─────┬─────┘ └─────┬──────┘
                   │              │             │
                   └──────────────┼─────────────┘
                                  ▼
                         ┌──────────────────┐
                         │ Hybrid Ranking   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Temporal        │
                         │ Evaluation      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Official         │
                         │ Submission       │
                         └──────────────────┘
```

---

## Datasets

### MIND

The project uses the Microsoft News Dataset (MIND), a benchmark dataset for news recommendation.

The processed MIND dataset contains:

* **51,282 articles**
* **50,000 users**
* Article title, abstract, category and subcategory information
* User click histories
* Impression-level candidate sets

The processed interaction data was divided chronologically into training, validation, and test periods.

| Split      | Interaction Rows | Impressions |
| ---------- | ---------------: | ----------: |
| Train      |        2,210,878 |      61,417 |
| Validation |        1,178,464 |      33,654 |
| Test       |        2,454,102 |           — |

A temporal split was used instead of random splitting to avoid future information leaking into earlier predictions.

### EB-NeRD

The pipeline was subsequently extended to EB-NeRD for large-scale news recommendation evaluation and the RecSys Challenge 2024 setting.

The EB-NeRD workflow includes:

* Test-data preparation
* Article feature construction
* BM25 retrieval
* Semantic embeddings
* Behavioral features
* Hybrid inference
* Large-scale submission generation
* DuckDB/chunked processing for large outputs

---

## Methodology

### 1. Lexical Retrieval — BM25

BM25 is used to capture explicit lexical overlap between user interests and candidate article content.

The implementation uses:

```text
k1 = 1.5
b  = 0.75
```

For MIND, the user's historical clicked articles are converted into a bounded lexical query. Query construction includes practical processing such as:

* Recency handling
* Stopword removal
* Duplicate removal
* Query-length limiting

BM25 was selected because it is computationally efficient, interpretable, and effective when important news entities and terms occur explicitly in titles and abstracts.

---

### 2. Semantic Retrieval

Semantic retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting MIND article embeddings have:

```text
Embedding dimension: 384
Number of articles: 51,282
Data type: float32
```

Semantic similarity complements BM25 by allowing articles with similar meaning to receive high scores even when their exact wording differs.

---

### 3. Behavioral Modeling

User interaction history is used to personalize recommendations.

The behavioral feature store includes user-category preferences derived from historical interactions.

The MIND behavioral feature construction processed:

* **5,107,639 history rows**
* **49,108 users**
* **283,756 user-category pairs**

The behavioral features are constructed from historical information available before the prediction point to preserve temporal validity.

---

### 4. Hybrid Ranking

The final ranking stage combines:

```text
Lexical relevance
       +
Semantic relevance
       +
Behavioral relevance
       ↓
Hybrid candidate ranking
```

The motivation is that each signal captures a different aspect of relevance:

| Signal     | Main question                                                     |
| ---------- | ----------------------------------------------------------------- |
| BM25       | Does the candidate contain terms related to the user's interests? |
| Semantic   | Is the candidate conceptually similar to the user's interests?    |
| Behavioral | Does this user historically prefer this type of content?          |

---

## Evaluation

The project uses ranking-oriented evaluation metrics.

### Mean Reciprocal Rank

For an impression whose first relevant article occurs at rank `r`:

```text
MRR = (1 / N) Σ (1 / r)
```

MRR rewards systems that place a relevant article very near the top of the ranking.

### Hit@K

Hit@K measures whether at least one relevant article appears within the first `K` positions.

### nDCG@K

nDCG applies a logarithmic discount to lower-ranked results and measures the quality of the ranking relative to an ideal ranking.

---

## MIND Semantic Retrieval Results

A larger validation evaluation was performed over **32,878 impressions**.

| Metric  |       Result |
| ------- | -----------: |
| Hit@5   | **0.532757** |
| Hit@10  | **0.709258** |
| MRR     | **0.344645** |
| Runtime | **458.13 s** |

A smaller smoke test over 100 impressions produced:

| Metric  |   Result |
| ------- | -------: |
| Hit@5   |     0.55 |
| Hit@10  |     0.69 |
| MRR     | 0.368859 |
| Runtime |   1.56 s |

These experiments established that the semantic representation provides useful retrieval capability before combining it with lexical and behavioral signals.

---

## Temporal Evaluation and Leakage Prevention

Random interaction splits were deliberately avoided.

The evaluation follows:

```text
Past interactions
       ↓
Training
       ↓
Validation
       ↓
Test
```

This is particularly important for news recommendation because both user interests and article popularity change over time.

The project includes tests for temporal leakage and validates that future interaction information is not incorrectly used to construct historical features.

---

## Competition Submission

The project includes submission builders for both MIND and EB-NeRD.

For MIND, the final prediction artifact contained:

```text
2,370,727 impressions
93,115,001 ranked candidate rows
```

The MIND submission format required each line to contain:

```text
<impression_id> [candidate_position_1,candidate_position_2,...]
```

rather than the article IDs themselves.

For example:

```text
1 [9,10,1,14,8,16,12,11,7,15,13,5,6,4,3,2]
```

The submission builder maps the model's predicted article IDs back to their original candidate positions.

### Submission issues identified and fixed

Two important submission-format problems were encountered:

1. **Incorrect output representation**

   The initial builder wrote article IDs directly:

   ```text
   1 N98744 N55764 N101071 ...
   ```

   The evaluator expected an impression ID followed by a single bracketed rank list. The builder was corrected accordingly.

2. **Incorrect impression ordering**

   Impression IDs were initially sorted lexicographically, resulting in:

   ```text
   1
   10
   100
   1000
   ...
   2
   ```

   The builder was changed to sort impression IDs numerically:

   ```text
   1
   2
   3
   4
   5
   ...
   ```

The final submission file was locally validated with:

```text
Total lines: 2,370,727
Bad format lines: 0
```

The generated ZIP contained:

```text
prediction.txt
```

and passed ZIP integrity validation.

---

## Project Phases

The implementation was developed incrementally:

| Phase   | Work                               | Status   |
| ------- | ---------------------------------- | -------- |
| Phase 1 | Data & reproducibility pipeline    | Complete |
| Phase 2 | Lexical BM25 retrieval             | Complete |
| Phase 3 | Semantic retrieval                 | Complete |
| Phase 4 | Behavioral modeling                | Complete |
| Phase 5 | Hybrid ranking                     | Complete |
| Phase 6 | Temporal evaluation & ablation     | Complete |
| Phase 7 | EB-NeRD + MIND evaluation          | Complete |
| Phase 8 | Competition submission & reporting | Complete |

The Git repository preserves the development history through phase-level commits.

---

## Repository Structure

```text
News-Recommender/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── embeddings/
│   └── results/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── retrieval/
│   └── evaluation/
│
├── tests/
│
├── requirements.txt
├── pipeline.py
└── README.md
```

Large generated datasets, embeddings, predictions, and submission artifacts are kept outside the Git history.

---

## Reproducibility

### Environment

The project was developed and tested using:

```text
Python 3.11.9
```

The project dependencies are specified in:

```text
requirements.txt
```

### Basic setup

```powershell
git clone https://github.com/Salvotore-Originals/News-Recommender.git
cd News-Recommender

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Dataset-specific scripts under `src/data/`, retrieval scripts under `src/retrieval/`, and evaluation/submission scripts under `src/evaluation/` provide the main pipeline stages.

Tests can be executed with:

```powershell
pytest
```

---

## Testing

The project contains automated tests covering important pipeline components, including:

* BM25 retrieval
* EB-NeRD BM25 processing
* Semantic retrieval
* MIND semantic evaluation
* Temporal leakage
* MIND inference
* Submission-related processing

The tests are intended to catch both algorithmic errors and data/pipeline failures.

---

## Where the System Breaks at 10×

The main scalability limitations are computational rather than conceptual.

### 1. Prediction volume

The final MIND artifact contained:

```text
93,115,001 prediction rows
```

At approximately 10× this scale, naive full-table loading and transformation would create substantial memory and serialization pressure.

**Current mitigation:** chunked processing and DuckDB-based workflows were introduced for large EB-NeRD outputs.

### 2. Dense retrieval

Computing dense similarity against a much larger article catalogue becomes increasingly expensive.

**Better solution at 10×:** approximate nearest-neighbor indexing rather than exhaustive similarity search.

### 3. BM25 retrieval

Exact lexical retrieval also becomes more expensive as the article collection grows.

**Better solution:** efficient inverted indexes and candidate-generation limits before hybrid ranking.

### 4. Behavioral features

Continuously updating user histories at much larger scale requires bounded state and incremental computation.

A production implementation would require:

* Incremental feature updates
* Caching
* Event-time processing
* Distributed feature computation

### 5. Observability

At 10× scale, the pipeline would require systematic monitoring of:

* Stage latency
* Memory consumption
* Candidate coverage
* Score distributions
* Failed impressions
* Data drift
* Feature freshness

The current system is therefore best characterized as a **reproducible research/competition pipeline**, not a distributed production recommender.

---

## Limitations

The current system has several limitations:

* `all-MiniLM-L6-v2` is a compact general-purpose encoder rather than a news-specific language model.
* Behavioral personalization is feature-based rather than a learned sequential model.
* Hybrid ranking is less expressive than a learned ranking function.
* The system primarily optimizes click-oriented ranking accuracy.
* Diversity, novelty, serendipity, and coverage are not the primary optimization objectives.
* Large-scale production deployment would require distributed and incremental processing.

---

## Future Work

Potential improvements include:

1. Train a learned ranking model using lexical, semantic, and behavioral features.
2. Introduce approximate nearest-neighbor semantic retrieval.
3. Develop stronger sequential user representations.
4. Add news-specific or domain-adapted language models.
5. Optimize beyond-accuracy metrics such as diversity, novelty, and serendipity.
6. Introduce incremental feature computation and distributed inference.
7. Add comprehensive pipeline monitoring and experiment tracking.

---

## References

1. Wu, F. et al. **MIND: A Large-scale Dataset for News Recommendation.** ACL, 2020.
2. Kruse, J. et al. **EB-NeRD: A Large-Scale Dataset for News Recommendation.** RecSys Challenge '24, 2024.
3. Robertson, S. and Zaragoza, H. **The Probabilistic Relevance Framework: BM25 and Beyond.** Foundations and Trends in Information Retrieval, 2009.
4. Reimers, N. and Gurevych, I. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.** EMNLP-IJCNLP, 2019.

## Project Repository

**GitHub:** https://github.com/Salvotore-Originals/News-Recommender
