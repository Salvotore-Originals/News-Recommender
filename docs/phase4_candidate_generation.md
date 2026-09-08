# Phase 4 — Candidate Generation

## 1. Objective

Phase 4 implements corpus-level candidate generation for the EB-NeRD Small dataset.

The objective is to generate a high-recall candidate pool before downstream neural re-ranking. Candidate generation is evaluated independently from the official impression candidate lists so that retrieval operates over the article corpus rather than merely re-ranking already supplied candidates.

The candidate-generation pipeline combines:

1. BM25 lexical retrieval
2. Semantic retrieval using article embeddings
3. Freshness + category-preference retrieval
4. Candidate-source unions

All candidate-generation branches enforce temporal eligibility.

## 2. Temporal Protocol

Candidate generation follows a point-in-time protocol:

- User history event timestamp must satisfy:
  `history_timestamp < impression_time`
- An article is eligible only when:
  `published_time <= impression_time`
- The Fresh + Category branch additionally requires:
  `published_time >= impression_time - 48 hours`

Future articles are therefore excluded from candidate pools.

The Phase 4 validation experiments reported zero future-publication violations for BM25, Semantic, and Fresh + Category retrieval.

## 3. BM25 Candidate Generation

The EB-NeRD Small dataset contains 20,738 articles.

The canonical EB-NeRD BM25 implementation uses:

- BM25Okapi
- `k1 = 1.5`
- `b = 0.75`
- article `text` as the indexed document field
- history-derived lexical queries
- a static BM25 corpus index

The BM25 index is constructed once and reused across impressions.

Temporal eligibility is applied before selecting the final top-K candidates.

### Static BM25 index protocol

The BM25 IDF statistics are computed over the complete article corpus, while article eligibility is determined point-in-time using publication timestamps.

This avoids rebuilding a separate BM25 index for every impression while ensuring that future articles cannot appear in the retrieved candidate pool.

## 4. Semantic Candidate Generation

Semantic retrieval uses the article embedding artifact generated with:

`all-MiniLM-L6-v2`

Embedding dimensionality:

`384`

The article embeddings are normalized before cosine-similarity retrieval.

For each impression, a query embedding is constructed from the user's temporally valid recent history. The semantic branch then retrieves the highest-scoring temporally eligible articles.

## 5. Fresh + Category Candidate Generation

A dedicated Fresh + Category candidate source was implemented to address newly published articles that may not be retrieved effectively by history-only lexical or semantic similarity.

Configuration:

| Parameter | Value |
|---|---:|
| Freshness window | 48 hours |
| Decay period | 24 hours |
| Top categories | 3 |
| Candidates/category | 200 |

The point-in-time category preference is calculated from history strictly preceding the impression timestamp.

Freshness is modeled using exponential decay:

`exp(-age_hours / 24)`

Candidate priority combines category preference with freshness.

## 6. Candidate-Source Union Evaluation

Candidate generation was evaluated using:

1. BM25
2. Semantic
3. Fresh + Category
4. BM25 UNION Fresh
5. Semantic UNION Fresh
6. BM25 UNION Semantic
7. BM25 UNION Semantic UNION Fresh

For a union at K, the top-K candidates from each source are combined by membership. The resulting union is not truncated back to K.

## 7. Phase 4 Evaluation Results

The final evaluation used 1,000 validation impressions from EB-NeRD Small.

| Candidate source | Recall@50 | Recall@100 | Recall@200 | Recall@500 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.031 | 0.039 | 0.056 | 0.085 | 500.00 |
| Semantic | 0.003 | 0.013 | 0.032 | 0.058 | 500.00 |
| Fresh | 0.536 | 0.654 | 0.674 | 0.674 | 164.36 |
| BM25 + Fresh | 0.548 | 0.661 | 0.687 | 0.695 | 652.90 |
| Semantic + Fresh | 0.538 | 0.661 | 0.688 | 0.694 | 657.81 |
| BM25 + Semantic | 0.034 | 0.049 | 0.083 | 0.129 | 969.11 |
| All three | **0.550** | **0.666** | **0.697** | **0.708** | 1117.57 |

Fresh + Category achieves 0.674 Recall@500.

Combining all three sources increases Recall@500 to 0.708, an additional 3.4 percentage points.

## 8. BM25 Optimization and Integration

Phase 4F/4G introduced an optimized BM25 resource layer.

The optimization precomputes article publication-time ordering and uses binary search to determine the eligible corpus prefix for each impression.

The BM25 scoring semantics remain unchanged.

Validation results:

- Optimized BM25 integration tests: 8/8 passed
- Full Phase 4 regression suite: 74/74 passed
- Phase 4E and Phase 4G 1,000-impression recall results: identical

## 9. Runtime Observation

The 1,000-impression Phase 4G evaluation reported:

- BM25 mean generation time: approximately 0.259 seconds/impression
- Semantic mean generation time: approximately 0.021 seconds/impression
- Fresh + Category mean generation time: approximately 0.002 seconds/impression
- Total evaluation time: approximately 302 seconds

The temporal-lookup optimization did not produce a measurable end-to-end speedup in this evaluation. The dominant BM25 cost remains:

`bm25.get_scores(query_tokens)`

Therefore, no end-to-end runtime improvement is claimed for Phase 4G.

A substantially faster BM25 implementation would require optimizing the scoring operation itself and would need separate equivalence validation.

## 10. Correctness and Regression Validation

Phase 4 established:

- Candidate-generation tests pass.
- Temporal eligibility tests pass.
- Fresh + Category tests pass.
- BM25 optimization tests pass.
- Optimized BM25 integration tests pass.
- Full Phase 4 regression suite: 74/74 passed.
- Future-publication violations: 0.
- Phase 4E and Phase 4G result artifacts are identical.

Artifacts:

`data/results/ebnerd/small/candidate_union_recall_1000_phase4e_baseline.csv`

`data/results/ebnerd/small/candidate_union_recall_1000.csv`

## 11. Main Findings

### Finding 1 — Freshness is critical

Fresh + Category retrieval substantially outperforms history-only BM25 and semantic retrieval for candidate discovery.

### Finding 2 — Retrieval sources are complementary

BM25 and Semantic provide additional candidate coverage when combined with Fresh + Category retrieval.

The all-source union reaches 0.708 Recall@500.

### Finding 3 — Temporal correctness is maintained

All three candidate sources produced zero future-publication violations in the 1,000-impression evaluation.

### Finding 4 — BM25 scoring remains the main performance bottleneck

Optimizing temporal eligibility does not materially reduce total runtime because BM25 corpus scoring dominates the per-impression computation.

## 12. Phase 4 Conclusion

Phase 4 establishes a temporally safe, corpus-level candidate-generation layer for EB-NeRD Small.

The final candidate-generation strategy combines:

`BM25 + Semantic + Fresh/Category`

with candidate-source union evaluation demonstrating 0.708 Recall@500 on the 1,000-impression validation experiment.

The candidate-generation layer is ready to supply candidate pools to the downstream neural re-ranking stage.

**Phase 4 is complete.**
