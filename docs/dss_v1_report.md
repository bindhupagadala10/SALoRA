# DSS Version 1 Report

## Vocabulary Statistics

| Domain  | Vocabulary Size |
| ------- | --------------: |
| IMDB    |          48,163 |
| Twitter |          12,169 |
| Amazon  |          15,996 |
| Yelp    |          26,600 |

---

## Jaccard Similarity

| Domain Pair      | Jaccard |
| ---------------- | ------- |
| IMDB ↔ Twitter   | 0.1502  |
| IMDB ↔ Amazon    | 0.1943  |
| IMDB ↔ Yelp      | 0.2881  |
| Twitter ↔ Amazon | 0.2212  |
| Twitter ↔ Yelp   | 0.2232  |
| Amazon ↔ Yelp    | 0.2731  |

Observations:

* Lowest similarity: IMDB ↔ Twitter
* Highest similarity: IMDB ↔ Yelp
* All domains exhibit measurable vocabulary divergence

---

## OOV Analysis

| Transfer         | OOV    |
| ---------------- | ------ |
| IMDB → Twitter   | 0.3525 |
| IMDB → Amazon    | 0.3476 |
| IMDB → Yelp      | 0.3713 |
| Twitter → IMDB   | 0.8364 |
| Twitter → Amazon | 0.6810 |
| Twitter → Yelp   | 0.7341 |
| Amazon → IMDB    | 0.7833 |
| Amazon → Twitter | 0.5807 |
| Amazon → Yelp    | 0.6565 |
| Yelp → IMDB      | 0.6528 |
| Yelp → Twitter   | 0.4188 |
| Yelp → Amazon    | 0.4288 |

---

## DSS v1

Formula:

DSS_v1 = 0.5 × (1 − Jaccard) + 0.5 × OOV

Results using IMDB as source:

| Transfer       | DSS v1 |
| -------------- | ------ |
| IMDB → Twitter | 0.6011 |
| IMDB → Amazon  | 0.5767 |
| IMDB → Yelp    | 0.5416 |

---

## Preliminary Findings

The DSS v1 ranking aligns with intuition:

1. IMDB → Twitter (highest shift)
2. IMDB → Amazon (moderate shift)
3. IMDB → Yelp (lowest shift)

This suggests that vocabulary overlap and OOV rates capture meaningful domain divergence signals.

The next step is to incorporate semantic divergence using Sentence Transformer embeddings to build DSS v2.
