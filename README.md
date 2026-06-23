# SALoRA

**Shift-Aware Parameter-Efficient Adaptation for Cross-Domain Sentiment Analysis**

## Overview

SALoRA is a journal publication effort focused on domain adaptation in sentiment analysis.

The project investigates whether measured domain divergence can be used to automatically determine the amount of adaptation capacity required for efficient cross-domain transfer learning.

The central hypothesis is:

> Larger domain shifts require larger adaptation capacity, while smaller domain shifts can be handled using lower-capacity adapters.

To test this hypothesis, the project introduces:

1. **DSS (Domain Shift Score)** – a metric that quantifies domain divergence.
2. **SALoRA (Shift-Aware LoRA)** – a framework that uses DSS to automatically select LoRA rank.

---

# Research Question

Can measured domain divergence be used to automatically determine how much adaptation capacity is required for efficient cross-domain transfer?

---

# Datasets

| Dataset        | Domain                 |
| -------------- | ---------------------- |
| IMDB           | Movie Reviews          |
| Sentiment140   | Twitter / Social Media |
| Amazon Reviews | Product Reviews        |
| Yelp Reviews   | Business Reviews       |

All datasets are cleaned and sampled for fair comparison.

---

# Methodology

## DSS v1

Current implementation combines:

### Jaccard Similarity

Measures vocabulary overlap between domains.

### OOV Rate

Measures how much target-domain vocabulary is unseen by the source domain.

### DSS v1 Formula

DSS_v1 = 0.5 × (1 − Jaccard) + 0.5 × OOV

---

# Current Results

## Vocabulary Sizes

| Dataset | Vocabulary |
| ------- | ---------: |
| IMDB    |     48,163 |
| Twitter |     12,169 |
| Amazon  |     15,996 |
| Yelp    |     26,600 |

## Example DSS v1 Scores

| Transfer       |    DSS |
| -------------- | -----: |
| IMDB → Twitter | 0.6011 |
| IMDB → Amazon  | 0.5767 |
| IMDB → Yelp    | 0.5416 |
| Twitter → IMDB | 0.8431 |

Highest observed shift:

Twitter → IMDB

Lowest observed shift:

IMDB → Yelp

---

# Project Structure

```text
SALoRA/

├── data/
│   ├── raw/
│   ├── processed/
│   └── dss_samples/
│
├── docs/
│
├── experiments/
│
├── notebooks/
│
├── paper/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── metrics/
│
├── src/
│   ├── dss/
│   ├── training/
│   ├── models/
│   └── evaluation/
│
├── README.md
└── requirements.txt
```

---

# Current Progress

## Completed

* Dataset collection
* Dataset preprocessing
* Dataset auditing
* DSS sampling
* Vocabulary analysis
* Jaccard similarity
* OOV analysis
* DSS v1
* Embedding generation

## In Progress

* Full DSS analysis
* Zero-shot transfer experiments

## Planned

* DSS v2
* LoRA baselines
* SALoRA framework
* Correlation analysis
* Journal manuscript

---

# Author

Bindhu Pagadala

Manipal Institute of Technology Bengaluru

B.Tech Computer Science Engineering
