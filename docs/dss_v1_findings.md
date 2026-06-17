# DSS v1 Findings

## Objective

The objective of DSS v1 is to quantify domain shift between sentiment analysis datasets using lexical characteristics before model training.

DSS v1 combines:

* Jaccard Distance (Vocabulary Divergence)
* Out-of-Vocabulary (OOV) Rate

Formula:

DSS_v1 = 0.5 × (1 − Jaccard) + 0.5 × OOV

---

## Key Observations

### 1. DSS is Directional

Unlike Jaccard similarity, DSS is directional because OOV depends on the source and target domains.

Example:

* IMDB → Twitter = 0.6011
* Twitter → IMDB = 0.8431

This indicates that transferring from Twitter to IMDB is substantially more difficult than transferring from IMDB to Twitter.

---

### 2. Twitter Exhibits the Highest Domain Shift

Transfers originating from Twitter consistently produce high DSS values.

Examples:

* Twitter → IMDB = 0.8431
* Twitter → Yelp = 0.7555
* Twitter → Amazon = 0.7299

This is likely caused by the short, informal, and slang-heavy nature of Twitter text compared to review-based datasets.

---

### 3. IMDB and Yelp Appear Relatively Similar

The lowest DSS score was observed for:

* IMDB → Yelp = 0.5416

This suggests that both datasets share characteristics associated with review-oriented language, including longer text length and opinion-rich content.

---

### 4. OOV Contributes Significantly to Domain Shift

Many domain pairs exhibited moderate Jaccard similarity but high OOV rates.

This indicates that vocabulary overlap alone is insufficient for characterizing transfer difficulty.

The OOV component captures target-domain terminology that is absent from the source domain.

---

### 5. DSS Produces a Meaningful Range of Values

Observed DSS values ranged from:

* Minimum: 0.5416
* Maximum: 0.8431

This spread suggests that DSS is capable of distinguishing between low-shift and high-shift transfer scenarios.

---

### 6. Preliminary Validation of the DSS Hypothesis

The resulting rankings align with intuitive expectations:

High Shift:

* Twitter → IMDB
* Amazon → IMDB
* Twitter → Yelp

Lower Shift:

* IMDB → Yelp
* IMDB → Amazon

This provides initial evidence that DSS captures meaningful characteristics of domain divergence.

---

## Current Limitation

DSS v1 relies exclusively on lexical information.

It measures:

* Vocabulary overlap
* Vocabulary absence

However, it does not capture semantic similarity between domains.

For example, two domains may discuss similar concepts using different words and therefore appear artificially distant under DSS v1.

This limitation motivates the development of DSS v2 using semantic representations derived from sentence-transformer embeddings.

---

## Next Step

The next stage of the project is to evaluate whether DSS v1 correlates with actual transfer performance degradation.

This will be measured through zero-shot cross-domain sentiment classification experiments using RoBERTa.

The resulting performance drops will be compared against DSS values to determine whether domain shift can be used as a predictor of transfer difficulty.
