# Dataset Report

## Selected Domains

### IMDB Movie Reviews

* Domain: Movie Reviews
* Raw Size: 50,000
* Cleaned Size: 49,582

### Sentiment140 (Twitter)

* Domain: Social Media
* Raw Size: 1,524,334
* Cleaned Size: 1,519,906

### Amazon Cell Phones & Accessories

* Domain: Product Reviews
* Raw Size: 173,000
* Cleaned Size: 172,760

### Yelp Reviews

* Domain: Business Reviews
* Raw Size: 8,539
* Cleaned Size: 8,537

---

## Cleaning Procedure

Applied to all datasets:

1. Remove null texts
2. Remove empty texts
3. Remove duplicate texts
4. Standardize labels

---

## Sampling Strategy

The smallest cleaned dataset was Yelp with 8,537 usable reviews.

To ensure fair domain comparison, all domains were sampled to 8,537 reviews for DSS computation.

---

## Rationale

Using equal sample sizes prevents domain divergence metrics from being biased by corpus size differences.

This is particularly important because:

* Twitter contains over 1.5 million reviews
* Yelp contains only 8,537 reviews

Without normalization, vocabulary-based metrics would be dominated by dataset size rather than linguistic differences.
