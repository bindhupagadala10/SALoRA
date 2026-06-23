# Experiment Log

## DSS V1

Completed:
- Jaccard Similarity
- OOV Rate
- Semantic Distance
- DSS Matrix

## Baseline Models

Model: RoBERTa-base

Domains:
- IMDB
- Twitter
- Amazon
- Yelp

Training:
- 80/10/10 split
- 3 epochs
- Learning rate 2e-5
- Batch size 16

## Transfer Matrix

Completed 4x4 cross-domain evaluation.

Observations:
- Twitter is the most distant domain.
- Amazon and Yelp form a close review-domain cluster.
- IMDB lies between review domains and Twitter.