# NLP Projects

This directory contains project-style learning artifacts that apply the concepts from the NLP notebook track to complete ML/NLP lifecycles.

## Flagship project

- [Customer Support Intelligence](customer_support_intelligence/) — one complete end-to-end NLP system that begins from **ready-made raw data** and covers data contracts, EDA, explicit hypothesis formation, leakage prevention, train/validation/test splitting, baselines, TF-IDF models, validation tuning, final evaluation, slice/error analysis, entity extraction, retrieval, serialization, inference, robustness, monitoring, drift and retraining policy.

The project is intentionally centered on **one deeply documented notebook** rather than many shallow projects.

The raw corpus is disclosed as educational synthetic/curated data, but the construction recipe is intentionally absent from the student-facing code so learners must discover patterns through analysis.

## How to use the flagship project

Do not read it as a recipe. At every stage, pause and make a prediction first:

1. What signal do I think exists?
2. Which transformation will preserve or destroy that signal?
3. If I change this parameter, what quantity should move?
4. If the result does not move as expected, which layer should I debug?

The flagship notebook contains visual maps for the overall reasoning loop, representation changes, error-debugging flow, and production feedback.
