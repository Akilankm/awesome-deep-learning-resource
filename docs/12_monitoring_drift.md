# Monitoring, Drift, and Retraining

Monitor data quality, input drift, prediction mix, confidence/entropy, latency, error rate and—when labels arrive—real performance. Drift is evidence of changed assumptions, not automatic proof of failure.

Retraining should be governed: detect → investigate → collect labels → train candidate → validate → regression-test → approve → deploy → monitor, with rollback and lineage.

## Learning questions
- What assumption does this stage make?
- What can fail silently?
- What evidence would you monitor?
- How does this affect business or production behavior?
