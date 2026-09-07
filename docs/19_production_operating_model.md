# Production ANN Operating Model

A production ANN needs an owner, model/data lineage, artifact registry, input/output contract, staged deployment, observability, rollback, retraining policy and business KPI.

## Minimum monitors
- schema/range/missingness
- input-distribution drift
- prediction mix and confidence
- latency/throughput/error rate
- delayed labeled performance and calibration
- downstream business outcomes

## Response loop
alert → investigate → identify data/model/system cause → mitigate or rollback → collect labels → train candidate → validate → test/regression checks → approve → redeploy.
