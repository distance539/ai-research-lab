# Deviations and failures

The initial two-question smoke computed all10 condition/question pairs and passed score parity, but exited1 at the budget projection check. Total per-condition time included fixed task construction over the67349 training rows. Multiplying this setup cost by32x2 was not a valid sample-count projection. The initial projection was about1107s versus600s budget; actual smoke condition processing totaled17.30s.

Before full inference, timing was split into fixed setup, independent scoring and official evaluation; the revised projection keeps fixed setup once and multiplies scoring/evaluation by64. The original600s/4GiB limits, plan, model, examples, conditions, scorer, labels and metrics were not changed. This is a resource-estimator repair, not prompt selection based on smoke scores. All initial outcomes and failure status remain in smoke-budget-failed; the initial executable and raw logs remain with the local article verification records (private filesystem paths excluded from distribution).

The official harness emits a non-repository Git warning when the working directory is not a Git checkout. Actual upstream identity is verified by pinned commit and file hashes; this warning does not indicate failed model inference. No human semantic annotation was performed or claimed. Candidate arithmetic is checked by executable independent audits.

The second smoke still conservatively projected816.96s and exited1 before any full run; it is retained in smoke-budget-recheck with timing/memory. The budget was therefore explicitly revised to1200s on the same local CPU;4GiB unchanged. This supersedes the earlier600s resource limit, not the scientific protocol. Both failures are real and not presented as successful runs.
