# 098 frozen analysis protocol

Recorded before 098 recomputation and inference, after the 093 results were already public. This is exploratory, not preregistered blind confirmation. Compare A=shuffle93 and B=shuffle94 checkpoints from fixed 093 commit; choose the first two shuffled seeds in numeric order, not best versus worst. Same 64 development rows, same JSON prompt, greedy32 decoding, strict JSON joint metric. No training or scoring changes; all 192 confirmation rows stay uninferred.

Primary difference B-A; row-paired percentile bootstrap, SciPy1.13.1, 20000 resamples, seed9800, confidence .95, linear percentiles. Report all four paired correctness cells and exact two-sided binomial/McNemar discordance diagnostic. Report independent resampling ONLY as a deliberately wrong diagnostic for these paired data. Explicitly retain degenerate/sparse-discordance limits, unknown document grouping and selection bias. No equivalence claim, no probability of model superiority, no training-seed-population claim.

New experiment: load both hashed real SFT checkpoints, first 2-row resource smoke, then regenerate all64 JSON answers and assert exact tokens/prompt hashes/scores against frozen predictions. If local checkpoints unavailable, fetch fixed public093 sources and reconstruct via its unchanged run.py. Resource cap1200s/6GiB; no confirmation inference. Identity includes model/tokenizer revisions, data hashes, checkpoint bytes/numerical identity, prompt/config, decoding and all source hashes.

Statistical replay has no model inference. Model regeneration writes new evidence separately. Do not call a replay a new model run. Group/row mismatch, duplicate IDs, wrong labels, cache mismatch or failed official comparison must abort.
