# 106 protocol and forward research registry

Registered: 2026-10-22T09:07:37.429061+08:00

This is a development failure audit and fresh real-model replay, not a new efficacy trial. The starting100 queries and085/086 outcomes were already observed; these hypotheses are post-observation. `research_registry.json` freezes the forward questions and stopping rules before106 replay and107 interventions. It is a local dated registration, not an externally preregistered study.

Freeze all085 candidate IDs/top50, weights/tokenizer revision, title-space-abstract, max512 paired longest_first, CPUfp32, batch16, four threads. Set seed106; inference is eval/no_grad. Do not retrieve again or read confirmation labels. 809development queries have historical exposure; only300BEIRtest queries remain for111. No training.

Run5queries x10candidates smoke first, then100x50 if measured extrapolation<900s andRSS<4GiB. Save all failures, scores, token lengths, identities, stages and shape checks. Reference085 scores are used only for reproducibility checks.

New deterministic taxonomy, applied in this precedence: candidate_absent, candidate_hit but top10_miss, top10_hit with nDCG regression, top10_hit with gain, top10_hit with tie (epsilon1e-12). This partitions100queries. Separately, for all91candidate-hit queries, cross-tab any relevant candidate truncated at512 against nDCG regression. The cutoff512 is the frozen model input limit, not a fitted threshold. Never classify a missing candidate as untruncated. No semantic error cause is assigned by this script.

Select one example per nonempty category by smallest SHA256(106:query_id). Show IDs/ranks only; this is descriptive selection, not a representative sample. No new query text paraphrases or label changes. No significance test or confirmatory claim from these observed development rows. No extra length variants run in106;086 is historical context.

Main prospective hypothesis and two competing explanations, metrics, controls, budgets and stopping rules are in research_registry.json. Preserve negative results; 107-111 remain unexecuted. A descriptive truncation association cannot prove lost evidence or score-domain mismatch.
