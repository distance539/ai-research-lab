# 094 protocol — frozen before any target inference, 2026-10-07

Question: can an independently traced candidate-token likelihood calculation reproduce the official SST-2 task's score on the same 64 development rows?

Reuse original SmolLM2-135M-Instruct revision12fd25f77366fa6b3b4b768ec3050bf629380bac; NOT a trained093 checkpoint. Reuse08864development and128reserved IDs, no reserved inference. Load exact official nyu-mll/glue sst2 revisionbcdcba79d07bc864c1c254ccfcedcce55bcc9a8c parquet; all67349train/872validation rows match previous stanfordnlp/sst2 snapshot by idx,sentence,label. Zero-shot means zero training examples in prompts; training split available for subsequent095 only. No training or task selection by score.

Official harness0.4.9.2 commitad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9, unmodified SST2 YAML. Only dataset transport uses supported custom_dataset to read hash-verified parquet offline. Name,path,configuration,template,choices,label,metric unchanged. Explicit num_fewshot0,no chat template,no BOS,CPUfloat32,eager,one candidate/batch,4threads,max512; overlong inputs abort rather than truncate. Target delimiter is a single space. Score each choice by sum of conditional token log-probabilities, no EOS, no length normalization; ties choose first choice. No decoding is performed.

First independently trace first4development rows and prepare human arithmetic worksheet (not model-output human gold). Required human recomputation must be truthfully recorded; machine audit is not a substitute for human review. Official smoke uses first2 rows, then64 formal rows; save stage timings, commands, failures, shapes, token scores, raw harness evidence privately, and derived public per-row outcomes. If human review is still pending, report it and keep article incomplete; do not fabricate confirmation.

Main test: absolute likelihood differences <=2e-5, same choice/gold/row identity and exact aggregate. Resource preflight: four-row independent run before official formal; estimate <=600seconds and peakRSS<4GiB. No search/early stopping. Confidence intervals disabled; this is protocol audit, not new accuracy estimate on unseen test. Previous192reserved examples across SST/AGNews remain untouched.

Model response/request caches disabled. Identity key includes all model/tokenizer/data file hashes, raw task YAML, effective settings, scorer, and own executable hashes. Data/model download caches are hash checked. No reusable inference cache may be keyed by model name alone.
