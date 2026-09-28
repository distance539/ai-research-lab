# 085 preregistered development protocol

Written 2026-09-28T01:06:19.761424+00:00 before model inference.

Compare fixed RRF k=60 original order against one frozen MS MARCO MiniLM-L6 cross-encoder on exactly the same 50 candidates per query. Select 100 of809 development queries by SHA256(85:query_id), without inspecting labels or reranker scores. Full corpus retrieval was performed in082/083; do not narrow corpus. No hyperparameter search, training, prompt, sigmoid, or test queries. Smoke only estimates runtime and checks shapes. CPU4threads, batch16, max512 longest_first, float32. Threshold900seconds extrapolated/4GiB RSS.

Primary nDCG@10; secondary MRR@10, Recall@10 and Recall@50. Use all annotated relevant documents in IDCG, not only candidate-relevant documents. Label-based ideal candidate order is diagnostic oracle only, never model input. Query failures: no annotated relevant candidate; relevant candidate exists but none in output top10; at least one top10 hit. Also report partial coverage, paired wins/ties/losses and 1-oracle / oracle-achieved gaps. All labels treated under BEIR qrels; unjudged treated nonrelevant by evaluator, not asserted semantically wrong.

Cost report covers new reranking only, separate load/token audit/inference/evaluation; inherited retrieval cost not remeasured. Same candidate membership implies unchanged Recall@50. Automatic cases: largest positive/negative nDCG differences (3 each), deterministic query-ID tie-break. Cases are hypotheses, not human labels. No confidence/generalization claim from this selected development sample. Preserve every failed run.
