# 086 — pre-inference protocol
Registered before any086 inference: 2026-09-29T09:05:18.210802+08:00

Research question: with exactly the same100 development queries and50 RRF candidates per query as085, can document title inclusion and maximum paired length change the ranking of CrossEncoder versus frozen RRF?
Primary outcome: macro nDCG@10, secondary MRR@10/Recall@10; Recall@50 invariant. Four fixed conditions title256/title512/abstract256/abstract512. All results reported; no winning condition promoted to confirmation. Seed86 only controls runtime; query selection remains seed85. No training. Model/tokenizer revision233902d25c440f23af6f7d6e94d2946bac0bee0a. CPU float32,4threads,batch16,Identity logits. Ties document ID string descending. Original5183-document retrieval unchanged.300BEIR test queries not evaluated. This is development exploration, not a complete benchmark or a new retrieval run.

CrossEncoder and official BEIR rerank unchanged. For abstract-only set corpus title to empty before official concatenation. Max length includes query and special tokens; longest_first truncation. Save actual input IDs hash, retained query/title/abstract token counts via fast-tokenizer offsets, and relevant-pair flags. Offsets classify a token crossing title boundary as abstract. Token loss is not evidence-sentence loss.

Report title effects at each length and length effects at each title setting; difference-in-differences is descriptive of this frozen system, not a population interaction inference. Predeclared examples: three largest absolute query-level length effects under title, ties query ID string ascending. No human gold or causal claims about evidence locations.

Resource decision after smoke5queries×10candidates×4, using only runtime/RSS, not quality. Proceed100×50×4 if scaled inference<1200sec and RSS<4GiB. Failure preserves this index. Independent audit recomputes all metrics, identical candidate sets, per-pair ranks, actual token accounting and aggregate effects. Minimal archive replay uses all4 real smoke conditions with hash-verified external cache; complete full replay optional unless differences arise.
