# 110 protocol

Frozen at 2026-10-26T09:03:14.948740+08:00 before new110 inference/statistics, after108/109 outcomes were known. Not outcome-blind preregistration.

Continue106 SciFact line: exact100 development queries,5183corpus,50candidates,one CEpass and fixed k60 equalweight blend. No selection of model,weight,length or query after statistics. No training.

Primary descriptive uncertainty:10000 paired cluster bootstrap draws seed110. Build undirected query graph using shared positive-qrels documents (including those absent from candidates); components by transitive closure. Uniformly draw G components with replacement G times, pool all queries with multiplicity, divide total differences by total query count. Preserve query-weighted estimand. 2.5/97.5 percentiles linear interpolation. Query bootstrap is sensitivity only; neither removes development selection bias or guarantees population coverage. Save all indices and replicate differences. Leave-one-component-out range diagnoses influence, not CI.

Report all three comparisons blend-CE,blend-RRF,CE-RRF; primary remains blend-CE. Enumerate every prior registered experiment106–109, with oracle transform limitations. Fourprobe109 only descriptive, no fourquery bootstrap claim. Automatic three lowest blend-CE query IDs for failure audit; no new semantic labels.

Smoke5x10 estimates resource; formal100x50 if <=1200s and <=4GiB. Four auxiliary shape forward pairs separately disclosed. Existing cache hashchecked; network downloads not retimed. Frozen300confirmation remains untouched.

111 freeze: same model/tokenizer,raw title-space-abstract,max512,50candidates,k60weights1/1,ties scoreDESC/stringID DESC. Same shared-positive-document connected-component algorithm,10000draws seed111,primary paired mean blend-CE and component interval,query sensitivity. Form confirmation groups from goldIDs before merging scores; do not adjust rule after seeing outcomes. Evaluate once on300confirmation, report blend-RRF as well. Registered practical threshold remains mean>=0.01; report whether interval covers0 and0.01 separately, do not retrofit a new binary success test. No tuning/retesting after confirmation. Shared-doc clustering is a dependence proxy; topic/claim dependencies may remain.
