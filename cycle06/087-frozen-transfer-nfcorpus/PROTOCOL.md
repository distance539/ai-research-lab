# 087 Frozen transfer protocol
Registered: 2026-09-30T09:03:41.133762+08:00; before NFCorpus download/inference.

Question: Do dense-minus-BM25 and RRF60-minus-BM25 directions persist from SciFact to BEIR NFCorpus? Dataset/query-task transfer within scientific/medical retrieval; not an isolated far-domain effect.

Freeze082 BM25 ES7.17.9 build, english analyzer, best_fields title/txt,tie_breaker .5,default k1=1.2,b=.75,one shard. Freeze083 all-MiniLM-L6-v2 revision/tokenizer,256tokens,title-space-text,384dimensions,cos_sim,CPUfloat32,batch16,4threads. Freeze084 RRF k60,two top100 input lists,top100 output,ID descending ties. No target tuning/retraining/reranker/086 preprocessing variants. Seed87 is execution identity only.

All NFCorpus BEIR test queries, full BEIR corpus. Smoke5queries/32documents: resource/correctness only. Budget900seconds neural encoding/<4GiB PythonRSS; ES512MiB heap,server peakRSS unmeasured. On resource failure preserve incomplete,no score-driven resampling. SciFact809development metrics reused from fixed public runs;300confirmation untouched.

Primary nDCG10: trec_eval LINEAR relevance gain,retain graded labels. Recall100/MRR10 use relevance>0. Paired contrasts within dataset; cross-dataset difference descriptive,not paired causal domain effect/CI. Save all outputs,graded audit,query scores,contrasts,failures,identities,timing. Link-based relevance differs from SciFact.

Hypotheses: dense advantage sign persists; RRF advantage sign persists. Report negative/null results. No parameter promotion after target inspection. Protocol hash saved before inference.
