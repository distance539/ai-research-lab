# 111 — A reviewable SciFact research report

Final frozen confirmation for the106–111project. All300 BEIRtest queries (=originalSciFactdev), full5183corpus. BM25+all-MiniLM-L6-v2 eachtop100, official Pyserini RRFk60 select50, then one15000pair CrossEncoderpass shared by pureCE and equalweight rankblend. PureRRF is a cheaper reference. No training, extra conditions, query exclusion or post-confirmation tuning.

## Minimum real run

Tested Python3.12.14/macOSarm64/CPUfloat32. Direct dependencies in requirements.txt and all observed versions in requirements-observed.txt. Allow about4GiB free RAM for safety and2GiB disk plus Python environment. Two models about182MB total, dataset2.8MB, optional ES archive306MB. Network download times not measured. Large resources stay in caller-supplied cache.

```bash
python -m pip install -r requirements.txt
python prepare.py --cache ./cache
python run.py --cache ./cache --out fresh-smoke --smoke
python audit.py --out fresh-smoke
python ce_audit.py --out fresh-smoke
python audit_statistics.py --out fresh-smoke
```

The minimum command after setup is `python run.py --cache ./cache --out fresh-smoke --smoke`. It actually runs5previously observed development queries×10candidates; it never opens a new confirmation subset. Expected outputs: raw pair_scores.csv,rank runs,metrics,budgets,modelshapes,identity,status,pairedresampling anddecision.json. Output directory must be new. CandidateIDs/scores are bundled so this replay needs neither ES nor historical author directories. Download script verifies fixed hashes of corpus,queries,bothsplits,bothmodels and BEIR source. No hidden self-written module dependencies.

## Full unchanged model replay / saved evidence audit

```bash
python run.py --cache ./cache --out fresh-confirmation
python audit.py --out fresh-confirmation
python ce_audit.py --out fresh-confirmation
python audit_statistics.py --out fresh-confirmation
python test_audit.py --out fresh-confirmation
python test_statistics.py --out fresh-confirmation
```

This replays the published frozen method;300queries are now consumed and cannot be reused for model selection while calling them unseen. All real recorded results are under results/. To audit saved evidence without model weights use `python audit.py --out results`, `python ce_audit.py --out results`, `python audit_statistics.py --out results`; these standard-library scripts rewrite only audit records, so copy results first when retaining exact archival bytes. `statistics_report.py --out <copied-results>` recomputes saved resampling with NumPy and overwrites statistical files; it does not rerun inference.

## Rebuild the original retrieval (optional larger run)

prepare.py also fetches fixed BEIR source; `--with-es-macos-arm64` obtains hash-locked official ES7.17.9 for this tested platform. For other platforms install official sameversion; portability and timing remain unverified.

```bash
python prepare.py --cache ./cache --with-es-macos-arm64
ES_JAVA_OPTS='-Xms512m -Xmx512m' ./cache/elasticsearch-7.17.9/bin/elasticsearch \
 -Ediscovery.type=single-node -Enetwork.host=127.0.0.1 \
 -Ehttp.port=19282 -Etransport.port=19382 \
 -Expack.security.enabled=false -Expack.ml.enabled=false \
 -Eingest.geoip.downloader.enabled=false -Ecluster.name=cycle6-111
```

Keep that localhost-only service running in a separate terminal:

```bash
python retrieve.py --cache ./cache --out fresh-retrieval
python audit_retrieval.py --out fresh-retrieval
```

It indexes all5183documents once (newcontent-keyed index, never deletes old index), verifies every indexed field on reuse, encodes entire corpus and all300queries, and saves bm25/dense/RRF/candidate runs. `--smoke` uses five observed developmentqueries but still fullcorpus. Compare `fresh-retrieval/candidates.json` with `retrieval/candidates.json` before using recomputed candidates; published model replay intentionally reads bundled locked inputs. Ties are scoredescending/documentIDstringdescending; tiny floating-point changes can affect tied ordering across platforms. Stop ES with Ctrl-C after use.

## Scientific boundary and provenance

PROTOCOL.md/final_freeze.json precede confirmation retrieval; confirmation_groups.json precedes scores. Primary practical point-gain threshold .01 was registered in106; report uncertainty vszero and.01 separately. Shared-positive-doc graph is a dependence proxy, not a guarantee of independence. All 300 include label-preserving rawtext, no oracle transformation.107semantic approval is a historical record; no new semantic labels inferred here.

Model/tokenizer exactrevision233902d25c440f23af6f7d6e94d2946bac0bee0a; joint512tokens,CPU4threads,batch16,rawIdentitylogits,seed111. Dense revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41,max256,384dimensions. No prompt/decode. CE andblend share15000scoredpairs plus4auxiliaryshape pairs; preflight50pairs separate. Retrieval cost measured separately, pureRRF noadditionalCEcost. Independent audits and corruption checks validate arithmetic; they are not human semantic annotations.

SOURCE_MAP.md,upstream_verification.json,model/data/inputlocks,execution_history.json andVALIDATION.md provide provenance. EXPeriment ledger/prior summaries retain106–110negative controls, developmental gain and107actualapproval. All ownmodules included,Apache2.0; third-party attribution inNOTICE/BEIR/ST/vendorlicenses. No weights,corpusarchive,environment,credentials,privateconfig or author-specific paths distributed.

## Recorded final result

RRF nDCG10 .7113379806;CE .6927649047;blend .7103051471. Blend-minusCE mean .0175402424,65wins197ties38losses;component95%interval[-.0023445088,.0382749873]. Point practical threshold .01 met, interval covers both0 and.01. Blend-minusRRF -.0010328335,interval[-.0199434018,.0168316567]. No stable-superiority claim and no retuning. Recall50 .9403333333 allmethods,17queries withzero relevantcandidate. Fresh retrieval51.5798s,CE391.0365s,CEpipeline396.3307s plusstatistics4.3434s;CEpeakRSS1075986432bytes.5323656nonpaddingtokens. Source model output,oracle and all failurecases retained.
