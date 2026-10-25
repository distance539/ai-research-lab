# 109 — Ablation and interaction on four SciFact probes

4already-observed107developmentqueries ×50fixedcandidates ×2textconditions=400realMiniLMpairs. 2x2cells: Aoriginal/front-gold-evidence; BpureCE/fixedrankblend. Oracle diagnostic only: front knows relevantdocument and evidence; not deployable and not111candidate. 300confirmationqueries remain unevaluated. No training.

## Run
Python3.12, suggested4GiBfreeRAM. Model~91MB, data~2.8MB BEIR plus originalSciFactarchive, framework install additional. Exact dependencies requirements-observed.txt.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-observed.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out replay-smoke --smoke
python audit.py --out replay-smoke
python test_audit.py --out replay-smoke
```

Smoke uses originaltop2plus relevantdocument perquery (deduplicated),16scoredpairs here; no comparison with formal400pairmetrics. Omit --smoke with a new --outdirectory for formal. prepare verifies all required downloadedmodel/tokenizer/datahashes; existingverifiedcache allowed and explicitly passed. No implicit localmodules or absolute paths. Resources are fixedrevision/hash downloads.

Offline audit of bundled results: `python audit.py --out results`, `python test_audit.py --out results`. Standardlibrary only.

## Evidence
PROTOCOL.md/config/registration_snapshot freeze design before109inference, after historical107outcomes; no blindpreregistration claim. materials.json is byte-identical107packet and its pendingmarker historical; human_review_confirmation.json provides actual2026-10-24approval. material_tools does not generate or reselectprobes. SOURCE_MAP/NOTICE/licenses document provenance.

results/pairs.json/raw_runs.json retain everynewscore,inputtokenhash,tokenbudget. cells.json/fusion_trace.json store4rankings. per_query.json stores4cellmetrics,ranks,conditional/averageeffects,interaction. summary.json descriptive only. status/environment/shape_check/identity/budget_ledger document stages,versions,resources,actualshapes,cachekeys.4additionalshape forwards separate from400scoredpairs. No networkdownloads in inference; prepare stage supports freshsetup.

Cellkeys AB:00originalCE,10frontCE,01originalblend,11frontblend. Interaction=(11-10)-(01-00). ApositiveI is not apositiveAeffect. BothBlevels reuse A's singlepass. Samecounts do not imply identicalvisiblecontext. initialretrievalcost notremeasured. No statisticalCI/significancetest for4nonrandomobservedprobes.

VALIDATION.md details actualresults and failures. execution_history logs actualcommands/exitcodes. code.zip is separately archived by article; publicdirectory is the main readerentry.
