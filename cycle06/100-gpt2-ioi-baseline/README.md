# 100 GPT-2 small IOI behavior baseline

Scope:64 controlled synthetic IOI prefixes from two public templates, ABBA/BABA, eight names/eight cyclic pairs with symmetric role exchange. Actual official GPT-2 small124,439,808parameters, Transformers4.49.0, CPUfloat32/eager. No training, generation, interventions or circuit localization. All64rows correct is a restricted design result, not a full benchmark score.

## Minimal independent audit (no dependencies)

```sh
python audit.py
python test_checks.py
```

Expected:64row identities, token positions, differences, probability relations and aggregates pass; four deliberate corruptions rejected. This is independent arithmetic, not human gold labeling.

## Reproduce actual end-to-end smoke

Python3.12.14 used. Reused prior099environment;25required package versions locked, observed full environment separately recorded. No credentials needed. Around551MB fixed model/tokenizer assets, environment additional; preset4GiB RSS/1200s budget. Observed formal peak1,604,501,504bytes on macOSarm64. Crosshardware numerical equality is not guaranteed.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment.lock.txt
python prepare.py --cache ./cache
TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --smoke --out smoke-new
python audit.py --results smoke-new
```

Outputs: records.json (raw scores, exact synthetic text, token IDs/positions), summary.json, stages.json, environment.json, identity.json. Entry refuses existing output directory, input/source changes and truncation. Cache contains only fixed revision assets with8SHA256checks. Identity includes model/tokenizer resources, prompts/samples, scientific code/config/protocol and no-generation scoring setting. Does not reuse output caches.

## Full frozen run and public-template reconstruction

```sh
python run.py --cache ./cache --out results-new
python audit.py --results results-new
python rebuild_samples.py --out samples-rebuilt.json
```

Compare rebuilt samples with samples.json. rebuild_samples fetches one pinned official file, verifies SHA256, parses only literal names/templates using AST and reproduces64rows. It is an independent provenance check added after inference; run.py consumes the pre-frozen samples. No hidden parent modules or author-machine paths required.

## Actual evidence and limitations

results/ contains64formal rows, smoke/ contains2rows, smoke-readout-failed/ preserves original stopped run and source snapshot. Both final passes exit0. Initial vector readout check failed due GEMV/GEMM rounding (0.0001602); same-shape readout exactly0error. See DEVIATIONS.md. PROTOCOL.md/plan.json froze actual code before final smoke and formal; manifests preserve full distribution identities. Main mean4.44598639, preference64/64, vocabularytop1 64/64, mean absolute IOprob0.59983986. No significance interval: repeated names/templates make rows dependent. Other8names/templates2,3 remain uninferred for later104. TransformerLens port/activation changes need separate consistency checks.

## Reading route

SOURCE_MAP.md links pinned official modules and licenses. Read samples.json and config.json first; then run.py preparation/assertions/model readout; audit.py independent recomputation; results/summary.json and raw records. Original paper: Wang, Variengien, Conmy, Shlegeris, Steinhardt (2022), https://arxiv.org/abs/2211.00593v1 . Our limited factorial design is not the paper's100,000sample evaluation or26head circuit.
