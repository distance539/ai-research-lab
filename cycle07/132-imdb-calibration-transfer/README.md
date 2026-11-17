# Frozen calibration: SST-2 → IMDb
Requires Python 3.12, CPU, about 1 GB disk beyond the runtime; allow 6 GiB RAM. Install with `python -m pip install -r requirements-observed.txt` in an isolated environment. Run commands from this directory.
   
Smallest offline reproduction: `python run.py analyze && python audit.py`. This rebuilds results.json from the archived real logits and checks all four cells and the three-way split lock. No corpus or network required for analysis. For fresh inference:

```sh
python run.py fetch
python run.py register
python run.py smoke
python run.py formal
python run.py analyze
python audit.py
```

Use the same --cache and --archive arguments on fetch/register/smoke/formal to reuse verified downloads. Defaults are cache/ and aclImdb_v1.tar.gz. The model and source data identities are in resources.lock.json; the Stanford v1 tarball checksum in run.py is from TFDS. Stanford HTTPS failed during this run; fetch uses a content-identical pinned mirror. No review text or weights included. Labels for inference come from Stanford test neg/pos. The included pools.json contains identity digests only; rebuilding the registration must yield exactly the same file content. One movie and one normalized text per selected item; this changes the sampling distribution. Use the archived register output, not a fresh random split, in subsequent work.

Four CPU threads, eager FP32, evaluation mode, no prompt or sampling; right truncation, 512 tokens, dynamic padding to batch maximum. Source predictions and temperature are archived from the fixed source study linked in SOURCE_MAP.md. No new fit. Output files formal.json and smoke.json contain labels, token counts, logits and IDs; *_env.json contains batch shapes and timings. Replays replace derived files in the working copy. Threshold .90 is inclusive and exploratory. The original failed smoke used an expired temporary environment before loading any model; initial_smoke.log is preserved. No semantic relabeling. Use `python plot.py output.png` for the outcome figure.
