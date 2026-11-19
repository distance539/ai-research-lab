# Inference configuration audit (134)
CPU float32 eager vs SDPA, fixed IMDb exploration 1000, raw SST2 LAC threshold. No target calibration or confirmation predictions. Python 3.12.14, torch 2.6.0, transformers 4.49.0. Install `pip install -r requirements.txt`.

```
python common.py fetch --cache cache --archive aclImdb_v1.tar.gz
python run.py smoke --cache cache --archive aclImdb_v1.tar.gz
python run.py formal --cache cache --archive aclImdb_v1.tar.gz
python analyze.py
python audit.py
```

Offline arithmetic replay: `python analyze.py && python audit.py`. Real smoke needs about 268 MB weights plus IMDb archive (about 84 MB compressed) and dependencies, roughly 1.3 GB measured process RSS. Full model forward uses 1000 exploration rows per implementation, max_length512, batch16, 4 threads, seed134. Predictions/environment and execution.json are saved. Timings are descriptive, fixed order, no benchmark speedup claim. All resources checksum checked, pools copied byte-for-byte from pre-inference 132 registration. Never run common.py register to create alternate pools. Original review texts and weights not redistributed. No new semantic labels or human review. Data history: SST2 previously explored, threshold reuse is not fresh-test calibration. IMDb is publicly available, contamination cannot be excluded.

Observed: both 890 correct, 901 covered, sizes0/979/21; no paired flips or set changes. Tiny numerical differences are preserved. A current empirical rate of90.1% is not proof of exchangeability or future coverage. `audit.py` independently uses Decimal50 softmax. `SOURCE_MAP.md` documents official source and licenses. Local code is MIT. No training or quantization performed. No GPU backend claim.
