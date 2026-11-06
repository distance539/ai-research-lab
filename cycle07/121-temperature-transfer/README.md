# Temperature transfer across scoring interfaces

From this directory run `python run.py && python audit.py` (Python 3.12 tested, standard library). This rebuilds result.json from 1,232 new SmolLM2 prompt scores and archived classifier scores, then independently checks 12 cells, hashes, and smoke/formal agreement. Expected temperatures C/A/B approximately 2.482/17.452/16.231. Positive prediction counts on audit: 192/0/0. See protocol.json for frozen template text, label order, seeds, decoding and design.

To reproduce the neural network stage in a separate clean copy:

    python -m pip install -r requirements.txt
    python prepare.py --cache .cache
    python score.py --model-dir .cache/model --data .cache/validation.parquet --smoke > smoke.log
    python score.py --model-dir .cache/model --data .cache/validation.parquet > formal.log
    python run.py && python audit.py
    python plot.py --output transfer_nll.png

Keep formal.log with formal.json; audit.py verifies the output digest in the last log line. Relative paths assume the current directory is this code folder. prepare downloads pinned files (~270MB weights); CPU/FP32/4 threads, budget 30min, allocate several GiB. Actual model time/peak process RSS in formal.log. No API, training, new semantic labels or IMDb access. Both model score streams use the same 256 calibration and 360 previously used audit identities. C inputs inherited byte-for-byte from the [fixed 117 package](https://github.com/distance539/ai-research-lab/tree/f33e57cca855c0d2c6da5aed4753de76db0e7a89/cycle07/117-temperature-scaling); its README describes reproducing classifier inference. The inherited 616 classifier forward passes were not repeated today. No review text or weights redistributed. Scoring SmolLM2 uses a full-vocabulary forward but conditions only on the two predeclared candidates. See SOURCE_MAP.md for versions and licensing.

Cache identity includes the protocol hash (template, resources, data and split identities, decoding, labels) plus score.py hash, both emitted in logs. Fresh scores computed separately by interface and split; no cross-template memoization. Reuse of formal outputs is checked by audit.py. Arithmetic checks automated; no human semantic review claimed.
