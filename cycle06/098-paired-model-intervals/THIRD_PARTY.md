# Distribution and attribution

Original teaching scripts in this package are MIT licensed (LICENSE.txt). common.py/scoring.py/prepare.py originate from distance539/ai-research-lab article093, fixed source identity in parent_source.json; attribution retained. They are not official SmolLM training code.

Official dependencies: Transformers4.49.0 and SmolLM2-135M-Instruct Apache2.0; PyTorch2.6.0 BSD-style; SciPy1.13.1 BSD3. Full applicable upstream licenses are included; official executable implementation files are not copied. NumPy and PyArrow are installed as dependencies under their upstream licenses.

SST2 dataset card does not establish a redistribution license. Do not redistribute raw parquet, source reviews, full prompts or input token IDs. Included128 generation records are short JSON sentiment labels, source indices, prompt hashes and derived scores; no review text is included. No human semantic annotations or numerical human error rates are claimed. Checkpoints, model weights, cache, environment, credentials and private paths are excluded. Reader download scripts use pinned public resources and verify SHA256.
