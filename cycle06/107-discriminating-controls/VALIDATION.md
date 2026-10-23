# Validation 2026-10-23

Technical execution passed; semantic acceptance pending. No person has approved the8 transforms. Human review forms remain null. No article quality passed claim.

- Smoke32pairs,0.667520s inference/3.782162s total,peakRSS586137600bytes. Estimated800pair inference16.688s below1200s.
- Formal800pairs,20.856352s inference/24.600020s total,peakRSS786579456bytes onCPUfloat32. All4conditions have75683retained tokens; max512; batch16/4threads/seed107. Shapes[2,512],[2,512,384],[2,1], direct logits vs officialpredict error0.
- Original200scores equal106 stored scores exactly. Identity200scores equal original exactly. Target final ranks original/front/tail:454=1/1/1,30=1/2/1,280=1/1/1,155=2/4/2. All unchanged documents have identical tokens/scores. Recall50unchanged1for all4.
- Longfront-minus-tail nDCG mean-0.184535;short-0.100127;difference-0.084409. These use original qrels provisionally. No confidence interval or causal effect claim. No neural training, no confirmation use.
- Independent stdlib audit of all800pairs, ranks, binarynDCG, token totals, permutations and candidate sets passed.7mutations rejected. Arithmetic is automated, not human semantic review.
- Pre-inference material build initially failed exact BEIRtext equality because original sentence trailing whitespace; per-sentence strip matches BEIR5183/5183documents. Corrected before selection freeze and any107forward. No score-driven amendments. Initial log retained outside distribution.
- Network sandbox blocked anonymous source reads; bounded retry used. Upstream evidence recorded separately. No credentials or browser session extraction.

Package validation and anonymous publication checks are separate article records; do not infer completion from this file. Semantic gate remains pending even if public technical package is verified.
