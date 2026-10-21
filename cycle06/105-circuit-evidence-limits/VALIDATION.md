# Validation evidence — 2026-10-21

- plan.json frozen before smoke and formal inference; immutable scientific file hashes checked on entry.
- Official fixed sources freshly fetched, GPT2 installed bytes equal, four parent files anonymously equal. First sandbox proxy failure recovered with authorized anonymous network; no TLS weakening or credential extraction.
- All25dependency versions matched. Every run verifies eight model/tokenizer resource hashes. Public-template256sample AST rebuild exact.
- One-row smoke exit0,4.5756s process wall,RSS1466449920B,projected192.405s. Formal exit0,230.8982s wall,RSS1465073664B;7936scored,1024controls,24refs. Separate stage records preserved.
- Empty and all144END selfpatch exactly0error for256samples. Pre-vs-postprojection retained control max full-logit error0.0002212524<0.001.
- audit.py independently recalculates all8448saved score cells (7936unique scored conditions), all8subset/group/mode statistics, means/MAE/counts/selection and identity. Full and smoke audits passed. This is deterministic arithmetic, NOT human review.
- test_checks.py rejects8deliberate corruptions (score,identity,missingK,duplicate,rowargmax range,selfcontrol,NaN,aggregateMAE).
- replay.py re-executes ALL7936conditions with independent dense postprojection patches; every vocabulary argmax identical. Maximum candidate logit error0.0001831055;IOprob error0.00000882149; independently recomputed reference means float64→float32 error0.0000004768372. Numerical agreement, not bitwise claim.213.160s process wall.
- Parent104 clean predictions and ABC-zero predictions checked byte-for-value equality for all256rows; parent sample generation and resources unchanged.
- Reproduction needs pinned public model files; extraction validation reuses existing hash-checked cache, not implicit author modules. Cross-device bitwise identity unverified. No new heldout data, no human labels needed for deterministic synthetic identities.

Commands and exit codes: results/execution.json,results/replay_execution.json,independent_checks.json; outputs in results/arithmetic_audit.json and results/independent_model_replay.json. Package/public verification live alongside the article to avoid recursive self-hashing.
