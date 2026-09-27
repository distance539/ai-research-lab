# Validation 2026-09-27

809 development queries; 8 runs; 136650 candidate contribution rows. run.py exit 0; 4.859690666664392 seconds; peak RSS 396066816 bytes. Independent audit checked 310733 scores and all metrics. Seven rational/float rank differences left all three metrics unchanged. Original inputs unchanged; duplicate BM25 preserves ranks.

A five-query real smoke preceded full runs. The final rerun after adding code identity and expanded auditing reproduced all metrics. One attempt resolved the venv executable symlink to the base interpreter and failed for missing pytrec_eval. Using the venv entry fixed this; missing-package environment reporting was also improved. Raw failure logs remain locally in verification/wrong_interpreter because they contain machine paths.

No inference/retrieval rerun or 300-query confirmation evaluation occurred. development.log, audit.log, diagnostics.log and results contain distributable raw records. plot.py consumes saved summary/per-query data. Subsequent archive and public-commit verification belongs to the article verification directory.
