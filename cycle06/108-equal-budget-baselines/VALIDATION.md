# 108 validation —2026-10-24

Real CPU smoke5x10 then formal100x50 with official CrossEncoder/Rerank.5000fresh scores exact match106; no training/no confirmation evaluation. RRFndcg0.6709061301781298,CE0.6691325921143205,blend0.702520589195864;blend-minusCE0.0333879970815435,20/66/14win/tie/loss;blend-minusRRF0.03161445901773418,18/72/10.Recall50all0.905;top10miss21/17/14.18oldCEregressions:17improved,9recoveredtoRRF;13of20oldCEgainsreduced;1newharmvsRRF. No confirmatory success claim.

FormalCE129.8994534s,basewall134.7155337s,peakRSS855687168B. Fusion2.955375ms (one measurement, includes traces, excludes evaluation/writes). Same5000pairs/1761679nonpaddingtokensforCEandblend,extraCE0/training0. RRFzeroadditionalCE: totalcostnotmatched. Shape[2,512]/[2,512,384]/[2,1],manualscoreerror0,4auxiliarypairforwards outside primary5000. Modelweights/tokenizer/datafiles hashchecked before every run; all observed dependency versions matched.

Independent stdlib reconstitution of5000pairs/ranks/metrics/budgets/trace/means/counts/tradeoffs passed; exactFraction score comparisonmax6.94e-18,70duplicate fusion scores handled by specified float/stringID rule. Originalmetrics auditmax0;7damage tests rejected. Logs and input identities saved. Parent sample reused and already observed; dependence and selection bias remain.

Firstsmoke failed before inference: missingtorch in oldtemporaryenv. Rebuilt exact pinned environment; initial restricted PyPI failed, authorized download succeeded. No result or failedlog erased; outsidepackage fulltrace records in articleverification_logs. Modelcard webtool unavailable; anonymousfixedrawdownload200succeeded. No generated numeric plots; methodcharticons are illustrative only.

Archive/publication checks are separate article records. This file documents experiment completion; do not infer remote availability until published fixedcommit is verified.
