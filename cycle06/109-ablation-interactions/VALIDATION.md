# Actual validation — 2026-10-25

Smoke16pairs and formal400pairs completedCPUfp32. Formalmeans00=.9077324384,10=.7654015779,01=1,11=.9077324384. Meaninteraction.0500632989; query30positive.3690702464/query155negative-.1688170509,other2zero. MeanAeffectsnegative at both Blevels; no synergistic-success claim. Eachtextcondition75683nonpaddingtokens; unchangednegativeinputs/scoresexact; all400scoresexact107original/frontafterfreshinference. Recall50all1 byselection.

Formal inference10.558777960s,total13.222144541s,peakRSS775897088B; shape[2,512]→[2,512,384]→[2,1],manual/predictmaxerror0. No training/confirmation. Independentstdlibaudits reconstruct allmetrics and reciprocalfractions; seven corruptiontests rejected in smoke/formal. Newaudits do not replace actual107semanticapproval.

Firstauditcommand ran in wrongworkingdirectory(exit2,noauditexecuted); correctedcommand passed, originalfailurearchived locally. Initialanonymoussourcefetch failed restrictednetwork; approvedanonymousretry returned200for13sources and confirmedexactofficialsource. No experimentparameters changed.

ZIP extraction and anonymouspubliccopy validation are recorded externally by articlepackage_validation.json/publication_verification.json, to avoid self-referentialpackagehashes. Do not inferpublication from thisfile alone.
