# Actual execution and boundaries — 2026-10-22

- Fresh real CPU inference: smoke5x10, full100x50. All5000full logits and ranks exactly match085 frozen reference; no cache substitution. Full elapsed133.407338916s including imports/resource hashing/load/token audit/evaluation, neural inference129.097431333s, peakRSS893075456bytes. CPUfloat32/fourthreads/batch16; Python3.12.14. No training, GPU memory, new retrieval cost or total machine energy measured.
- Firstshape input[2,512], last_hidden[2,512,384], logits[2,1], predict[2]; raw vs official score maxerror0. All3data and7model filehashes match. Runtime versions are requirements-observed.txt; environment/status/identity retain executed facts.
- Main nDCG10 RRF0.6709061301781298, CE0.6691325921143205, oracle0.9061314719276545.20wins/62ties/18losses. Classification9/8/16/20/47. Eligible91: untruncated75with17regressions; any relevanttruncated16with1regression. Association only. No human semantic labels, causal claim or futurecandidate result.
- Independent metric/oracle audit maxerror<=1e-12; independent discovery checker all100query/5000score rows pass;8mutation tests rejected. audit_execution.json records script hashes/commands/exitcodes/logs; independent_check.json records input identities.
- PROTOCOL/config/registry/implementation frozen in registration_snapshot before replay.085/086 results were already seen; this is not outcome-blind hypothesis discovery. Only candidate-preserving inference repeated.300confirmation not evaluated.

## Failed attempts retained

Old shared OneDrive environment import stalled and was interrupted; its installed packages were not changed. Available local runtime lackedSentenceTransformers. Initial restricted network dependency installation failed; authorized retry succeeded. Resolver briefly upgradedNumPy; before any inference, explicitly restoredNumPy1.26.4/SciPy1.13.1/scikit-learn1.6.1. Full observedversions recorded.

Firstrun omitted--smoke despite smoke output name; interrupted exit130 duringPILimport, before any data/model inference. early_aborted_status.json is its unmodified internal state with exit_codenull/stagesempty; wrapper exit130 is separately recorded in execution_history.json. Do not reinterpret null as success. Then correctsmoke andfull finishedexit0. Anonymous officialsource fetch initially hit network sandbox denial; authorized read succeeded. No scientific variants were tried beyond declaredrun.

Both docs and extra audit tests added after inference are clearly separate from frozen experimentalcode. No model outputs modified. Data/model cache is external and hashverified; not distributed. All needed project modules included; script paths are relative to package. No credentials/privateconfiguration/weights/venv/fullcorpus distributed.
