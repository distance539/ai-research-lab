# 108 Equal-budget baselines on SciFact

100条已观察开发查询×50冻结候选，真实MiniLM交叉编码器一次计分5000对；比较pureRRF、pureCE、constant60等权倒数排名融合。仅开发诊断；不训练，不用300确认查询，不宣称新算法或完整benchmark。纯RRF无需CE；CE和融合共享同一模型预算，不能称三方案总成本相同。

## 最小运行

Python3.12，建议4GiB空闲内存。模型约91MB、数据ZIP约2.8MB，框架安装另计。关键版本requirements.txt，全依赖固定requirements-observed.txt。

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-observed.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out replay-smoke --smoke
python ce_audit.py --out replay-smoke
python audit.py --out replay-smoke
python test_audit.py --out replay-smoke
```

smoke为5查询×10候选50对，不代表正式100×50指标；去掉--smoke并使用新的--out目录运行全量。输出目录必须不存在。prepare会校验所有模型/分词器/已用数据文件哈希；缓存显式传入，可复用已校验资源。输入来自固定公开父提交，已随包携带必要自写模块，无绝对路径依赖。已有结果可离线执行后3条审计命令。

## 产物

- PROTOCOL.md、research_registry.json、registration_snapshot.json：预定方案、主要指标、预算、停止规则与既有观察历史。
- results/runs.json、pair_scores.csv：原始排名、真实CE和仅评测用oracle；不把oracle当部署基线。
- blend_runs.json、fusion_trace.json：每对RRF/CE名次与融合分数；comparison_per_query.json、comparison_summary.json为全部配对结果。
- budget_ledger.json：候选数、有效token、CE调用数、单独融合算术耗时。没有重新计量初始检索成本。
- identity/environment/status/shape_check/token_audit.json：缓存键、锁定版本、运行环境、阶段状态、资源和实际形状。有效token不含padding，不等于FLOPs。
- independent_audit.json、mutation_tests.json：独立复算和7损坏反例；原始pytrec_eval审计见audit.json。

run.py调用未改变的ce_run.py官方推理路径后做排名融合；status.json记基础推理/评测阶段，整个命令退出码另见execution_history.json。单次模型计分结果在本次CE/融合条件中共享，不读取此前分数冒充新推理。额外4pair形状核对单列，不计入5000正式评分。

SOURCE_MAP.md给出固定上游文件、函数行号、许可证及自写部分。许可证和归属在LICENSE、NOTICE.md及第三方说明。大模型、完整语料、缓存、环境和密钥不分发。确认集仍未评测。VALIDATION.md记录实际结果和失败。
