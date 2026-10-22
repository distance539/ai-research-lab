# 106 从失败样本到研究问题：怎样避免事后编故事？

SciFact100条已观察开发查询×50冻结候选，真实ms-marco-MiniLM-L6-v2交叉编码器重新推理，建立可复算错误分类及106–111单一检索研究登记。不是完整benchmark、语义人工金标、改进方法效果验证或独立新测试。300条确认查询留给111。

## 最小运行

Python3.12，CPU建议4GiB可用内存。模型约91MB、数据ZIP约2.8MB；PyTorch安装另计。固定关键依赖见requirements.txt，全量实测版本见requirements-observed.txt。

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out replay-smoke --smoke
python audit.py --out replay-smoke
python discover.py --out replay-smoke
python check_discovery.py --out replay-smoke
```

真实入口`run.py --cache ./cache --out replay-smoke --smoke`执行5查询×10候选，不冒充完整分类分布。删除`--smoke`并使用新输出目录运行100×50；随后执行同一审计链。输出目录必须不存在。已校验缓存可显式传入，不能依赖作者机器路径。初次prepare联网，后续可离线。

只检查已保存完整结果：`python audit.py --out results`、`python check_discovery.py --out results`；`python test_audit.py --out results`注入八种损坏并验证拒绝。这些标准库审计不等价于真实推理。

## 产物和冻结边界

- `results/runs.json`、`pair_scores.csv`：全部真实分数、相关标记、token长度和排名。
- `results/taxonomy.json`、`discovery.json`、`examples.json`：五个互斥分类、截断关联表、按哈希选出的类别示例；不含人工语义诊断。
- `results/status.json`、`environment.json`、`shape_check.json`、`identity.json`：阶段、耗时、峰值进程RSS、形状与完整缓存身份。
- `research_registry.json`、`PROTOCOL.md`、`registration_snapshot.json`：已见结果启发的假设、未来实验、主要指标、资源与停止规则；为本地登记，不宣称第三方预注册。

继承085固定公开提交的100×50候选和qrels，保留inputs及哈希锁；官方Rerank文件原样复制，其他模块为本项目独立实现或从085改写。SOURCE_MAP.md列永久源码、许可证与改动。未复制模型权重或完整语料，prepare下载固定revision并核对hash。SciFact标注引用Wadden等，保留CC BY4.0与ODC-By1.0说明。

## 解释范围

前面809条开发查询全部已有观察历史，这100条不能重新命名为测试集。候选缺失、前十未命中和分数退步是可计算状态；截断是输入属性。二者的关联不是因果，也不证明模型不懂科学术语。107的文本变换必须经真实人员确认标签保持；本包没有替代该审核。

未来主假设是固定50候选、一次CE计分下，以常数60、等权倒数排名融合保留RRF先验，确认集nDCG@10相对纯CE提升至少0.01，同时报告相对纯RRF差值。106未运行此改进。所有后续实验必须遵守registry，未胜出也保留结果。

详见VALIDATION.md的真实执行范围、失败历史和复核结果。
