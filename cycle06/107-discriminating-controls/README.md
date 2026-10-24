# 107 为直觉找反例：设计最小的区分性实验

4条已观察SciFact开发查询×50冻结候选×4条件，共800次真实MiniLM交叉编码计分。原文、原样重建、证据句前移、证据句后移。利用原始rationale选择目标句，仅作标签辅助诊断，不是可部署方法或完整benchmark。**2026-10-24 用户确认八项文本变换全部审核通过，原文及材料哈希见 human_review_confirmation.json。原始运行记录中的 pending 为确认前历史状态。**

## 运行

Python3.12，CPU建议4GiB空闲内存；模型约91MB，两个数据包约6MB，框架安装另计。固定版本requirements.txt，已观测全量版本requirements-observed.txt。

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out replay-smoke --smoke
python audit.py --out replay-smoke
```

最小入口覆盖相同4查询，每条前2候选加相关文档去重，本次实际为2个/查询，共32次计分。删除`--smoke`、指定新输出目录运行完整800次。每个输出目录必须不存在。缓存可显式传入并逐文件校验，无作者机器隐含路径。prepare只从原始归档提取语料和训练标注，不读取原始dev/test标注或BEIRtest qrels；BEIRZIP保留原完整结构但不使用确认标签。

离线检查已保存结果：`python audit.py --out results`；`python test_audit.py --out results`验证7种损坏被拒绝。这是算术复算，不是人工审核。

## 产物和协议

- materials.json：4个公开训练样例、原证据句索引、四种句序、token可见性、预分数选择规则和全部合格ID。数据按SCIFACT-LICENSE.md署名；不是新人工金标。
- PROTOCOL.md / registration_snapshot.json：推理前冻结的假设、控制、主要比较和资源停止规则；research_registry.json原样继承106。
- results/pairs.json与runs.json：完整800分数、输入哈希、token数量；per_query.json和summary.json逐查询与配对结果。
- results/identity.json、status.json、environment.json、shape_check.json：完整版本身份、命令、退出码、阶段和资源测量。
- HUMAN_REVIEW.html：原文和8变换的无模型分数审核页；HUMAN_REVIEW_RESPONSE.json空白模板。没有真实回复前，不能将null改为通过。
- build_materials.py可从固定缓存重建选择和材料。会重新生成时间戳与审核模板；请在独立副本运行，保留既有真人修改。不要在已有审批包上覆盖运行。

SOURCE_MAP.md给出官方固定实现、许可证和本地对应。完整原始语料/权重/环境不打包。共4篇有归属的公开摘要和4条原始训练claims用于小型可检查样例。新代码Apache2.0，官方BEIR源码原样及许可保留。

## 解释边界

两长两短非随机匹配；前后移同时改变位置、篇章与截断上下文。只有长组改变证据完整可见性，短组校准无截断的顺序敏感性；不能把组差当独立因果效应。候选数和有效token总量相同；各条件完整重新计分。300确认查询仍未评测，108的融合试验尚未运行。

VALIDATION.md记录实测范围、失败与后续真人确认。

## 2026-10-24 真人审核确认

用户直接回复“我审核完了,都没问题”，覆盖4查询各front/tail的8项变换合法性与完整文档相关性标签保持。未提供逐项理由，故不代填理由或虚构审查人姓名。HUMAN_REVIEW_RESPONSE.json保留原始空白模板；materials/config/results中的pending和audit.py对历史日志的约束保持原样。以单独的human_review_confirmation.json记录后续确认，不改写历史实验。审核页不含分数，但正文和聊天已披露结果，因此不声称独立盲审。实验数值、负结果和小样本限制均不变。
