# 总体覆盖达标，子群体也安全吗？

在源码目录执行 `python run.py`，以已归档的真实模型预测重新计算一个全局阈值、十二行分组计数和逐句标签集合。接着执行 `python audit.py`，用原始分数差核对每个标签成员及计数。两步仅使用标准库。配置见 protocol.json，长度计入特殊符号，排除填充。结果中 n、hits、doubles、empty 分别指句数、覆盖数、双标签数和空集数。

## 从模型重跑

CPU、约 268 MB 权重、建议至少 6 GiB 可用内存。Python 3.12，完整环境见 requirements-observed.txt。在当前目录运行以下命令。

```sh
python -m pip install -r requirements-observed.txt
python model_run.py prepare --cache ./cache
python model_run.py smoke --cache ./cache
python model_run.py run --cache ./cache
python run.py
python audit.py
```

本次复用通过 SHA-256 校验的公开资源和已有 256 条校准 logits，重新分词，并对 8 条试跑及 360 条历史审计句子执行前向。execution.json、四份日志、两个 environment.json 与预测文件是执行证据。控制条件记录于 protocol.json。该数据曾用于系列开发。未读取 IMDb。算法与代码的行号、来源和许可证见 SOURCE_MAP.md。包内只有身份、派生分数、配置与代码，不含权重及评论原文。
