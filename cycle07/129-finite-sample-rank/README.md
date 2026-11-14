# 校准样本不够时，预测集为什么突然变大？

在本目录执行 `python run.py`，以包内已实测的预测重建 results.json（两规则、十一种规模）、memberships.json（每条身份与返回标签）及 lock.json。随后执行 `python audit.py`，使用归档原始 logits 独立复算。标准库即可；校准使用 split_manifest.json 的原顺序。q=null 表示正无穷。空集合是合法测量结果。

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
