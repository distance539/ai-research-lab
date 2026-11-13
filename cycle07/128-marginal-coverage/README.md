# 九成覆盖保证，究竟保证了谁？

在本目录执行 `python run.py` 即可使用包内真实模型 logits 重建 200 次划分、逐条命中和汇总（Python 3.12 标准库）。再运行 `python audit.py`，用相反类与真类的 logit 差独立复核。splits.json 每行记录一个种子的两个名单、阈值、留出命中数和两池漏覆盖 ID（JSON 数组中的一项）；by_id.json 的值为 [留出次数, 命中次数]。results.json 保存两套分母与完整频数；lock.json 将分析与资源及源码摘要绑定。无提示词、解码或训练。

## 从公开模型复跑

CPU，建议至少 6 GiB 可用内存，权重下载约 268 MB。固定环境版本见 requirements-observed.txt；先安装依赖，在当前目录执行：
```sh
python -m pip install -r requirements-observed.txt
python model_run.py prepare --cache ./cache
python model_run.py smoke --cache ./cache
python model_run.py run --cache ./cache
python run.py
python audit.py
```
prepare 从固定版本准备原始资源；本次复用已校验缓存，重新分词并真实前向 8 和 360 条，仅原校准池 256 条使用既有 logits。分割和阈值完全限定在原校准池；历史审计池仅评分；不使用 IMDb。execution.json 是真实命令、耗时与退出码；run_environment.json 包含批次形状、设备、版本及进程峰值内存。只分发索引、标签、派生预测等，不分发原文和权重。图表可选：安装 requirements-plot.txt 后执行 python plot.py --output coverage.png。
