# 少回答一些，真的能少犯错吗？

从本目录执行以下命令（Python 3.12.14，分析只用标准库）。

    python run.py && python audit.py

输入是已经固定的 DistilBERT 在 SST2 历史协议审计池上的 360 行真实输出。生成 curve.json（两个全前缀、排序后的全部 ID、均匀随机期望）和 thresholds.json；audit.py 独立重建 722 个点和 4 个阈值。主读数 270 条：按最大概率有 8 错，固定编号打乱有 18 错。空接受集风险 null。随机打乱规则与主要读数见 PROTOCOL.md，不在曲线上挑阈值。复跑会覆写派生结果，请在副本中执行。

## 从原始模型复跑
在 Python 3.12 环境内执行：

    python -m pip install -r requirements-observed.txt
    python model_run.py prepare --cache .cache
    python model_run.py smoke --cache .cache
    python model_run.py run --cache .cache
    python run.py && python audit.py
    python plot.py --output risk_curve.png

prepare 下载固定版本的模型、词表与原始验证文件并对照 resources.lock.json，已校验的缓存可直接用于 smoke/run。8 条和 360 条都是同一审计池的真实前向；smoke 8 属于 360，不能相加为独立样本。run_predictions.json 是本次逐条重放；audit_predictions.json 是归档输入；两者逐项核对 logits、标签、身份与类别。analysis.log/audit.log 记录指标与独立检查，execution.json 记录命令、时间和退出码。VALIDATED_CACHE 代表上面命令中的 .cache。模型权重约 268 MB，默认 CPU float32，4 线程，限时 30 分钟、峰值 RSS 6 GiB。run_environment.json 记录实际系统、依赖、耗时、峰值 RSS 和形状。原始语句、权重、缓存不分发。每个推理批都运行相同的分类头，无 prompt 或解码步骤。lock.json 的资源锁、split、模型代码与配置哈希共同标识计算；分析另外锁定预测和脚本。未使用 IMDb，SST2 先前已经使用，本结果属于域内开发证据。
