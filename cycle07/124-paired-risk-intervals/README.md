# 错误只有少数几个时，拒答收益有多可靠？
Python 3.12.14。用 pip install numpy==1.26.4 安装最小依赖，随后在本目录执行：

    python run.py && python audit.py

points.json 含全部361个接受数、两臂错误数、收益和逐点区间。primary_counts.json 是主比较4000次配对差值的分子。audit_predictions.json 是实际模型在历史SST2审计池360句上的冻结输出。独立审计从原始 logits 重建错误与排序，按抽中次数展开独立复算。运行将覆盖同名派生产物。

完整模型重放（CPU，固定权重约268 MB）：

    python -m pip install -r requirements-observed.txt
    python model_run.py prepare --cache .cache
    python model_run.py smoke --cache .cache
    python model_run.py run --cache .cache
    python run.py && python audit.py
    python plot.py --output gain_intervals.png

本次复用已下载且重新校验的模型与数据文件，实际执行8句和360句推理。执行记录及 stdout、耗时和内存见 execution.json 与 *_environment.json。VALIDATED_CACHE 表示可替换的缓存路径。下载、分词、批次、设备与模型锁见 config.json、resources.lock.json、split_manifest.json 和 lock.json。数据无文本分发；原模型与数据从固定官方版本取得。原始论文、原理对应和各部分许可证见 SOURCE_MAP.md。协议在产生区间之前固定，既往点估计已知。
