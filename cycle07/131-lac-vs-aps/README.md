# LAC 与确定性 APS 的配对对照

Python 3.12，在干净虚拟环境安装 requirements-observed.txt 所列版本。最小入口为 `python run.py`，使用随包的真实校准和审计 logits，生成 results.json、memberships.json。运行 `python audit.py` 可独立核对原始计数并执行末项精确置一的诊断，输出应与 audit.log 一致。CPU 双精度 softmax 的末位可能受平台影响，对比时同时检查平台与端点直方图。

从原始 SST-2、DistilBERT 重跑需约 268 MB 权重及最多 6 GiB 内存。
```sh
python -m pip install -r requirements-observed.txt
python model_run.py prepare --cache ./cache
python model_run.py smoke --cache ./cache
python model_run.py run --cache ./cache
python run.py
python audit.py
```
校准输入为归档的固定256条，推理阶段重新分词并前向8条及360条。正式运行 4 条命令和退出码在 execution.json。首次开发运行在审计脚本写入前调用该文件，退出码2；补齐脚本后完整重跑，详见首次和最终执行记录。来源映射与分发许可见 SOURCE_MAP.md。只含公开样本的身份及派生测量，不分发文本或权重。协议见 protocol.json；输入摘要涵盖模型、分词器、数据、推理配置、无模板无解码的二分类程序和后处理路径。归档预测是固定输入，不使用未经校验的外部推理缓存。
