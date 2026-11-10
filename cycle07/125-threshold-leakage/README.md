# SST-2 阈值选择与审计泄漏
在本目录运行 `python run.py`，仅用 Python 标准库即可从已锁定的真实模型输出重算 selected.json 与 results.json。随后运行 `python audit.py` 独立从 logits 和官方标签核对全部候选。两池分别含 256 与 360 个不同的身份。阈值规则见 PROTOCOL.md。可单独运行 `python run.py select`，这个阶段无需读取审计预测。校准选择 0.99；审计 14/304 错；同池选择 0.98，16/321 错。
## 重放模型
使用 Python 3.12，在隔离环境中依次执行：
```sh
python -m pip install -r requirements.txt
python model_run.py prepare --cache ./cache
python run.py select
python model_run.py smoke --cache ./cache
python model_run.py run --cache ./cache
python run.py
python audit.py
```
首次准备下载固定权重和 validation parquet。已备齐可跳过 prepare；推理逐文件检查哈希。真实校准预测复用前篇，审计重放先8再360句。run_predictions.json 保存新预测，入口分析锁定历史快照；两份的身份、输出类别和 logits 在正式批次完全一致。原始句子与权重不附带。CPU、批量16、4线程；30分钟和6GiB预算。资源身份与锁见 resources.lock.json 和 lock.json；分类任务无提示模板、无解码。execution.json 列运行命令（VALIDATED_CACHE 表示自选缓存目录）、退出码和耗时。计量见 run_environment.json；复算日志见 audit.log。无新语义标签。本目录自写代码遵循 MIT；框架、数据、权重的出处分别见 SOURCE_MAP.md。
