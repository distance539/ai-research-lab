# SST-2 全局温度与拒答排序
在本目录执行 `python run.py`。标准库从随附的两列真实 logits 生成 paired.json（逐样本分数、次序及每个接受预算的风险）与 results.json。执行 `python audit.py` 用高精度二分类恒等式独立验证。预期两种温度下 360 个位置均相同，固定接受 270 条均错 8 条。0.99 门槛在缩放后无人达标，风险为 null。
## 从权重重放
在 Python 3.12 环境中依次运行：
```sh
python -m pip install -r requirements.txt
python model_run.py prepare --cache ./cache
python model_run.py smoke --cache ./cache
python model_run.py run --cache ./cache
python run.py
python audit.py
python plot.py curve.png
```
prepare 下载固定版本，已有缓存按 resources.lock.json 逐件核对。冻结 SST-2 validation，沿用历史校准池 256 条的拟合温度，审计 360 条正式重放与锁定输入逐值相同。run.py 读取冻结审计预测，两个模型入口分别保存 smoke_predictions.json 和 run_predictions.json，核验分类头的输出。CPU 四线程、批量16、最多512 token、6 GiB 与30分钟预算。无模板、无解码、无训练。分词器与模型共享 revision。lock.json 绑定配置、代码、参数与输入，execution.json 保存命令、退出码及日志名，环境记录含计时、形状和进程峰值内存。源码地图区分本地实现与上游；不分发权重或原句。
