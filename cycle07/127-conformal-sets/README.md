# 二分类分割保形预测集

在本目录执行 `python run.py`。Python 标准库即可使用包内真实校准与审计 logits，写出 threshold.json、sets.json、results.json、lock.json；用 `python audit.py` 独立复算 720 个候选判断。输入是历史 SST-2 validation，256 条校准和 360 条协议审计，名单互斥。protocol.json 锁定唯一名义覆盖率 90%，对照为同一分类头的最大分单元素集合。

## 从模型权重开始

需要 Python 3.12、CPU 和约 6 GiB 可用内存。新建虚拟环境后在本目录执行下列命令。模型文件约 268 MB。缓存目录由读者指定。

```sh
python -m pip install -r requirements-observed.txt
python model_run.py prepare --cache ./cache
python model_run.py smoke --cache ./cache
python model_run.py run --cache ./cache
python run.py
python audit.py
```

本次使用已校验下载缓存。模型程序逐文件验证资源哈希，读取原始文本后重新分词。先试跑八条，再对全部审计样本执行前向。校准使用随包既有的 256 条真实预测；未训练，未重跑校准池，未访问 IMDb。阶段日志、退出码见 execution.json；形状、峰值进程内存及运行时间见 run_environment.json。lock.json 绑定协议、配置、版本、输入及执行代码；没有模板或解码。SST-2 许可在数据卡标为 unknown，仓库仅含派生分数。详见 SOURCE_MAP.md。
