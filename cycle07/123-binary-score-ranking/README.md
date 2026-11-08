# 最大概率、熵和分数差，谁更会挑出错误？

在本目录的副本运行（参考环境 Python 3.12.14）：

    python run.py && python audit.py

两步只用标准库，生成 ranking.json 并从原始固定 logits 独立复算 1080 个分数、三个完整身份序列和 1083 个前缀。输入 audit_predictions.json 为历史 SST2 验证划分中预登记的 360 个协议审计样本的真实模型输出。输出含逐行分数、完整排序、每个接受数的错误数和条件风险。预注册文件 PROTOCOL.md，输入及代码哈希见 lock.json。只使用原标签。

复现官方模型与绘图（约 268 MB 模型权重，CPU 四线程）：

    python -m pip install -r requirements-observed.txt
    python model_run.py prepare --cache .cache
    python model_run.py smoke --cache .cache
    python model_run.py run --cache .cache
    python run.py && python audit.py
    python plot.py --output prefix_errors.png

缓存目录可指定为任意位置。下载阶段验证官方 revision 与文件哈希，重放时再次检查资源锁；无生成提示或解码。正式重放与输入逐行核对。代码、输入与固定配置身份见 lock.json。execution.json 内 VALIDATED_CACHE 指代已校验缓存。默认写入当前源码副本，重放覆盖对应派生文件。run_environment.json 给出设备、版本、耗时、内存与形状。未训练、未优化阈值；未载入 IMDb。数据卡未明确授权原文分发，归档不含权重、原始语句。详见 SOURCE_MAP.md。
