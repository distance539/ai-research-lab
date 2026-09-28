# 085 Reranker 的上限在哪里：召回失败还是排序失败？

范围：真实 SciFact 全5183篇语料检索所得的冻结 RRF候选，809条开发查询中哈希选定100条，每条50候选，共5000对。真实 ms-marco-MiniLM-L6-v2 交叉编码器 CPU 推理；不是809查询完整benchmark，不是事实核验，不是原BERT论文训练复现。300确认查询未评测。

## 运行

Python3.12（实测3.12.14），CPU约1GB进程内存；建议空闲4GB，依赖安装另需磁盘。模型约92MB，数据ZIP约2.8MB，安装PyTorch另计。首次联网安装及下载；后续可离线复用hash匹配缓存。

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out results/replay
python audit.py --out results/replay
MPLCONFIGDIR=./cache/matplotlib python plot.py --out results/replay --dest results/replay.png
```

最小真实入口在run.py命令末尾加 `--smoke`，执行5查询×10候选。正式命令执行100×50。仅需重放结果审计可直接 `python audit.py --out results`，只用标准库，无需下载；它不替代真实推理。入口参数不依赖作者绝对路径。cache可显式指向既有已校验资源根目录，需包含scifact/及models/<revision>/；prepare会核验。

## 协议与产物

PROTOCOL.md在推理前写入，config.json记录全部查询ID/seed85/CPU4线程/batch16/max512。输入继承084公开固定commit，input.lock.json记录来源与哈希。model.lock.json逐文件固定模型/tokenizer版本；resources.lock.json沿用082数据资源锁，含历史资源但本篇只消费SciFact三个文件；prepare不下载ES或BM25索引。

results/保存runs.json、5000行pair_scores.csv、逐查询指标、summary、token/shape/环境/分阶段状态和identity缓存身份。所有指标以pytrec_eval计算，MRR@10独立实现。audit.py重算指标、候选oracle闭式公式、损失恒等式和每对分数排名。Oracle只用于诊断，从不喂给模型。run.py每次真实推理，不静默读取旧分数缓存。

本次实际结果：nDCG@10原RRF0.67090613，重排0.66913259，oracle0.90613147；20胜62平18负。Recall@50均0.905。9条无候选相关文档，8条有相关候选但top10未命中，83条top10命中。原始日志development.log与smoke.log；没有训练、GPU、多模型选择或人工错误标注。

配对候选受控，但神经模型新增了计算；原检索耗时不在135.33秒重排耗时中。部分文本截断，模型预训练污染未审计。样本是开发子集，未估计总体显著性。

参阅[SOURCE_MAP.md](SOURCE_MAP.md)、[NOTICE.md](NOTICE.md)、[PROTOCOL.md](PROTOCOL.md)。本目录是可浏览源码交付；博客本地code.zip仅为归档。
