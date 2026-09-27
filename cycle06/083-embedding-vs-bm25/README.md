# 083 MiniLM 与 BM25：SciFact逐查询比较

完整5183文档、809开发查询；冻结all-MiniLM-L6-v2，官方SentenceTransformer+BEIR精确检索。BEIR test未检索；无微调、融合或人工金标。

## 安装与资源

实测Python3.12.14、macOS26.6 arm64、CPU；完整固定版本见requirements.txt（包括本环境继承的BM25依赖）。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
```

prepare获取固定BEIR提交、哈希锁定SciFact数据、固定revision模型/tokenizer，并校验所有文件；可重复使用已有已验证缓存。权重约91MB，语料ZIP约2.8MB；环境另需磁盘空间，建议2GiB可用内存及3GiB磁盘。不同系统依赖安装/性能待人工核验。无API密钥。旧BM25结果随baseline提供，重跑其官方链路参见SOURCE_MAP链接082固定版本；本篇不需要启动ES。

## 最小真实运行与完整复现

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out results/my_smoke --smoke
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out results/my_run
python audit.py --run results/my_run --embedding-audit
python precision_audit.py --run results/my_run
MPLCONFIGDIR=./mpl-cache python plot.py --input results/my_run/paired.csv --out paired_deltas.png
```

输出目录必须不存在，防止覆盖旧运行。smoke是5查询×16文档，标签仍来自真实数据，分数无基准含义。完整入口重新编码全部语料/开发查询，输出results.json、run.trec、per_query.json、summary.json、身份与形状、逐文档token长度、状态和环境；下载由prepare独立完成。

配置、选择规则与种子见config.json和PROTOCOL.md。缓存身份包含模型/tokenizer、prompt、数据、解码、源码及依赖；本入口从不偷用缓存向量。向量.npy运行时生成，归档不包含向量缓存、模型、语料或环境。要复核全库浮点分数先运行完整入口；已保存的分数与指标可完全离线审计：

```bash
python audit.py --run results/development --out /tmp/083-offline-audit
```

该离线审计只用标准库，支持Python3.9以上；真实模型入口需本固定依赖。prepare要求Python3.12安全解压。所有自写模块随包，路径均由读者指定，不依赖作者绝对目录。

## 真实结果与成本

MiniLM nDCG@10 0.6602002542，BM25 0.6942695073；150胜444平215负。Recall@100 0.9324268644 vs0.9350638649。开发集结果不是官方test榜单。正式CPU总58.5879秒（含导入，下载除外）；推理/检索53.5491秒；峰值RSS1160839168字节。查询/语料矩阵809×384、5183×384；3681文档截断，无查询截断。不同字段/截断/历史预训练预算未控制，不能作纯表示因果对照。

日志development.log；阶段/环境/输出见results/development；EARLY_RUNS保留失败和探索状态，smoke.log、smoke2.log、resource_smoke.log是原日志的路径脱敏副本。原始记录在本地文章verification/raw，不公开作者目录。VALIDATION记录打包前验证；发布后固定commit另记博客元数据。

许可证与来源见NOTICE.md、licenses、SOURCE_MAP.md；大资源通过锁定下载获取。没有发布权重、完整受限语料或凭据。
