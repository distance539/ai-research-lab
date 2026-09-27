# 084 RRF改变哪些排名？SciFact开发集受控融合

复用083公开固定版本中的真实BM25和all-MiniLM-L6-v2排名，在完整5183文档来源的809条BEIR train查询上运行固定官方Pyserini融合。没有重新编码、启动ES、微调或执行BEIR test；这是冻结真实排名→融合→评测→审计的端到端实验。

## 安装与最小入口

实测Python3.12.14、macOS26.6 arm64、CPU，依赖版本见requirements.txt。建议至少1GiB可用内存，磁盘200MB加Python环境；不同平台安装和性能待人工核验。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py
python run.py --out results/my_smoke --smoke
```

小运行使用5条真实开发查询，仍读取冻结的每路top100；不是缩小原检索语料。完整运行与审计：

```bash
python run.py --out results/my_run
python audit.py --out results/my_run
python diagnostics.py --out results/my_run
MPLCONFIGDIR=./mpl-cache python plot.py --input results/my_run --out rank_changes.png
```

输出目录必须不存在，避免覆盖。完整输出含8套排名runs.json.gz、逐查询per_query.json、136650行主配置候选贡献trace.csv.gz、汇总、极端案例、身份、阶段状态、计时与内存。随机种子84；主k=60，k=10/100为开发敏感性，不据此替换主结论。每路深度与输出100独立设置。

已保存记录可离线用`python audit.py --out results`核验（标准库Python3.9+），但会刷新results/audit.json；审计副本更适合保留原始字节。正式run.py执行完整官方fusion与trectools模块，不需要安装整个Pyserini或Java。仅两份上游文件原样随包，动态加载器提供准确的模块导入名，没有替换官方计分函数。

## 数据、模型与身份

inputs包含两份真实run与qrels，input.lock.json锁定其SHA256及083来源commit。全部可分发小资源已包含，不需下载权重。若删除输入或vendor，`python prepare.py --download`从固定提交恢复并验哈希。代码只使用公开ID、相关性映射和实验分数，未打包完整语料、模型权重、缓存、环境或密钥。

all-MiniLM-L6-v2与tokenizer均固定1110a243fdf4706b3f48f1d95db1a4f5529b4d41；原始编码最大256、384维、标题空格摘要、无prompt与解码。模型文件锁model.lock.json、原数据/BEIR锁resources.lock.json、BM25来源baseline_provenance.json均保留。原检索完整复现请按SOURCE_MAP中083/082固定公开入口执行；本篇有意冻结它们的输出，融合本身不加载模型。

## 如何解释结果

主RRF nDCG@10=0.712219，BM25=0.694270；交替合并=0.713112，RRF略低。50+50两种规则使用完全相同候选集合，RRF仍略低。结果为开发探索，无显著性保证。100+100并集不等于最终100；候选变多和排序公式的影响必须分开。保留同分ID降序协议；官方默认升序另做诊断。负结果、7个有理数/浮点排序差异在audit.json保留。

完整操作日志development.log/audit.log；results/status.json含命令、阶段、退出码。SOURCE_MAP对应官方文件、函数、行号与许可证。NOTICE说明改动及分发范围。
