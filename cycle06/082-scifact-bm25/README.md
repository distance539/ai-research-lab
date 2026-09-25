# 082 SciFact：官方 BM25 开发集实验

Python 3.12；真实 BEIR→Elasticsearch 7.17.9/Lucene 8.11.1→BEIR/pytrec_eval。
完整 5183 文档、809 BEIR train 查询。BEIR test=原始 dev 的300查询仅作结构审计，未检索、未调参。
本目录所有自写模块随包交付，不依赖作者绝对路径。命令里的路径由使用者指定。

## 安装与数据

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache /path/to/research_cycle6 --with-es-macos-arm64
```

`--with-es-macos-arm64` 仅适用本次实测 macOS Apple Silicon；其他系统需自行安装官方同版本发行包，跨平台结果待人工核验。发行包约292 MiB（十进制约306 MB），BEIR数据约2.7 MiB；建议预留2 GiB磁盘。只安装词项路径实际需要的依赖，不安装上游全部神经模型依赖；固定BEIR源码通过 `sys.path` 加载，所有上游Python文件哈希逐项核验。`resources.lock.json` 包含URL、真实SHA、文件大小及原始与导出数据哈希。原始latest URL不是不可变revision，哈希不符即停止，不自动更新锁。

若网络受控，在允许联网的终端运行获取命令。禁止将校验失败当作成功；没有本地已验证缓存时不能只运行toy冒充本实验。

## 启动实验服务（单独终端）

```bash
ES_JAVA_OPTS='-Xms512m -Xmx512m' /path/to/research_cycle6/elasticsearch-7.17.9/bin/elasticsearch \
  -Ediscovery.type=single-node -Enetwork.host=127.0.0.1 \
  -Ehttp.port=19282 -Etransport.port=19382 \
  -Expack.security.enabled=false -Expack.ml.enabled=false \
  -Eingest.geoip.downloader.enabled=false -Ecluster.name=cycle6-082 -Enode.name=cycle6-local
```

仅本机监听，需允许本机端口绑定。运行结束在服务终端 Ctrl-C。索引名由内容身份生成；代码不会删除任何已有索引。初次运行需等待服务启动。

## 一条最小入口与审计

```bash
python run.py --cache /path/to/research_cycle6 --out results/my_run
python audit.py --run results/my_run
```

`--limit 5` 是最小CPU smoke，按数字ID排序选前5开发查询，但仍索引全部语料。默认运行全部809查询。重复运行时换新的输出目录，加 `--reuse-index`；程序核对索引内全部文档字段与当前语料后复用。确认集没有命令行入口，未来启用必须显式修改协议。

主要产物：`raw_results.json`保存上游所有返回（通常101）；`results.json`与`run.trec`保存最多100候选；`per_query.json`保存完整逐查询指标；`summary.json`保存聚合；`qrels_used.json`、`data_audit.json`、`identity.json`、`environment.json`、`status.json`可追溯。`explanation.json`保存真实服务对首查询首文档的打分解释，`analyzer_example.json`保存分析器token结果。没有训练checkpoint，索引本身为可重建状态。runfile只有候选分数，不是809×5183稠密矩阵。

## 已执行范围

`results/development`：退出0，5183文档/809查询/80827保留候选，nDCG@10=.6942695072544098、Recall@100=.9350638648537289、MRR@10=.6625198657955147。133查询top10无相关，47查询top100无相关。独立标量重算max error=1.1102230246251565e-16。运行状态记录各阶段时间，Python峰值RSS；服务只记录结束时JVM堆快照，未测服务峰值RSS。下载、服务启动不计入运行秒数；此前smoke已预热服务。

`smoke`失败：原始摘要和导出摘要有1055纯空白差异，审计改为同时记录差异及归一化等价。
`smoke2`失败：上游BM25类缺少两个稠密抽象接口的具体实现。
`smoke3`成功：5真实查询，完整语料。
`compat.py`仅显式拒绝稠密API，保留官方检索评分。没有空函数、模拟评分或神经模型权重。详见 `VALIDATION.md` 与 `SOURCE_MAP.md`。

## 固定协议和边界

`PREREGISTRATION.md`在真实检索前保存。一个配置，无参数搜索；seed=82记录但BM25无随机初始化。english analyzer；标题/摘要独立字段；best_fields、tie_breaker=.5；单分片；batch64；BM25默认k1=1.2,b=.75。服务构建必须匹配指定hash。nDCG@10为主，Recall@100和MRR@10为辅；pytrec_eval同分按字符串文档ID降序，本地逐查询审计一致。截断前后的候选同时保存。空查询结果必须仍有分母，本实现发现漏查询会拒绝验收。

缓存身份包含模型/分析器/prompt(null)/数据/解码(null)/上游提交/本地代码/依赖/具体查询ID。配置变动不能复用旧结果。开发查询可能共享论文，不在本文计算独立同分布置信区间。未运行官方benchmark测试集、稠密检索、重排、事实核验、训练、GPU或人工错误分类。

## 分发

代码包保留本次必要源码、固定依赖、配置、许可说明、运行日志与结果；排除环境、缓存、完整语料、发行包、索引、模型权重。语料仅由下载脚本获取；结果中的qrels派生自SciFact标注，保留作者和CC BY 4.0 attribution。最小样例为自写人工构造的评测fixture，绝不当作真实实验。初始失败日志可能包含原机器路径，这些只作历史证据，不作为运行依赖。
