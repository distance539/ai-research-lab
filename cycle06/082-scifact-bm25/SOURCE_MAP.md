# 固定源码地图

访问并实际阅读：2026-09-25。GitHub官方API返回BEIR提交，之后成功完成的本地git clone也得到相同HEAD；API JSON保存在evidence/。完整上游代码不打包，由prepare.py按提交和压缩包SHA取得；上游Python逐文件哈希在resources.lock.json。本地代码均为独立编写的调度、审计与适配，不原样复制论文作者函数。

## 仓库、版本、许可

- BEIR： https://github.com/beir-cellar/beir ，commit `ef83d29307061c65d04b035b4f4e7c18bd8374af`，源码pyproject版本2.2.0（本次为该提交，不冒称发布标签）。Apache-2.0，`licenses/BEIR-Apache-2.0.txt`。
- SciFact： https://github.com/allenai/scifact ，commit `68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`。代码Apache-2.0；声明/标注CC BY 4.0；摘要ODC-By 1.0。原始LICENSE逐项说明，保存于`licenses/SciFact-LICENSE.md`。当前HF卡的CC BY-NC 2.0与原始仓库不一致，本实验引用原始来源并不重新打包语料。
- Elasticsearch： https://github.com/elastic/elasticsearch ，实际服务7.17.9/build `ef48222227ee6b9e70e502f0f0daa52435ee634d`，Lucene8.11.1，均由运行中的info核验。下载发行包附Elastic License 2.0，源码头为Elastic-2.0/SSPL-1.0双选，不能笼统称Apache。只提供下载器，不重打包发行包。`licenses/Elasticsearch-LICENSE.txt`。

## 模块与永久链接

| 公式/模块 | 已核验永久文件/函数/行 | 本地对应与改动 |
|---|---|---|
| 语料、查询、qrels加载及查询筛选 | [GenericDataLoader.load，69–91](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/datasets/data_loader.py#L69-L91)；[_load_corpus/_load_queries/_load_qrels，104–134](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/datasets/data_loader.py#L104-L134) | `run.py:data_audit/main`；直接调用官方loader，额外独立检查重复/ID引用/原始划分/空白差异。未复制loader。 |
| 建索引字段 | [BM25Search.index，89–102](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/lexical/bm25_search.py#L89-L102) | `run.py:main`直接调用；显式refresh与count；禁止上游initialise删除已有索引。 |
| 候选返回及同ID过滤 | [BM25Search.search，54–87](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/lexical/bm25_search.py#L54-L87) | `run.py`保存原始101返回，再截到最多100；`common.py:ranked`独立定义同分顺序。 |
| 字段合并 max + .5×min | [ElasticSearch.lexical_multisearch，202–245](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/lexical/elastic_search.py#L202-L245) | 官方原样执行；`run.py`保存真实explain。公式由best_fields与tie_breaker语义解释，见[7.17官方文档](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/query-dsl-multi-match-query.html)。没有手写检索替代。 |
| BM25默认模型与参数 | [SimilarityService，71–75](https://github.com/elastic/elasticsearch/blob/ef48222227ee6b9e70e502f0f0daa52435ee634d/server/src/main/java/org/elasticsearch/index/similarity/SimilarityService.java#L71-L75)；[SimilarityProviders.createBM25Similarity，282–293](https://github.com/elastic/elasticsearch/blob/ef48222227ee6b9e70e502f0f0daa52435ee634d/server/src/main/java/org/elasticsearch/index/similarity/SimilarityProviders.java#L282-L293) | 运行发行包中的LegacyBM25Similarity，默认1.2/.75；`config.json`记录。Java源文件通过固定SHA的raw URL实际下载阅读，字节哈希在evidence/source_reads.json。不把Legacy计分简化成自写公式。 |
| nDCG/Recall宏平均、四舍五入 | [EvaluateRetrieval.evaluate，69–122](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/evaluation.py#L69-L122) | `run.py`调用官方评测+同版本pytrec_eval逐查询接口；`common.py:metrics`独立教学公式，只用于核算真实run，并不生成检索分数。`audit.py`逐项比较；MRR@10独立计算。 |
| 上游抽象接口兼容 | [BaseSearch，6–38](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/base.py#L6-L38) | `compat.py:RunnableBM25`为基于官方接口编写的独立适配；两个dense-only方法明确报TypeError，不伪装为支持；其余行为全继承官方。未修改上游文件。 |
| 原始划分及原始许可 | [README数据说明](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/README.md#L94-L126)；[LICENSE，1–14](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md#L1-L14) | `run.py:data_audit`独立比较下载数据；不声称运行SciFact原论文事实核验模型。原始数据latest URL以实际哈希锁定，不虚构数据commit。 |

## 引用与分发边界

原样附带的是许可文件、官方API返回和实际运行输出，不包含完整上游仓库或神经权重。BEIR官方组件通过固定资源下载直接执行。本地脚本、适配层和指标重算是本篇自写实现；其必要改动全部在上述表格列明。

数据请引用 David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, Hannaneh Hajishirzi, *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020。BEIR请引用 Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych, 2021。qrels_used.json是SciFact标注的检索格式派生，保留CC BY 4.0与上述作者归属，不将其标成自有标注。原始语料只留本地缓存，不进入code.zip。
