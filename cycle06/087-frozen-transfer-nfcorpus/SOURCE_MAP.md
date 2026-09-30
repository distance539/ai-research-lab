# 087 固定源码地图

访问与实际核验日期：2026-09-30。upstream_verification.json包含18个匿名HTTP200来源的文件哈希、三仓库commit API实返SHA、官方与本地比较。完整BEIR通过固定archive获取，逐个Python文件核对；不打包整库。

| 公式/模块 | 官方永久入口、已核验commit与行 | 本地对应、使用方式、许可 |
|---|---|---|
| 语料/查询/qrels | [BEIR GenericDataLoader.load L69–91](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/datasets/data_loader.py#L69-L91) | run.py调用完整未改官方loader；common.py独立读取用于审计；Apache-2.0 |
| BM25检索/索引 | [BM25Search.search/index L54–102](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/lexical/bm25_search.py#L54-L102) | run.py直接执行；compat.py继承082自写适配、拒绝dense-only接口。新增空查询补零并保持323分母，不改分数；Apache-2.0 |
| 字段打分合并 | [ElasticSearch.lexical_multisearch L202–245](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/lexical/elastic_search.py#L202-L245) | 原样官方best_fields与tie_breaker=.5；title/txt两个字段；Apache-2.0 |
| ES运行时BM25 | [SimilarityProviders L282–293](https://github.com/elastic/elasticsearch/blob/ef48222227ee6b9e70e502f0f0daa52435ee634d/server/src/main/java/org/elasticsearch/index/similarity/SimilarityProviders.java#L282-L293) | 沿用082已核验源码映射；本次服务实际info再次匹配7.17.9/build ef4822，Lucene8.11.1；发行包Elastic License2.0，不重新打包。源码许可见文件头双选。 |
| 冻结神经编码 | [SentenceTransformer.encode L461–681](https://github.com/UKPLab/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/SentenceTransformer.py#L461-L681) | encoder.py调用官方3.4.1，增加长度/形状/耗时/外部缓存记录；官方安装文件与原文件统一换行后完全一致；Apache-2.0 |
| token截断 | [Transformer.tokenize L476–514](https://github.com/UKPLab/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/models/Transformer.py#L476-L514) | 固定256、title-space-text；官方运行，无prompt；Apache-2.0 |
| 掩码均值与单位化 | [Pooling.forward L137–184](https://github.com/UKPLab/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/models/Pooling.py#L137-L184)、[Normalize.forward L13–15](https://github.com/UKPLab/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/models/Normalize.py#L13-L15) | run.py独立masked mean对照官方输出，error0；主链直接执行官方；Apache-2.0 |
| 全库余弦检索 | [BEIR exact search L38–124](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/dense/exact_search.py#L38-L124)、[cos_sim L11–30](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/dense/util.py#L11-L30) | run.py/encoder.py调用未改原版；本次无ID碰撞、向量有限；Apache-2.0 |
| RRF入口 | [Pyserini reciprocal_rank_fusion L30–56](https://github.com/castorini/pyserini/blob/527c917120ff154f84b09ff2b69dc8946085aae4/pyserini/fusion/_base.py#L30-L56) | vendor/fusion_base.py完整原样，fusion.py模块加载适配；Apache-2.0，保留头与vendor/LICENSE.txt |
| 倒数排名与并集 | [TrecRun.rescore L141–146](https://github.com/castorini/pyserini/blob/527c917120ff154f84b09ff2b69dc8946085aae4/pyserini/trectools/_base.py#L141-L146)、[merge L234–288](https://github.com/castorini/pyserini/blob/527c917120ff154f84b09ff2b69dc8946085aae4/pyserini/trectools/_base.py#L234-L288) | vendor/trectools_base.py完整原样；外层统一ID降序，同084；输出100，输入每路最多100；Apache-2.0 |
| 评测与独立计分 | [BEIR evaluate L68–120](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/evaluation.py#L68-L120) | run.py直接使用同评测后端pytrec_eval-terrier0.5.10；common.metrics独立线性gain公式，audit.py逐查询核对，不把等级二值化；自写Apache-2.0。BEIR文件说明接口，并非本地公式的复制来源。 |
| 配对差、源域对照 | [084固定公开原始指标](https://github.com/distance539/ai-research-lab/blob/804da74ba3484e0156183951e89228d8e71c780b/cycle06/084-rrf-rank-changes/results/per_query.json) | inputs/source_per_query.json仅取三个方法，所有值匿名读取核对相等。配对差/汇总/审计为独立教学实现。源域不重新编码。 |

目标权重和tokenizer为sentence-transformers/all-MiniLM-L6-v2，revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41；[固定模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md)，模型Apache-2.0，11文件URL/hash在model.lock.json。底层Transformers4.49.0与Torch2.6.0沿用083环境，非新模型或微调。

NFCorpus：Vera Boteva, Demian Gholipour, Artem Sokolov, Stefan Riezler, ECIR2016。原始主页https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/标明学术使用；本包不再分发正文、查询文本或原始qrels，只含自生成排名/度量/ID。下载器取得BEIR导出版本；官方MD5和实际SHA256、解压文件hash记录resources.lock.json，无不可变revision则以这些字节身份锁定。原始全文版与BEIR摘要版数量不同，不冒称完整复现原论文训练实验。SciFact源指标对应Wadden等EMNLP2020，参考许可全文在licenses/，不打包原始语料。

run.py/common.py/encoder.py/compat.py基于本系列082–084自写模块改写，代码完整随包；必要变化为目标数据、分级指标、漏查询补零、外部缓存身份。fusion.py仅重用自写加载层，vendor为明确标识的原样官方代码。无隐藏绝对路径依赖。PROTOCOL.md、README.md、VALIDATION.md解释研究范围及失败；本次不调用reranker，不以086四条件中“最佳”设置代替冻结检索协议。
