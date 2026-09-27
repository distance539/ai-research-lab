# 084 固定源码地图

实际访问与下载日期2026-09-27。Pyserini官方commit API确认`527c917120ff154f84b09ff2b69dc8946085aae4`（提交时间2026-09-26T20:14:02Z）。两份源文件逐行阅读并原样复制，input.lock.json含官方原始字节SHA256，运行前再次检查。许可证为Apache-2.0，原文件头保留；全文在vendor/LICENSE.txt。不是声称2009论文作者发布的原始代码，而是Castorini官方项目提供的RRF实现。

| 公式/模块 | 已实际核验的永久源码入口 | 本地对应及修改 |
|---|---|---|
| RRF入口、克隆输入 | [fusion/_base.py L30–56](https://github.com/castorini/pyserini/blob/527c917120ff154f84b09ff2b69dc8946085aae4/pyserini/fusion/_base.py#L30-L56) | vendor/fusion_base.py原样；run.py直接调用reciprocal_rank_fusion，无算法改写 |
| 每路1/(k+rank) | [TrecRun.rescore L141–146](https://github.com/castorini/pyserini/blob/527c917120ff154f84b09ff2b69dc8946085aae4/pyserini/trectools/_base.py#L141-L146) | vendor/trectools_base.py原样；rank由run.py按冻结分数/ID降序构造，从1开始 |
| 截断、按ID聚合、输出排序 | [TrecRun.merge L234–288](https://github.com/castorini/pyserini/blob/527c917120ff154f84b09ff2b69dc8946085aae4/pyserini/trectools/_base.py#L234-L288) | 官方depth=100或50，k=None返回完整并集；本地统一ID降序后取100。官方默认ID升序不混充同一协议，diagnostics.py单独比较 |
| dataframe输入形状 | [TrecRun.from_list L314–336](https://github.com/castorini/pyserini/blob/527c917120ff154f84b09ff2b69dc8946085aae4/pyserini/trectools/_base.py#L314-L336) | [N,6]列topic/q0/docid/rank/score/tag；本地仅提供adapter，无复制其他包 |
| 官方许可证 | [LICENSE.txt](https://github.com/castorini/pyserini/blob/527c917120ff154f84b09ff2b69dc8946085aae4/LICENSE.txt) | vendor/LICENSE.txt原样；完整上游仓库不打包 |
| MiniLM、池化、BM25、真实数据与评测协议 | [083已公开源码地图](https://github.com/distance539/ai-research-lab/blob/8a3b931f2aa147a71db7812177f0ded4283ec9e4/cycle06/083-embedding-vs-bm25/SOURCE_MAP.md) | inputs三个文件原样复制同commit；model.lock.json/resources.lock.json/baseline_provenance.json保留原身份。不依赖作者机器绝对路径 |
| 全库DenseRetrievalExactSearch | [BEIR search L38–124](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/dense/exact_search.py#L38-L124) | 083已执行，本篇不重复推理；固定MiniLM与tokenizer revision见README，BEIR Apache-2.0 |
| nDCG/Recall交叉核验 | [BEIR evaluate L68–120](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/evaluation.py#L68-L120) | 本篇run.py调用同系列pytrec_eval-terrier0.5.10；audit.py独立标准库重算二值指标，MRR@10独立实现 |
| 独立精确分数、预算和配对诊断 | [RRF原论文](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)的公式定义 | audit.py用Fraction重算，diagnostics.py解析trace；alternate、配对统计、绘图均为自写教学/审计实现，Apache-2.0 |

## 必要适配

run.py通过importlib按上游模块名加载两份完整官方Python文件，使fusion内部的`pyserini.trectools`导入指向原版TrecRun。没有加载Pyserini其他搜索模块或JVM，不声称测试整套Pyserini检索系统。官方计分与聚合字节未变；仅在接口外统一输入rank、同分规则和输出长度。精确分数审计保留浮点引起的内部名次差异。

## 数据许可与分发

SciFact来源固定`68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`：[官方LICENSE.md](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md)，claims/evidence CC-BY-4.0，abstracts ODC-By-1.0，代码Apache-2.0。SCIFACT-LICENSE.md保留全文；本包只分发必要ID/qrels与自生成排名，不复制整库文本。BEIR scifact.zip无不可变revision，用实际SHA256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`与文件锁标识。模型权重不分发。
