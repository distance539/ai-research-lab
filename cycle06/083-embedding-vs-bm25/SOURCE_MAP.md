# 083 源码地图

访问/核验日期：2026-09-26。官方源文件已实际下载并按行阅读；`upstream_verification.json` 保存远端与安装版哈希。Sentence Transformers wheel 使用 CRLF、GitHub 文件 LF，原始字节不同，但统一换行后的全文完全相同；Transformers BERT 文件字节也一致。不能把版本号单独当 commit。

## 固定上游与公式路线

| 模块与本地路径 | 已核验官方永久入口 | 使用方式与许可证 |
|---|---|---|
| 冻结推理、批处理；`run.py:Encoder.encode` | [SentenceTransformer.encode L461–681](https://github.com/huggingface/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/SentenceTransformer.py#L461-L681) | 调用官方3.4.1；Apache-2.0；未复制整文件 |
| 256 token截断；`run.py:model.max_seq_length` | [Transformer.tokenize L476–514](https://github.com/huggingface/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/models/Transformer.py#L476-L514) | 官方执行；最长长度显式锁定，不误用tokenizer的512默认上限 |
| 掩码均值 $z=\sum m_t h_t/\sum m_t$；`run.py:shape_check` | [Pooling.forward L137–184](https://github.com/huggingface/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/models/Pooling.py#L137-L184) | 主链官方执行；独立重算公式对照，掩码含特殊token，排除padding；Apache-2.0 |
| 单位化 $e=z/\|z\|_2$ | [Normalize.forward L13–15](https://github.com/huggingface/sentence-transformers/blob/7d52a069e0b37d976b3ed3f674a6180436c27574/sentence_transformers/models/Normalize.py#L13-L15) | 模型模块原样调用，Apache-2.0 |
| 目标模型与tokenizer；`config.json`, `model.lock.json` | [MiniLM模型卡固定revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md)；[模型模块](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/modules.json) | 官方权重不改；模型卡Apache-2.0；下载11文件均记录哈希，未分发权重 |
| 真实Transformer forward | [BertModel](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/bert/modeling_bert.py#L957-L985) | Transformers4.49.0，标签对象dcc2b5e指向该commit，Apache-2.0 |
| 标题空格摘要；`run.py:Encoder.encode_corpus` | [BEIR extract_corpus_sentences L8–23](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/models/util.py#L8-L23) | 基于官方列表分支改写为简洁单进程adapter；保留Apache attribution；新增形状/时间/长度/向量保存。未采用上游多GPU或prompt扩展 |
| 全库余弦；`run.py:DenseRetrievalExactSearch` | [BEIR search L38–124](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/dense/exact_search.py#L38-L124)；[cos_sim L11–30](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/search/dense/util.py#L11-L30) | 调用固定原版，无源码补丁；Apache-2.0。保留NaN处理、top_k+1与ID排除逻辑；本次向量有限且无ID相撞。官方单查询索引隐患未修复，本smoke为5查询 |
| 评测；`run.py`, `common.py:metrics` | [BEIR evaluate](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/beir/retrieval/evaluation.py#L68-L120) | BEIR/pytrec_eval实际执行；独立Python二值nDCG/Recall/MRR审计；不是复制上游评分函数 |
| BM25旧结果与本地公共函数 | [082已发布固定目录](https://github.com/distance539/ai-research-lab/tree/cae9252b4daa0dd50625e6b06b2f62cb36547226/cycle06/082-scifact-bm25) | `common.py`原样复用自写模块；`baseline/`四文件来自082并保留对应哈希。本篇不重跑ES，不声称同场计时 |
| 配对差 $\Delta_q=M_q(D)-M_q(B)$；`audit.py`, `precision_audit.py`, `plot.py` | 无上游代码依赖公式 | 独立教学/审计实现；完整源码随包；不构造人工金标，不作显著性推断 |

## 数据与分发

SciFact原始项目固定commit：`68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`，[许可证](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md)：代码Apache-2.0，claims/evidence CC-BY-4.0，摘要ODC-By-1.0。BEIR下载无不可变revision，使用082已实测下载SHA256及解压文件SHA256作为身份；未虚构数据版本。包不含完整语料或模型；qrels/ID与本次自生成检索分数用于复核，保留原作者 attribution。

完整BEIR不打包，`prepare.py`获取固定archive并核对所有上游Python文件哈希；Sentence Transformers/Transformers由固定版本requirements安装。本篇自写源码Apache-2.0。许可证全文见`licenses/`；模型卡许可证由固定README YAML核对。绘图输出不是imagegen数据图。
