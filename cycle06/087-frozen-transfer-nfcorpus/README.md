# 087 冻结配置迁移：SciFact → NFCorpus

真实BEIR NFCorpus全部3633文档/323条test查询。固定082的BM25、083的all-MiniLM-L6-v2、084的RRF60；源域809条开发指标重用公开固定结果。不是重新训练、远领域因果实验、原始NFCorpus全文学习排序论文复现，也未执行085/086重排器。原SciFact300确认查询始终未评测。

## 环境与资源

实测macOS26.6 arm64、Python3.12.14、CPUfloat32、4线程、batch16。requirements.txt为共享环境完整固定版本；不依赖作者机器路径。约1.1GB Python内存，ES另设512MiB堆；建议至少4GB可用内存、2GB磁盘，PyTorch安装空间另计。跨平台结果待人工核验。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache --with-es-macos-arm64
```

最后一个参数仅适用macOS arm64；其他系统省略并从Elastic官方取得同版7.17.9发行包。下载器锁定BEIR源码、模型/tokenizer和NFCorpus文件SHA256；目标ZIP约2.45MB、MiniLM权重约91MB、ES发行包约306MB。无API密钥。未提供新数据revision的上游以实测hash锁定；不匹配即停止，不静默更新。

在独立终端启动本机服务（复用已有正确服务时无需重复）：

```sh
ES_JAVA_OPTS='-Xms512m -Xmx512m' ./cache/elasticsearch-7.17.9/bin/elasticsearch \
 -Ediscovery.type=single-node -Enetwork.host=127.0.0.1 \
 -Ehttp.port=19282 -Etransport.port=19382 \
 -Expack.security.enabled=false -Expack.ml.enabled=false \
 -Eingest.geoip.downloader.enabled=false -Ecluster.name=cycle6-082 -Enode.name=cycle6-local
```

仅监听127.0.0.1；结束在服务终端Ctrl-C。程序不删除旧索引；内容身份生成新索引，遇到已有索引先核对所有文本。

## 最小入口与正式执行

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --smoke
```

输出真实5查询/32文档的三种排名、逐查询指标、聚合、形状、阶段时间、环境和身份。缩库smoke不是benchmark。正式执行删除`--smoke`，并换`--out full-new`。输出目录必须不存在。源域指标来自inputs/source_per_query.json，source_provenance.json记录固定公开commit，已匿名逐值核对。

审计真实保存结果：`python audit.py --cache ./cache --out full-new`。本包原始正式结果为results，故可换成`--out results`。audit依赖公开数据缓存中的qrels，包不再分发NFCorpus原始文本或qrels；首次应运行prepare。只重算结果不代表重新推理。

## 冻结协议与产物

PROTOCOL.md在目标数据下载及推理前登记；seed87。模型与tokenizer revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41，256token，title-space-text，余弦。BM25默认k1=1.2,b=.75，english analyzer，title/txt best_fields/tie_breaker=.5。RRF每路最多100、k60、完整并集计分后输出100、同分文档ID字符串降序。RRF花费两个检索器，不声称计算预算等同单路。无参数选择/训练/目标调优。

results/runs.json与bm25_raw.json保留全部候选和原始分数；rrf_union.json保留46852条并集分数。per_query.json和contrasts.json为三方法/两配对差；summary.json为源域/目标域汇总；data_audit.json保留12334条正标注的等级分布（1:11758，2:576）。nDCG线性gain保留等级，Recall/MRR用>0；不存在的查询不能从分母消失。bm25_missing/empty_query_check保留15条零命中及直接ES核对。

ids/encoding/shape_check/embedding_hashes保存[323,384]与[3633,384]向量身份；实际向量缓存在显式cache/embeddings/<cache_key>，不分发。identity包含模型/tokenizer/prompt/null-decoding、数据、配置、协议和执行代码；data模型上游逐文件锁定。environment记录真实RSS与版本；status记录各阶段与退出码。无GPU/训练checkpoint，索引与嵌入可重建。

## 本次执行与限制

正式nDCG10：BM25 .3268901559，dense .3159407492，RRF .3531146026。对应Recall100 .2481408450/.3115099240/.3224824791。dense相对BM25的109胜/106平/108负说明赢查询略多也可能均值更低；RRF122/136/65。独立审计最大误差3.33e-16。编码37.1345秒，总42.6544秒，Python峰值1044201472字节；ES峰值RSS未测，JVM结束快照另存。源域成本未重测，不作同场速度比较。

第一次命令误漏--smoke，提前执行全库BM25；在检测上游漏返15条零命中查询时失败退出1，尚未执行神经编码或汇总评测。initial_failed及日志保留；随后补齐空查询、保持分母，再运行正确smoke和formal。该修复是协议完整性修复，没有改变打分参数。目标查询已有调试访问，不能宣称完全盲测。详见VALIDATION.md。

后续在这些test输出上选方法就属于探索，不能重复称未见验证。单目标集合与相近医学/科学语料不足以证明普遍迁移；无置信区间、人工金标或医学判断。NFCorpus官方允许学术使用，读者需遵守原作者条款；本包仅分发自写源码、官方许可模块、自生成分数/ID和必要验证证据。
