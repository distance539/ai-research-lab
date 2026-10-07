# 094 — 官方 task 配置的逐项审计

真实SmolLM2-135M-Instruct、SST2 64条既有开发样本；官方lm-evaluation-harness0.4.9.2的multiple_choice协议。模型未训练；128SST及64此前新闻确认样本未推理。主结果32/64，128个候选似然与独立模型前向重算完全一致。状态：技术实验通过；真人四题算术核验待完成（HUMAN_REVIEW.md），不能把机器审计当人工金标。

## 一条最小真实入口

实测Python3.12.14/macOS26.6arm64、torch2.6.0、transformers4.49.0、CPUfloat32/eager/4threads。已有科研环境只新增依赖，原依赖精确版本未改。建议至少4GiB可用内存，模型269MB、数据约6MB重复快照与环境空间。无GPU要求，无训练检查点。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -c environment.lock.txt -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --private private-new --smoke
python audit.py --cache ./cache --out smoke-new
```

最小入口推理2题×2候选；去掉--smoke运行冻结64条。`--mode manual`独立计算4道复核题，仍需真人核对预测、正确数及平均；该mode名称不是完成真人工作的声明。输出目录必须全新，避免覆盖失败或人工编辑。跨设备位级一致性待人工核验。

## 协议与资源

- original model/tokenizer revision12fd25f77366fa6b3b4b768ec3050bf629380bac；非093微调后的模型。
- harness0.4.9.2 commitad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9固定git依赖；安装7个核心文件与官方commit逐字节相同。来源映射SOURCE_MAP.md，哈希upstream_verification.json。
- 官方nyu-mll/glue配置sst2，revisionbcdcba79d07bc864c1c254ccfcedcce55bcc9a8c；train67349/validation872与前篇Stanfordrevision8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb逐行相同。资源锁含全部实际文件SHA-256。prepare只下载固定公开资源或校验既有缓存。
- doc_to_text/choice/target/metric保持官方YAML；custom_dataset是受支持的离线数据传输覆盖，非原封不动默认下载命令。0shot、无chat、无BOS、候选前一个空格、无EOS、无长度归一化、无自由生成、512上限且超长失败、batch1、seed94。
- SHA256('88:'+idx)前64为开发，接下来128保留；samples传入升序的行位置以避开该版本日志题号映射问题。此前64新闻确认未使用。请求/响应缓存关闭，完整identity key保存。

## 预期产物与已执行范围

split.json保存题号和未用确认集；identity.json保存源码/数据/model/tokenizer/协议身份；independent.json保存每题候选token编号、逐token对数概率、总分与真实张量形状；official_samples.json保存官方逐题分数与金标签；official_metrics.json是官方聚合；effective_protocol.json记录默认值；audit.json和offline_audit.json保存对齐/算术检查；environment.json与stages.json记录真实资源及状态。private目录保存原始官方返回（含原文）及输入trace，不要公开提交它。

实际执行4题独立前向、2题官方smoke、初次64题运行（模型完成，题号断言失败）、排序修复后的64题官方运行。正式32/64，模型预测negative61/positive3，真实gold31/33；128得分最大误差0。正式12.097717秒（imports后），官方调用5.814287秒，峰值RSS1310195712B。CPU-only，GPU内存未测。独立计算和官方评分均真实运行，无toy替代。

初次61/64日志doc_id错配，排序samples修复，所选集合与按source_idx对齐得分均未改变；DEVIATIONS.md和initial_identity_mismatch.json保留证据。官方读取调用目录Git信息的非致命警告不是上游版本来源。

## 分发与边界

无原始影评、parquet、题干tokenIDs、模型权重、cache、虚拟环境、私有路径或凭据。原始官方返回与错误日志保留本地，公开包包含可再生成它们的完整脚本与派生逐题证据。human_review.json诚实记录人工状态；目前真人未返回，技术验证提前，后续核验只能称回顾性。准确率使用的是反复使用的开发样本，不是完整SST2或GLUE成绩；与先前自由生成结果同时改变了提示和评分，不支持单因素因果解释。
