# 090 — 更多样本还是更多 token：等监督预算的真实微调

SmolLM2-135M-Instruct + SST-2，CPU全参数SGD。32条嵌套训练池/64开发查询，128确认样本未推理。单种子、短预算、探索性实验；不是完整SFT或GLUE排行榜，不复现大规模预训练 scaling law。

## 最小入口

实测 Python3.12.14/macOS26.6 arm64、torch2.6.0、transformers4.49.0。requirements.txt固定直接依赖；environment.lock.txt记录复用环境完整版本。建议6GiB可用内存、5GB可用磁盘（环境另计）。模型269MB；3个正式检查点各538MB，另有3个smoke检查点。checkpoint序列化容器包含文件名前缀，因此跨输出名文件哈希可不同，不能据此认为参数不同。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --private ./private-new --smoke
python audit.py --cache ./cache --out smoke-new
```

删除`--smoke`并改用全新out目录运行正式实验。不可覆盖已有结果。prepare.py只从17个固定revision文件下载并SHA256核验，已有正确缓存直接复用，离线run每次再核验。原数据、模型、环境与输入token不分发。common.py/scoring.py/prepare.py完整包含，无上一文章目录或作者机器路径依赖。

## 预先冻结的规则

PROTOCOL.md先于试跑和正式运行保存；config.json记录所有选择。按SHA256('89:'+idx)在SST-2 train每类选16条，负/正交替；前4条/类组成嵌套8条，集合等于089的训练集（顺序调整为每batch类别平衡）。32条无规范化完全重复，与64开发及128保留确认样本无规范化完全重合；不排除短语/语义重合或上游污染。开发集仍为088中SHA256('88:'+idx)前64条；确认样本为随后128条。

目标JSON回答每条8tokens含EOS，经真实tokenizer逐条验证；只监督回答至EOS，prompt/pad标签为-100，不按EOS ID抹掉真实EOS。最大256不截断；动态右padding、无packing。所有组同初始权重、SGD lr.001/no momentum/decay、clip1、batch2、CPUfloat32/eager/4threads/seed90、eval模式但保留autograd。模型本身已经instruction-tuned。固定最后一步，不按开发分数选checkpoint。

| arm | 独特行 | 轮数 | 曝光 | 步数 | 监督token | 输入token | 含padding位置 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeat8_short | 8 | 4 | 32 | 16 | 256 | 2736 | 2944 |
| repeat8 | 8 | 16 | 128 | 64 | 1024 | 10944 | 11776 |
| unique32 | 32 | 4 | 128 | 64 | 1024 | 10556 | 11344 |

主对照是后两组：等监督量/更新数/类别，不是等全输入token、等FLOPs或等墙钟。短训练组是预算诊断，不用于独特样本因果结论。每个batch16有效target，meanCE分母一致。主指标JSON联合正确J=F且C；F严格格式、C唯一显式情感词匹配原标签的代理，不是人工语义金标。plain诊断沿用088，greedy32、无beam/采样。

## 本次真实结果

各64开发样本：before plain F64/C57/J57，JSON F0/C50/J0；repeat8_short plain64/59/59，JSON0/53/0；两个主组plain64/33/33，JSON64/58/58。两主组全部128条输出文本逐样本相同；plain62negative/2positive。它们不等价为参数相同，也不证明增加独特数据无效。JSON监督语境下NLL下降，plain偏置增大；未直接测其因果机制。

原小8行answerNLL before1.028291、short.666150、repeat8.161312、unique32.172013；32池NLL1.007893/.650737/.157833/.163056。32池含训练样本，不是未见泛化估计。不同batch的step loss不得画成独立验证曲线。无调参、无重跑挑优、无多种子/置信区间。

正式144更新/512次生成，118.538732秒（import后，含加载/校验/训练/推理/NLL/保存），峰值RSS2758885376B。三组训练5.448526/20.416364/20.713267秒。GPU内存未测。试跑每组2步/2开发、15.101763秒、RSS2623045632B；资源估计先通过再正式运行。SOURCE_MAP给出真实官方commit/行号、许可证和本地对应。

## 产物与验证

results与smoke含stdout/stderr、stages、identity、rows、steps、budget、predictions、summary、checkpoints、train_nll、environment；results/audit.json核验512输出/144步/32行再编码、分组、预算和配对结果。缓存identity含模型/tokenizer/data精确revision与哈希、template、prompt/decode、划分、执行代码哈希。

audit.py独立重建训练日程、数量和格式解析，核查返回token与原始输出一致、gold标签正确和确认隔离。官方CE与独立移位CE最大差1.19209e-7。三个最终模型参数更新非零，完整权重本地保存且记录哈希，不公开分发。跨平台位级一致性待人工核验。
