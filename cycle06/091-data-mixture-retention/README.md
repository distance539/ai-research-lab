# 091 — 数据混合与能力保留：96条开发样本的低基线实验

真实 SmolLM2-135M-Instruct + SST2/AG News，小预算CPU全参数SFT。三个等监督量混合比例，加一个半量目标控制。新闻基线仅8/32且几乎都答A，因此不能宣称验证了有意义的旧能力保留，也没有证明灾难性遗忘。单种子探索，所有失败/负结果保留。

## 最小运行

实测macOS26.6 arm64/Python3.12.14/torch2.6.0/transformers4.49.0。需要约6GiB可用进程内存（实测峰值5.63GB）、约7GB磁盘（模型269MB、数据约23MB、正式+smoke八个检查点各538MB，环境另计）。固定依赖如下；复用环境完整版本在environment.lock.txt。建议读者给系统留额外内存。跨平台位级一致性待人工核验。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --private ./private-new --smoke
python audit.py --cache ./cache --out smoke-new
```

正式运行去掉`--smoke`并使用全新的out目录，不覆盖旧结果。下载器核对20固定文件SHA256，已有正确缓存直接复用；run为离线执行。所有自写模块已包含，不依赖前篇本地路径。默认生成完整原始输出；公开分发证据前可运行`python export_evidence.py --out YOUR_OUT --private YOUR_PRIVATE`，先将原始预测复制至私有目录并保留完整审计，再删去可能复述原新闻的长回答及可逆token IDs。随后audit会明确报告完整重解析与删节记录数。

## 协议和范围

PROTOCOL.md先于推理登记；唯一修订是预处理发现新闻行112808长422tokens，把新闻训练上限256→512，保留原样本、不截断。该次失败发生于模型加载前，initial_preprocess_failed保存退出1和日志。SST2训练上限仍256；实际评测输入最长182，无截断。

SST2沿用09032训练（每类16，SHA256('89:'+idx)，负正交替）、08864开发（SHA256('88:'+idx)前64）；随后128确认未推理。AG News使用固定镜像parquet行位置编号，按SHA256('91:'+idx)在类内排序，train各取8=32；test各取8=32开发，随后各16=64保留。test子集明确作为开发，不是最终test benchmark。规范化精确重复/训练与保留交集为0，不声称语义去重、无预训练污染或独立文章抽样。标签A世界/B体育/C商业/D科技。

模型/tokenizer12fd25f77366fa6b3b4b768ec3050bf629380bac；SST2 8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb；AG News eb185aade064a813bc0b7f42de02595523103ca4。已过指令训练，不是base零起点。所有组134515008参数、原始权重独立初始化，SGDlr.001/momentum0/decay0/clip1，CPUfloat32/eager/4threads/seed91，eval模式保留autograd。greedy32，无调参/早停/最优检查点选择。模型、数据、模板、提示、解码、源码版本全部进入cache identity。

情感JSON含EOS8目标/行，新闻字母含EOS2目标/行；真实EOS与PAD同ID2，按长度和prefix保留。每更新16目标：目标2行；新闻8行分4×2微批次，官方mean loss乘n_valid/16后累积梯度，最后一次clip/step。公式与官方函数见SOURCE_MAP。

| arm | target/news更新 | target/news曝光行 | 监督token | 输入token | padded位置 |
|---|---:|---:|---:|---:|---:|
| target_only | 64/0 | 128/0 | 1024 | 10556 | 11344 |
| replay25 | 48/16 | 96/128 | 1024 | 25477 | 28316 |
| replay50 | 32/32 | 64/256 | 1024 | 40398 | 45288 |
| target_half | 32/0 | 64/0 | 512 | 5278 | 5672 |

主组三个等监督量/等更新数，不等输入、FLOPs、曝光行或墙钟。半量控制只匹配replay50目标暴露，故仍有额外更新/顺序/梯度等差别；不作单一机制因果结论。两任务流各自循环，不随机重洗。选择规则：news joint至少baseline-1，再最大化JSON joint，平局较小回放。只限开发。

## 实测结果

| 状态 | SST plain F/C/J | SST JSON F/C/J | News F/C/J |
|---|---|---|---|
| before | 64/57/57 | 0/50/0 | 30/8/8 |
| target_only | 64/33/33 | 64/58/58 | 29/8/8 |
| replay25 | 64/34/34 | 64/58/58 | 31/8/8 |
| replay50 | 64/34/34 | 59/54/51 | 32/5/5 |
| target_half | 64/55/55 | 61/54/51 | 30/8/8 |

SST分母64，news32。F严格格式，C唯一显式标签匹配的数据标签代理，J二者同时成立；不是人工金标。新闻全部类各8，因此常数A为8/32；baseline30个A并不能证明分类能力。replay50变20A/9C/3D，新闻配对2胜25平5负。replay25和target_only JSON同58分但1胜62平1负；总分不等于逐例行为相同。半量目标相对replay50保留更好plain行为：后者2胜39平23负。选择器选target_only；这是预定开发规则的结果，不能推广为回放无用。192确认样本未推理。

224更新/800生成，223.410775秒（第三方import后，包括锁核对/加载/训练/生成/保存），RSS5634686976B。四组训练20.903547/38.797355/60.886753/10.470313秒。GPU内存未测。成功smoke14更新/40生成、19.114804秒、RSS5302763520B；预估382.296秒<1200秒、RSS<6GiB后正式执行。

## 证据、审计与许可

results/与smoke/：阶段状态、配置身份、划分、行级掩码、逐步微批次、预算、逐例生成、聚合/配对、预定选择、环境、检查点哈希和日志。完整四个正式检查点及原始token traces/生成留在用户选择的private目录，不分发。

原始800输出已全部独立解析，64训练行重新编码、224步重建，官方CE最大误差2.384185791015625e-7。raw_audit_before_export.json记录原始审计及原文件哈希；distribution.json说明删节项。公开包的少数新闻复述行保留哈希、原token数量、原派生评分，audit不会冒称这些文本公开可重解析；可自行正式运行取得完整新输出。训练stdout不含原新闻文本。

SST2与AG News许可unknown；不分发原数据、原prompt/input IDs、模型权重、环境和缓存。完整下载脚本/版本/哈希以及第三方许可已提供。检查点哈希仅用于本地实物核对；torch保存的容器名前缀可能造成跨文件名哈希不同，不能据此判断权重变化。模型和官方软件直接API调用，独立教学调度不是官方SmolLM训练配方。详细映射和授权见SOURCE_MAP.md、THIRD_PARTY.md。
