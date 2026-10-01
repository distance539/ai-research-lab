# 088 — 微调前的能力与格式基线

真实 SmolLM2-135M-Instruct、公开 SST-2 validation 的固定64条开发样本、两种输出约定。共128次贪心生成；没有训练、提示词搜索或完整GLUE评测。模型已完成官方指令微调；本篇是后续任务微调之前的起点，不是未经微调的base模型。

## 环境与下载

实测macOS26.6 arm64、Python3.12.14、torch2.6.0、Transformers4.49.0、CPUfloat32、4线程、batch1/eager。requirements.txt为所需固定直接依赖；environment.lock.txt为复用环境所有包版本，可选择据此严格重建。跨平台位级一致性待人工核验。模型权重约269MB，全部锁定资源约273MB；实测峰值约1.25GB，建议4GB可用内存，2GB磁盘（安装环境另计）。无API密钥。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --smoke
```

这条最小入口运行2条真实样本×2约定，输出原始生成、token IDs、逐例评分、聚合、形状、环境和阶段状态。正式64条运行：删除`--smoke`，使用新的`--out full-new`。输出目录必须不存在，防止覆盖。下载器按revision和逐文件SHA256锁定，已有正确缓存可复用；不匹配会失败，不静默更新。

独立复核：`python audit.py --cache ./cache --out full-new`。包内已有正式结果`results/`、真实试跑`smoke/`、首次提前运行`initial_development/`，可直接对这些目录审计；审计本身不重新生成回答。

## 冻结协议

PROTOCOL.md在首次推理前保存。SST-2 validation872条按SHA256('88:'+idx)排序，前64开发、后128保留确认、余680不用。确认样本未做推理/案例分析，官方test和train没下载。本地划分不改变数据原始validation身份，不声称没有预训练污染。

模型和tokenizer同一revision12fd25f77366fa6b3b4b768ec3050bf629380bac；数据8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb。提示词逐字见config.json；同一system、任务、标签顺序和官方chat template，唯一区别是输出要求。无few-shot；max_new_tokens32、greedy、一束、EOS2；输入不截断，超过512即报错。JSON输入多11token，不是等token成本或纯语法因果实验。

F为格式合规率；C为唯一显式情感词匹配来源标签的正确率；J=F且C的联合正确率。JSON必须只有sentiment键，禁止重复键/围栏/额外键，值小写。C不是人工语义真值，否定句等可能误导抽取器。空输出/双标签计错，全部64保留分母；C|F在分母0时为null。独立audit重查源标签、token解码、prompt哈希、样本ID和指标；包含8个格式评分边界检查。

## 本次结果与资源

plain：F64/64、C57/64、J57/64；JSON：F0/64、C50/64、J0/64。JSON相对plain内容4胜/49平/11负。恒正类基线33/64。JSON中唯一显式标签63条、1条缺失；1条达到32token上限。原始输出保留，不自动修JSON或事后换prompt。

正式推理阶段12.6911秒；总13.0478秒（从第三方库import后开始，含缓存哈希/加载/推理/评测，不含进程启动与下载）；峰值1252720640字节。首例[1,72]→[1,72,576]→[1,72,49152]；134515008参数。所有输入无截断。checkpoint就是冻结的上游权重，没有训练检查点。

## 产物与限制

predictions.jsonl存每例输出和测量；summary/cases为聚合及事先规则选出的案例；split只存ID；identity含模型/tokenizer、数据、prompt、解码、执行源码哈希和cache_key；environment/stages分别记录成本和阶段状态。源码总清单/ZIP/远端验证另行归档；首次执行后才新增审计与文档，核心执行代码不变。

首次命令漏写--smoke，提前跑了完整开发集，记录在initial_development；随后正确2例试跑，按耗时和内存预算继续正式运行。首次与正式所有预测/评分完全相同（时间除外），但不是两个独立统计实验。缺pip导致环境导出命令失败，改用importlib.metadata枚举安装版本；无依赖变化。详细命令与退出码见commands.json。

结果只支撑一个检查点、一组开发样本和两条固定提示。无人工金标、置信区间、中文能力或SST-2全榜单结论。JSON失败不能证明模型普遍不会JSON；后续只在开发集研究格式监督与标签监督，保留确认集到方案冻结之后。原数据不随包分发，见THIRD_PARTY.md。
