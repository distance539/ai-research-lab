# 093 — 多种子与配对结果报告

真实SmolLM2-135M-Instruct，32SST2训练、64SST2+32AGNews开发。两次固定顺序控制、三个类内重排种子；共320更新、960生成。只选定091的target_only配置，改变训练顺序来源，不做新超参数搜索。192确认样本未推理。

## 最小运行

实测macOS26.6 arm64、Python3.12.14、torch2.6.0、transformers4.49.0、CPUfloat32/eager/4threads。依赖精确版本在requirements.txt和environment.lock.txt。建议至少6GiB可用进程内存；模型269MB、公共数据约23MB，五个检查点各538MB，正式加smoke约5.4GB检查点空间，另需环境和缓存空间。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --private private-new --smoke
python audit.py --cache ./cache --out smoke-new
```

预期：smoke10步更新/48生成，保存split/identity/rows/steps/budget/checkpoints/predictions/summary/paired/statistics/environment/stages；private保存完整权重和输入trace。正式去掉--smoke，使用全新out目录，五组各64步、每样本4次。源码不依赖前篇本地路径。跨硬件和不同软件位级一致性待人工核验。

## 冻结条件与统计对象

完整预注册在PROTOCOL.md，配置在config.json，20资源文件含数据/model/tokenizer精确revision与SHA256在resources.lock.json，官方代码映射在SOURCE_MAP.md。SST选择沿用089/090和088开发划分，AG News沿用091平衡开发组；test来源子集明确用作开发，未当blind test。所有数据只规范化精确去重，未证明不同电影/新闻事件独立。

五组均从同一原始已指令训练模型开始，134515008参数，SGD.001、clip1、batch2、64步、1024监督token、10556输入token。assistant答案至真实EOS8token/行，prompt/pad=-100；EOS与pad共ID2但按长度保留。无截断，greedy32。fixed93/fixed94只改torch seed而不改顺序；无dropout/随机解码。shuffle93/94/95用Python RNG每轮分别洗两类16行再配对，保持每batch一正一负。重排改变padding和配对，不声称等FLOPs。

主要报告JSON joint差值；另报显式标签内容代理、plain和news。三重排运行均值/样本SD仅描述这三个训练顺序，不是总体区间。逐固定checkpoint使用配对bootstrap10000次、seed9300、2.5/97.5线性百分位；SST按行、news按金标签类内抽样。开发区间仅条件敏感性，无选择偏差修正，无独立电影分组信息；禁止把3×64当192独立测试样本。

## 实际结果

| 状态 | plain F/C/J（64） | JSON F/C/J（64） | news F/C/J（32） |
|---|---|---|---|
| before |64/57/57|0/50/0|30/8/8|
| fixed93 |64/33/33|64/58/58|29/8/8|
| fixed94 |64/33/33|64/58/58|29/8/8|
| shuffle93 |64/33/33|64/58/58|29/8/8|
| shuffle94 |64/33/33|64/57/57|29/8/8|
| shuffle95 |64/33/33|64/58/58|29/8/8|

F严格格式，C显式唯一标签与数据标签一致，J两者皆对；没有人工模型输出金标。三个重排JSON J均值90.1042%、SD0.9021个百分点；JSON C差值均值+11.9792pp。shuffle93 C配对12胜48平4负，条件区间[0,25]pp；shuffle94为11/49/4，[0,21.875]pp。所有plain下降37.5pp，区间[-53.125,-21.875]pp。新闻[0,0]仅因本批每行差都为0，低基线8/32不能证明能力保留或等效。

formal267.899436秒（imports后，含资源核验/加载/训练/保存/生成/统计），峰值RSS2961309696B。smoke18.305181秒、RSS2617606144B，预估439.324336秒<1200秒，6GiB预算通过后正式运行。GPU内存未测。五个真实checkpoint保存于private，文件哈希和数值state哈希均记录。fixed控制权重数值一致；三个shuffle权重各异。

## 审计与分发

python audit.py --cache ./cache --out results 可离线核对资源、掩码、预测及配对统计。完整原始输出先审计，export_evidence.py将可能复述原新闻的长回答留在private，仅在公开记录保留哈希/派生评分，并明确不重复解析删节行。不得把删节行声称为公开可见的原回答。读者自行运行可产生自己的完整输出。原数据许可unknown，不分发parquet/原句/prompt/inputIDs；不含权重、环境、缓存或凭据。许可说明THIRD_PARTY.md。统计脚本输出不是论文结果；两篇统计论文仅提供方法背景。
