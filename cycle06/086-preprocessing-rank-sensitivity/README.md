# 086 标题、摘要与截断：预处理能改变方法排名吗？

真实SciFact100条开发查询×50冻结RRF候选×4条件。title256/title512/abstract256/abstract512；同一ms-marco-MiniLM-L6-v2权重与官方BEIR重排。仅改变重排阶段字段及paired token上限，不重新检索。不是完整809查询或官方test benchmark。300确认查询未评测。

## 运行与资源
Python3.12（本次3.12.14）；CPU建议4GB可用内存。模型约92MB，数据ZIP约2.8MB；PyTorch安装空间另计。requirements.txt固定关键依赖，requirements-observed.txt保留完整共享环境版本。首次联网下载，之后可复用同hash缓存。

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --smoke
python audit.py --out smoke-new
```

最小入口执行5查询×10候选×4真实条件。正式运行删除`--smoke`并使用新输出目录，例如`--out full-new`。输出目录必须不存在，避免覆盖。完整运行预算1200秒推理/4GiB峰值RSS，具体实测见results/status.json和environment.json；下载和历史检索耗时不计。无需API密钥，cache为读者显式路径，不依赖作者机器。

只审计已保存数值：`python audit.py --out results`（标准库即可）。这不替代真实推理。制图：`MPLCONFIGDIR=./mpl-cache python plot.py --out results --dest result.png`。数据图只来自summary.json，未使用图像生成模型生成数值。

## 可核查产物
results/runs.json：五套逐候选得分；pairs.csv：20000行输入token数/实际输入IDhash/相关标记/分数与名次；per_query.json与contrasts.json：逐查询指标和配对差；summary.json：四组全量结果、胜平负和token合计；shapes/status/environment/identity：张量检查、阶段状态、成本与缓存身份。候选相同，因此Recall50必须不变。查询token、文档token及三个特殊token之和逐行检查。标题是否有用不能通过只看均值或只看被截断比例决定。

PROTOCOL.md在086推理前登记；resource_decision.json只用试跑时间和RSS决定完整执行。第一次命令漏传--smoke，错误启动full；中断退出130。early_aborted保留其已有输出，early_aborted.log保留脱敏日志，不用于选择条件。之后正确smoke和formal均完整运行。所有试验均为开发性，不称未见验证。

SOURCE_MAP.md说明官方永久文件/行号、许可证、本地修改与独立教学模块。input.lock.json固定085公开commit及输入hash；model.lock.json固定模型/tokenizer全部下载文件；resources.lock.json固定数据。未计算训练成本、GPU显存、总体置信区间或证据句丢失比例。时间有固定运行顺序，不能作为严格性能测试。

## 本次结果

四组nDCG10：title256 0.6596858696；title512 0.6691325921；abstract256 0.6502066829；abstract512 0.6386130835。全部低于固定RRF 0.6709061302。长度效应随标题条件改变方向；交互量0.0210403219为开发性描述，不宣称显著。正式总时418.8804秒，RSS1040351232字节。
