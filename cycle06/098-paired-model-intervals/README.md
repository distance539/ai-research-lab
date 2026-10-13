# 098 两个模型差一点点：配对区间比排行榜更有用吗？

同系列两个真实SFT检查点、SST2同64条开发样本。A=shuffle93，B=shuffle94，模型/数据/提示/解码及原训练预算冻结。只比较固定检查点的样本条件差异，不估计训练算法总体差异，不使用确认集。结果已在093看过，本篇为探索性分析。

## 最小离线复算（不运行模型）

Python3.12.14；完整依赖版本见environment.lock.txt，直接依赖见requirements.txt。统计最小依赖numpy1.26.4、scipy1.13.1。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --out replay-new
python audit.py --summary replay-new/summary.json --out replay-new/audit.json
python test_checks.py
```

预期summary.json、paired_rows.json、bootstrap_histogram.json、stat_environment.json；A58/64、B57/64，差-1/64，配对区间[-3/64,0]。独立错误抽样区间[-8/64,6/64]仅作错误诊断。audit.py不用NumPy/SciPy或生产评分函数，独立解析JSON、重算四格表、二项尾概率和本例经验bootstrap解析分位数。它不证明总体覆盖率，也不构成人工语义审核。

## 真实模型端到端重放

```sh
python prepare.py --cache ./cache
TOKENIZERS_PARALLELISM=false python run.py --infer --smoke --cache ./cache --checkpoints ./checkpoints --out model-smoke-new
TOKENIZERS_PARALLELISM=false python run.py --infer --cache ./cache --checkpoints ./checkpoints --out model-full-new
python audit.py --predictions model-full-new/records.json --summary model-full-new/summary.json --out model-full-new/audit.json
```

checkpoint_lock.json列出两个检查点文件名、SHA256和数值state哈希；模型权重不分发。若没有前篇检查点，先按固定公开代码重建（不是下载一个未发布的权重地址）：

```sh
python fetch_parent.py --out ./parent093
python -m pip install -r ./parent093/requirements.txt
python ./parent093/prepare.py --cache ./cache
TOKENIZERS_PARALLELISM=false python ./parent093/run.py --cache ./cache --out ./results --private ./checkpoints
```

fetch_parent.py已实际匿名获取16个固定提交文件并逐项验证，包含完整自写运行模块、资源锁和配置。上条原版093命令重建全部五组，每组64更新；保留原组名及out目录名results，以复现序列化文件名。098只读取shuffle93/94。093完整真实训练此前已完成；本次重用原检查点，没有再次训练五组。跨硬件、不同依赖的位级相等待人工核验，出现checkpoint或输出不一致时立即停止，不能改哈希伪装通过。正式重建需约6GiB进程内存上限、模型269MB、数据约23MB、五组checkpoint约2.7GB，另计环境及缓存；本篇推理只需两组约1.1GB检查点。

## 实际执行范围

本次macOS/CPUfloat32/eager/4threads：2题×2模型资源试跑后，64题×2模型全部真实生成128输出；逐输出token、标签和prompt SHA与固定093证据完全一致，只有耗时不同。无训练、无few-shot搜索、无更换评分器。128SST2+64新闻确认样本均未推理。正式时间和RSS在formal/environment.json；原始逐行日志formal/stdout.log，命令和退出码commands.json。资源检查共11文件，具体修复史DEVIATIONS.md。

两万次配对bootstrap经SciPy官方实现与直接共享索引实现逐次相等。random_state传入Generator，method显式percentile。独立标准库复算结果formal/independent_audit.json；8项身份/标签/分数/摘要破坏测试。正文不声称95%概率某模型更好、等效或盲测显著性；SST2接口缺少电影分组ID，按行独立只是条件假设。无人工模型输出金标。所有分发文件及其SHA256在MANIFEST.json；公开固定提交见博客入口。
