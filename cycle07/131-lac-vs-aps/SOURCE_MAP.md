# 源码地图 · 2026-11-16

- DistilBERT 两列 logits。Transformers 4.49.0 (Apache-2.0) [分类头与 forward](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/distilbert/modeling_distilbert.py#L918-L991)。model_run.py 调用官方类，保留配置、批次形状。许可证随包。

- [模型与分词器](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13) Apache-2.0。[SST-2 validation](https://huggingface.co/datasets/stanfordnlp/sst2/tree/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb) 卡片许可 unknown，只分发派生分数与样本身份。原句、权重由有需要的读者自行下载。

TorchCP 固定版本 627455f07cb1ab9ef39e403ae51534b6fbdf200c。以下代码逐字节取自上游，原版权声明保留，适用 LGPL v3。包内附完整 LGPL 与 GPL 文本；本篇自写代码适用 MIT。run.py 在临时目录恢复三个模块的原始导入路径，直接执行官方类和 common.py 的阈值函数。

- aps.py 对应 [官方文件](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/classification/score/aps.py#L36-L98)
- lac.py 对应 [官方文件](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/classification/score/lac.py#L43-L100)
- base.py 对应 [官方文件](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/classification/score/base.py#L11-L26)
- common.py 对应 [官方文件](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/utils/common.py#L40-L74)
- TorchCP-LICENSE.txt 对应 [官方文件](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/LICENSE)
- TorchCP-GPL.txt 对应 [官方文件](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/LICENSE.GPL)

run.py 的闭区间判定、集合和配对计数，audit.py 的标量代数与端点诊断均由作者独立编写。模型加载与下载模块沿用[上一篇固定版本](https://github.com/distance539/ai-research-lab/tree/45428db315a0096c2a30ba63af48c0197b6a6969/cycle07/130-subgroup-coverage)。
