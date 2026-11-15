# 公式、实现与来源 · 2026-11-15

- DistilBERT 两列 logits。Transformers 4.49.0 (Apache-2.0) [分类头与 forward](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/distilbert/modeling_distilbert.py#L918-L991)。model_run.py 调用官方类，保留配置、批次形状。许可证随包。
- TorchCP (SUSTech-ML)，LGPL v3，固定 627455f07cb1ab9ef39e403ae51534b6fbdf200c。[LAC 对真类与全类取分](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/classification/score/lac.py#L57-L100)，[有限样本秩与 kthvalue](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/utils/common.py#L40-L74)，[预测集生成调用](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/classification/predictor/split.py#L96-L111)。run.py 独立教学实现，使用 Python 双精度、精确有理数取秩、显式排序与列表表示，未复制或安装 TorchCP。上游源码及许可字节摘要见 upstream_verification.json。
- [模型与分词器](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13) Apache-2.0。[SST-2 validation](https://huggingface.co/datasets/stanfordnlp/sst2/tree/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb) 卡片许可 unknown，只分发派生分数与样本身份。原句、权重由有需要的读者自行下载。


- 边际与分组覆盖参见 Angelopoulos 与 Bates 教程第一节、第四节。run.py 的分层计数、audit.py 的二元差值秩验证为独立教学实现。未对各组单独校准。组内覆盖为命中次数除以组内句数，无推断性统计检验。
- 推理、下载和锁文件沿用作者前一篇的 [MIT 实现](https://github.com/distance539/ai-research-lab/tree/352668f5f607e7adbbb9cbcaf2029f9e79138212/cycle07/129-finite-sample-rank)。必要代码已随包，分发不含权重或原文。
