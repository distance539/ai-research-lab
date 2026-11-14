# 公式、实现与来源 · 2026-11-14

- DistilBERT 两列 logits。Transformers 4.49.0 (Apache-2.0) [分类头与 forward](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/distilbert/modeling_distilbert.py#L918-L991)。model_run.py 调用官方类，保留配置、批次形状。许可证随包。
- TorchCP (SUSTech-ML)，LGPL v3，固定 627455f07cb1ab9ef39e403ae51534b6fbdf200c。[LAC 对真类与全类取分](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/classification/score/lac.py#L57-L100)，[有限样本秩与 kthvalue](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/utils/common.py#L40-L74)，[预测集生成调用](https://github.com/ml-stat-Sustech/TorchCP/blob/627455f07cb1ab9ef39e403ae51534b6fbdf200c/torchcp/classification/predictor/split.py#L96-L111)。run.py 独立教学实现，使用 Python 双精度、精确有理数取秩、显式排序与列表表示，未复制或安装 TorchCP。上游源码及许可字节摘要见 upstream_verification.json。
- [模型与分词器](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13) Apache-2.0。[SST-2 validation](https://huggingface.co/datasets/stanfordnlp/sst2/tree/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb) 卡片许可 unknown，只分发派生分数与样本身份。原句、权重由有需要的读者自行下载。

- 理论来源为 Angelopoulos 与 Bates 的[教程第一节](https://arxiv.org/html/2107.07511)。run.py 独立实现嵌套前缀、越界分支与诊断对照，audit.py 用差值比较和严格小于的计数验算成员资格。二类输出顺序与模型标签映射一致。未运行 TorchCP 基准。
- 自写推理及资源模块直接沿用[上一版 MIT 源码](https://github.com/distance539/ai-research-lab/tree/1aa811c3e0a5a5c18b9078d6fa4f2fa326d5e46d/cycle07/128-marginal-coverage)，必要模块与输入均随包。哈希对应输入、程序和推理配置，无提示词与解码参数。
