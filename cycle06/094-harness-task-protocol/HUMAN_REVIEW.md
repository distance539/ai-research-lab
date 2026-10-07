# 094 真人算术复核

状态：待用户实际复核。机器独立重算不代表真人审核；不是人工语义金标。

规则：对比两个候选总对数似然，取较大值。choice0=negative，choice1=positive。预测等于gold记1，否则0；四题准确率为得分之和除以4。

| idx | negative | positive | gold | 人工预测 | 人工得分 |
|---|---:|---:|---:|---|---|
| 400 | -3.8741276264 | -4.7651276588 | 1 | | |
| 615 | -6.3516988754 | -7.2886848450 | 0 | | |
| 106 | -2.9267070293 | -3.4737651348 | 1 | | |
| 650 | -1.3008216619 | -2.9927010536 | 0 | | |

人工准确率：待填。

请回报四个预测、正确数与准确率；无需公开任何影评原文。

Human response not yet received. Official functional validation will proceed in parallel; do not claim human-before-official sequence or quality passed until genuine review is recorded. No score-based configuration selection.
