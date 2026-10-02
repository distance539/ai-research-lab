# 089 — SFT 模板与标签掩码审计

真实 SmolLM2-135M-Instruct、SST-2 官方 train 中8条、088开发集前8条。全序列与仅回答两组各4步全参数SGD；这是小预算机制/管线诊断，不是完整SFT或GLUE结果。没有调参/确认集推理。

## 最小运行

实测 Python3.12.14、macOS26.6 arm64、torch2.6.0、Transformers4.49.0、CPU float32、4线程。直接依赖见 requirements.txt；完整复用环境版本见 environment.lock.txt。建议6GiB可用内存、4GB磁盘（安装环境另计）；模型资源约273MB，两组本地最终检查点约1.08GB。无需密钥。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python prepare.py --cache ./cache
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python run.py --cache ./cache --out smoke-new --private ./private-new --smoke
python audit.py --cache ./cache --out smoke-new
```

去掉`--smoke`，换成新输出目录可运行完整8条/4步实验。输出目录拒绝覆盖；private目录存真实逐token可逆数据和最终模型state_dict，请勿加入公共分发。下载器复用逐文件SHA256正确的缓存；不存在时获取固定revision，不追main。首次准备需互联网，训练可离线。

`results/` 是完整实验、`smoke/` 是先行2条/1步试跑；二者各含原始stdout、预测、每步loss/梯度/形状、掩码位置、截断检查、聚合、阶段、版本身份和检查点哈希。`audit.py`从原数据重新编码，独立核对监督位置/数量和全部F/C/J。`role_audit.py`补充真实角色边界和错误EOS掩码的反例；`role_audit.json`是实际运行输出。

## 受控协议与结果

先读PROTOCOL.md，配置逐字见config.json。8条训练样本按SHA256('89:'+idx)选取、正负各4；验证按088的SHA256('88:'+idx)固定前8条，沿用其64开发/128保留确认划分。训练样本与开发及保留集规范化完全相同文本交集为0；不证明上游无污染或不存在近重复。官方train包含影评片段，不能把idx一概当独立完整影评。

两组输入、初始权重、顺序、batch2、4步、SGD lr0.001、无动量/衰减、clip1相同。模型用eval模式关闭随机模块但保留autograd，全134515008参数参与优化；不是torch.inference_mode。输入右侧padding，PAD=EOS=2；必须由真实长度/attention屏蔽padding，保留真实EOS。回答的首token由prefix-1位置预测，不预先移位labels。最后EOS之后模板换行ID198被双方删除。单轮对话模板前缀逐ID核验，不声称适配任意多轮模板。

每组实际输入684token；full有效监督676，assistant64，二者不是等监督量/损失尺度的公平性能比较。所有梯度都触发clip；梯度方向及数值不同。监督缩减不等于前向上下文缩减或训练成本同比降低。四批样本不同，不能把各批loss首尾解释为同一批的下降。

独立CE与官方loss最大差4.7684e-7，所有忽略位置logit直接梯度为0，提示输入embedding梯度非零。两组首层q_proj最大变化分别2.1085e-6/2.4959e-6。长度64会让8条回答全部丢失；96丢1条EOS；128/256无丢失；正式无截断。首批[2,92]→[2,92,49152]。

8条开发：训练前plain F8/C6/J6、JSON F0/C6/J0；两组更新后plain F8/C7/J7、JSON F0/C6/J0。两组都只把idx481的plain从negative改成positive；JSON原始输出全部不变。单样本变化不支持总体收益或掩码优劣。全部48条生成保存。F格式、C唯一显式情感词匹配来源标签的代理、J二者同时满足；不是人工语义金标。

正式总9.94299秒（第三方import后起算，含校验/加载/训练/保存/推理/评测），两组训练1.43516/1.29538秒，峰值RSS2661089280字节；GPU内存未测量。试跑5.51837秒/RSS2333704192；资源预估通过后原配置正式运行。预算900秒/6GiB，跨平台位级一致性待人工核验。

## 继承与许可

自写prepare/scoring与提示协议来自088公开固定提交4ed2968d7ca40b222d419372728a888529482e43；所需模块已完整包含，无机器绝对路径依赖。SOURCE_MAP.md指向真实固定官方源码；运行直接调用已安装官方实现，无复制第三方可执行源代码。训练权重/数据/环境/缓存不在ZIP；SST-2许可unknown，原句与可逆输入token仅本地保留。第三方许可和自写源码许可见THIRD_PARTY.md。缓存键包括版本、数据、模板、配置、解码、拆分和执行源码哈希。审计与文档在主实验后新增，核心执行源码哈希保持一致。
