# 运行与失败记录

2026-09-25；命令完整参数、出口状态、耗时在各results/*/status.json，原始stdout/stderr在同名.log。

1. 系统Python3.9无依赖，bundled Python3.12可用；无现成Java。复用bundled Python，在项目缓存建立词项专用venv，安装固定依赖；官方ES7.17.9约292MiB含Java19.0.2。官方SHA512核验通过，SHA256进入资源锁。
2. 默认shell代理127.0.0.1:32221不可达；沙箱直连DNS失败。获工具网络权限后，官方资源下载成功。最初git clone慢，先用API核验commit并获取codeload固定提交，后git完成且HEAD一致。
3. 默认沙箱禁止服务绑定127.0.0.1:19382，退出1；获本机服务执行权限后启动成功。`evidence/es_combined.log`含失败与成功，`es_stdout.log`保存成功服务启动。
4. 依赖探索：elasticsearch7.9.1 + numpy2.3.5导入时报np.float_删除。固定numpy1.26.4/scipy1.13.1解决，不改官方客户端。
5. results/smoke退出1：原始摘要空白不同。记录1055条ID；归一化后5183文本相等，检索输入仍用原BEIR字节。
6. results/smoke2退出1：当前官方BM25Search继承的BaseSearch要求两个dense方法。compat.py仅显式拒绝dense APIs；源文件保持哈希一致。
7. results/smoke3退出0：5183文档、5真实开发查询。之后新增索引复用全文验证分支，未改变排序协议。
8. results/development退出0：全部809开发查询。预注册一个配置，无指标驱动改参；raw_count为808×101与1×27；保留80827候选。下载、预处理、建索引、检索、评测阶段均有状态。
9. 独立audit退出0：809查询，nDCG/Recall与pytrec_eval最大差1.1102230246251565e-16；独立MRR与聚合重放一致。保存results/development/audit.json。缺查询、空返回、第二位命中、同分fixture都检查。
10. 确认集只做ID/字段/标注映射，不检索、不分类错误、不选配置；后续不能把结构已读数据称外部封存盲测。图像/源码包验收记录放文章根目录acceptance.json。

未测服务峰值RSS；没有GPU、神经训练、原论文完整benchmark或人工小金标。5-query smoke指标不用于任何参数选择。服务已预热，报告耗时排除下载与启动。结果可复核，但数据标注完整性和跨机器排序边界未由这些检查证明。
