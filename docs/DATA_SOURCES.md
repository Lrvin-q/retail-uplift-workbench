# 数据来源与许可边界

演示默认只用代码即时生成的虚构数据。真实研究使用四个互相独立的数据源，公开作品不重新分发它们的原始文件或逐行派生产物。

|数据|来源|已使用版本及校验|用途与限制|
|---|---|---|---|
|Hillstrom|[原发布地址](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv)、[TFDS 数据卡](https://www.tensorflow.org/datasets/catalog/hillstrom)|64,000行；3,964,977字节；SHA256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`|邮件随机实验；原站可用性不保证，下载须核对哈希，不把第三方同名文件自动当成同版。未把公开可下载等同于代码MIT许可，原始数据不随仓库分发。|
|Criteo uplift|[官方镜像与数据卡](https://huggingface.co/datasets/criteo/criteo-uplift)、[原作者说明](https://arxiv.org/abs/2111.10106)|v2.1；13,979,592行；311,422,618压缩字节；SHA256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`|数据卡标注CC BY-NC-SA 4.0，限制独立于代码许可；商业使用须自行核对授权。结果相关抽样限制因果解释。|
|RetailHero uplift|[原挑战仓库](https://github.com/retailhero/retailhero-uplift)|交易多表工程，原工程处理45,786,568行购买明细|用于粒度处理与特征工程；不主张因果识别，取得和使用遵循来源条款，仓库不附原始数据。|

这些链接来自已保存的项目来源记录，本次公开整理未重新下载数据，也未重新核验远端许可是否变化；使用前应阅读来源的现行条款。本仓库MIT许可不覆盖外部数据、论文、依赖库或其商标。

原始文件应放 `data/raw/`，该目录整体忽略；模型与客户级结果放 `artifacts/` 或忽略目录。不能把脱敏行号、匿名ID或模型分数简单视为适合公开的样本，因此本版一并排除。


## Starbucks 随机促销练习（新增汇总）

来源：[Udacity 官方目录](https://github.com/udacity/DSND_Term2/tree/master/portfolio_exercises/Experiments)与[官方 notebook](https://github.com/udacity/DSND_Term2/blob/master/portfolio_exercises/Experiments/Starbucks.ipynb)。使用官方训练 84,534 行和测试 41,650 行，匿名 V1—V7 为特征，购买为结果。原文件身份、哈希和本轮数据划分见[冻结协议](../reports/evidence/starbucks/protocol_v1.json)。

公开作品仅增加作者的聚合结果、协议、自绘图与报告，不转发原始行数据、客户分数、模型或第三方论文。公开可下载不等于可以按本仓库 MIT 重新授权；下载和使用前仍须核对来源条款。本案例区别于常见的另一套模拟 Starbucks 行为数据，不能混为一谈。
