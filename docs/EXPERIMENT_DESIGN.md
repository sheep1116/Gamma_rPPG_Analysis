# 实验设计摘要

## 因素

- 数据集：UBFC2、LGI_PPGI、PURE；
- Gamma：1.0 至 2.8；
- 方法：CHROM、POS、GREEN、LGI、PBV、OMIT；
- 处理位置：pre-ROI 主实验、post-ROI 固定掩膜控制实验。

## 控制原则

- 不生成 Gamma 视频；
- PURE 正式实验统一使用 AVI；
- AVI 完整性检查独立运行；
- post-ROI 固定原图关键点和 ROI；
- 每个数据集独立统计；
- PURE 运动类型保留元数据但不拆成独立数据集；
- 主比较使用完整配对样本；
- ROI 敏感性规律只基于 pre-ROI 的 Gamma=1.0 特征。
