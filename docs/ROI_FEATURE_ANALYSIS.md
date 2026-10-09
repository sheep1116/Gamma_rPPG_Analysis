# ROI 颜色、亮度与敏感视频分析

本分析只读取 `pre_roi_gamma` 结果。用于解释“哪些视频适合 Gamma”的特征必须来自 Gamma=1.0 的原始 ROI，避免用 Gamma 变换后的亮度循环解释性能变化。

## 基线特征

### RGB

- R/G/B 均值与中位数；
- R/G、B/G、R/B；
- R-G、G-B；
- 三通道均值范围。

### 编码 RGB luma

\[
Y'=0.2126R+0.7152G+0.0722B
\]

保存均值、中位数、标准差、P05/P25/P75/P95/P99、IQR 和 P95-P05 动态范围。

### 明暗与饱和

统计小于 5、10、20，以及大于 220、235、245、250 的像素比例。饱和和暗像素比例在 pyVHR 有效像素阈值筛选前计算。

### 时间稳定性

统计逐帧 ROI RGB/luma 均值的标准差、变异系数、相邻帧变化均值和 P95。

### 空间均匀性

统计 ROI 内 luma 标准差、IQR、高亮/阴影比例、左右和上下亮度差。

### 有效像素

分别记录几何 ROI 与经过 pyVHR RGB 阈值筛选后的有效像素，包括有效比例、因过暗/过亮被拒绝的比例和有效像素 luma。

## 运行

```bash
bash scripts/analyse_pre_roi_sensitivity.sh
```

该步骤只做后处理；调整敏感性阈值或统计设置不需要重新执行 rPPG 主实验。

## 正向样本诊断包

默认最多为每个数据集前 10 个多算法正向样本重新提取基线和最佳 Gamma 信号，生成 `rgb_timeseries.png`、`bvp_spectrum.png` 和 `roi_examples.png`；其余样本仍生成性能曲线和表格。该数量由 `max_signal_diagnostic_cases` 控制。
