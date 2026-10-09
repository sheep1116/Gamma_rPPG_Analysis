# ROI–Gamma 正向响应机制分析项目

该模块已整合到主仓库，只读取既有 v5 格式的结果 CSV，不重新执行人脸检测和 rPPG 提取。目标是回答：

> ROI 像素统计信息为什么会使某些视频在 Gamma 变化后出现 MAE 下降，即正向响应？

## 1. 分析框架

项目把问题拆成四层：

1. **预测层**：Gamma=1.0 时的 ROI 特征，是否与该视频—方法组合的最佳 MAE 改善量相关？
2. **分组层**：`positive_sensitive` 与 `stable` 视频的 ROI 特征是否存在系统差异？
3. **机制层**：同一个视频内，Gamma 引起的 ROI 特征变化是否与 MAE benefit 同步变化？
4. **稳定性层**：高 Gamma 下出现算法失败的视频，是否具有特殊的基线 ROI 像素结构？

统一定义：

```text
MAE benefit(gamma) = MAE(gamma=1.0) - MAE(gamma)
```

因此：

- `MAE benefit > 0`：Gamma 改善；
- `MAE benefit < 0`：Gamma 恶化。

## 2. 输入数据

将 v5 的结果根目录传给程序。目录应包含：

```text
GammaBrightness_rPPG_v5/
└── pre_roi_gamma/
    ├── UBFC2/
    ├── LGI_PPGI/
    └── PURE/
```

每个数据集目录至少需要：

```text
pre_roi_sensitivity_with_features.csv
pre_roi_response_curves.csv
pre_roi_roi_feature_changes.csv
brightness_metrics.csv
raw_results.csv
```

程序还会在存在时使用：

```text
pre_roi_cross_method_consensus.csv
pre_roi_baseline_roi_features.csv
quality_control.csv
```

## 3. 环境和配置

CPU 分析依赖统一由主仓库的 `requirements.txt` 管理，无需 CUDA 或 pyVHR。安装步骤见 [主 README](../README.md)。默认配置为 `configs/roi_mechanism.yaml`，保留原随机种子和统计选项。

## 4. 运行

在主仓库根目录执行：

```bash
bash scripts/run_roi_mechanism_analysis.sh "$HOME/Result/GammaBrightness_rPPG_v5"
```

指定独立输出目录：

```bash
python -m gamma_rppg.analysis.roi_mechanism.cli \
  --config configs/roi_mechanism.yaml \
  --result-root "$HOME/Result/GammaBrightness_rPPG_v5" \
  --output "$HOME/Result/ROI_Gamma_Mechanism_new"
```

默认输出是 `<result-root>/roi_gamma_mechanism_analysis`。再次运行会更新该目录中的分析文件；需要保留旧版本时指定新的输出目录。`python run_analysis.py` 与 `bash scripts/run_analysis.sh` 为兼容入口。

长任务可先运行 `tmux new -s roi_gamma_analysis`，执行分析后按 Ctrl+B、D 脱离会话。

## 5. 输出结构

```text
roi_gamma_mechanism_analysis/
├── run_manifest.json
├── tables/
│   ├── baseline_feature_associations.csv
│   ├── positive_vs_stable_feature_tests.csv
│   ├── within_video_gamma_mechanism_associations.csv
│   ├── gamma_response_group_summary.csv
│   ├── pixel_composition_group_summary.csv
│   ├── failure_feature_tests.csv
│   └── selected_detailed_features.csv
├── reports/
│   └── ANALYSIS_SUMMARY.md
└── figures/
    ├── 01_correlation_forest/
    ├── 02_feature_scatter/
    ├── 03_positive_vs_stable/
    ├── 04_gamma_mechanism/
    ├── 05_heatmaps/
    ├── 06_pixel_composition/
    ├── 07_failure_analysis/
    └── 08_cross_method_consensus/
```

## 6. 图表说明

### 6.1 相关性森林图

文件夹：

```text
figures/01_correlation_forest/
```

每个数据集—方法一张图：

- 点：Spearman 相关系数；
- 横线：bootstrap 95% 置信区间；
- 圆点：Holm 校正后显著；
- 方点：未达到校正后显著。

回答：

> 哪些 Gamma=1.0 的 ROI 特征更容易预测后续正向改善？

### 6.2 方法特异散点图

文件夹：

```text
figures/02_feature_scatter/<dataset>/<method>/
```

- 横轴：基线 ROI 特征；
- 纵轴：最佳 MAE benefit；
- 点按响应类型区分；
- 图内标注 rho、95% CI 和 `p_Holm`。

用于检查显著关系是否由少数离群点驱动。

### 6.3 正向敏感与稳定型对比

文件夹：

```text
figures/03_positive_vs_stable/
```

每张图包含箱线图和全部视频散点，并标注：

- Cliff's delta；
- Holm 校正后的 Mann–Whitney p 值。

回答：

> 正向响应视频在原始 ROI 像素结构上与稳定视频有何不同？

### 6.4 Gamma–ROI–MAE 联合机制曲线

文件夹：

```text
figures/04_gamma_mechanism/
```

上图为：

```text
MAE benefit 随 Gamma 的变化
```

下图为：

```text
对应 ROI 特征相对 Gamma=1.0 的变化
```

分别展示 `positive_sensitive` 和 `stable` 组的视频级中位数及 bootstrap 95% CI。

回答：

> 正向组从哪个 Gamma 开始改善？ROI 有效像素、暗像素、亮度波动等是否在相同 Gamma 区间发生变化？

### 6.5 样本×Gamma 热图

文件夹：

```text
figures/05_heatmaps/
```

左图为每个视频的 MAE benefit，右图为同一视频的 ROI 特征变化；视频排序完全一致。

用于定位：

- 哪些具体视频驱动总体规律；
- 改善是否只出现在窄 Gamma 区间；
- ROI 特征变化与 MAE 改善是否在同一视频同步。

### 6.6 有效/过暗/过亮像素组成

文件夹：

```text
figures/06_pixel_composition/
```

分别展示正向组和稳定组中：

- `valid_pixel_ratio_mean`；
- `rejected_dark_pixel_ratio`；
- `rejected_bright_pixel_ratio`；

随 Gamma 的变化。

这是现有 CSV 中最接近“像素组成”的图。它不能替代逐像素直方图，但可以直接说明 Gamma 是否让大量 ROI 像素跌出有效范围。

### 6.7 失败状态热图

文件夹：

```text
figures/07_failure_analysis/
```

每种方法包含：

- 样本×Gamma 成功状态热图；
- 成功率随 Gamma 的变化图。

适合分析 PURE 中 PBV、CHROM 等方法在高 Gamma 下的失效问题。

### 6.8 多算法一致性图

文件夹：

```text
figures/08_cross_method_consensus/
```

每个视频只保留一行，按：

- `multi_method_positive`；
- `single_method_positive`；
- `mostly_stable`；
- `mixed_response`；
- `mostly_negative`；

比较关键 ROI 特征，避免同一个视频因六种方法重复计数。

## 7. 统计方法

### 基线特征分析

- Spearman 相关；
- 以视频为单位 bootstrap 95% CI；
- 在每个数据集—方法内进行 Holm 多重比较校正。

### 正向组与稳定组比较

- Mann–Whitney U；
- Cliff's delta；
- Holm 校正。

### 视频内机制分析

对于每个视频，先在多个 Gamma 点之间计算：

```text
Spearman(ROI feature delta, MAE benefit)
```

再在视频间汇总：

- 视频内 rho 的中位数；
- 以视频为单位 bootstrap 95% CI；
- Wilcoxon 检验这些视频内 rho 是否系统偏离 0；
- Holm 校正。

这种做法避免把同一视频的多个 Gamma 点错误当作独立样本。

## 8. 默认重点特征

项目默认分析以下代表性指标：

```text
roi_luma_mean
roi_luma_p95
roi_luma_p95_minus_p05
temporal_luma_cv
temporal_g_cv
temporal_g_std
p95_frame_to_frame_luma_change
mean_frame_to_frame_luma_change
spatial_luma_std
rejected_bright_pixel_ratio
rejected_dark_pixel_ratio
valid_pixel_ratio_mean
bright_ratio_235
sat_ratio_245
r_over_g
b_over_g
```

这些指标覆盖：

- 原始亮度；
- 高亮和动态范围；
- 时间波动；
- 帧间变化；
- 空间不均匀性；
- 有效、过暗和过亮像素比例；
- RGB 通道关系。

可以在 `configs/roi_mechanism.yaml` 中修改。

## 9. 当前数据能够和不能够回答的问题

### 能够回答

- 哪类 ROI 汇总特征与正向响应相关；
- 正向和稳定视频的 ROI 特征是否不同；
- 随 Gamma 增大，有效像素和亮度波动如何变化；
- 这些变化是否与 MAE 改善同步；
- 高 Gamma 失败是否集中在特定视频和方法。

### 不能直接回答

现有 CSV 不能恢复：

- 0–255 的完整 ROI 像素直方图；
- 每帧逐像素空间位置；
- 完整 RGB 时间序列；
- BVP 频谱。

因此，本项目建立的是统计关联和候选机制。要进一步证明因果，需要对少量典型视频重新读取原始帧并保存逐帧 ROI 像素分布。
