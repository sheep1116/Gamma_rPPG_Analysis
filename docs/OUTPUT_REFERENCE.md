# 输出文件说明

## 顶层结构

```text
GammaBrightness_rPPG_v5/
├── diagnostics/PURE_AVI_CHECK/
├── pre_roi_gamma/
│   ├── UBFC2/
│   ├── LGI_PPGI/
│   └── PURE/
├── post_roi_analyse/
│   ├── roi_cache/
│   ├── UBFC2/
│   ├── LGI_PPGI/
│   └── PURE/
└── case_mechanism_figures/
    ├── case_index.csv
    ├── UBFC2/
    ├── LGI_PPGI/
    └── PURE/
```

不生成跨数据集合并目录。

## pre_roi_gamma

- `raw_results.csv`：视频—Gamma—方法任务结果；
- `brightness_metrics.csv`：ROI 颜色、luma、空间、时间和有效像素指标；
- `summary_by_gamma.csv`：全部成功任务；
- `summary_paired_by_gamma.csv`：全部 Gamma 共同成功的视频；
- `success_rate_by_gamma.csv`；
- `paired_statistical_tests.csv`；
- `per_video_delta.csv`、`per_video_best_gamma.csv`；
- `quality_control.csv`；
- `pre_roi_sensitivity_labels.csv`；
- `pre_roi_cross_method_consensus.csv`；
- `pre_roi_baseline_roi_features.csv`；
- `pre_roi_feature_improvement_correlation.csv`；
- `pre_roi_roi_feature_changes.csv`；
- `pre_roi_feature_change_vs_mae_delta.csv`；
- `pre_roi_feature_group_tests.csv`；
- `sensitivity_cases/`：案例摘要、ROI 特征、Gamma 曲线；前若干案例包含 RGB 时序、BVP 频谱和 ROI 对照图；
- `figures/`、`sensitivity_figures/`、`logs/`。

## post_roi_analyse

- `post_roi_raw_results.csv`；
- `post_roi_quality_control.csv`；
- `post_roi_summary_by_gamma.csv`；
- `post_roi_summary_paired_by_gamma.csv`；
- `post_roi_success_rate.csv`；
- `pre_vs_post_common_task_results.csv`；
- `pre_vs_post_complete_case_samples.csv`；
- `pre_vs_post_paired_results.csv`；
- `pre_vs_post_summary.csv`；
- `pre_vs_post_statistical_tests.csv`；
- `pre_vs_post_baseline_check.csv`；
- `pre_vs_post_success_rate.csv`；
- `figures/`。

## case_mechanism_figures

- `case_index.csv`：全部入选典型案例的索引；
- `README.md`：英文图表说明和案例列表；
- `<dataset>/<category>/<sample>__<method>/case_metadata.csv`；
- `<dataset>/<category>/<sample>__<method>/roi_luma_histogram.png`；
- `<dataset>/<category>/<sample>__<method>/roi_luma_histogram_stats.csv`；
- `<dataset>/<category>/<sample>__<method>/rgb_timeseries.png`；
- `<dataset>/<category>/<sample>__<method>/bvp_spectrum.png`；
- `<dataset>/<category>/<sample>__<method>/signal_extraction_status.csv`；
- `<dataset>/<category>/<sample>__<method>/bvp_spectrum_status.csv`。

图表标题、坐标轴、图例均使用英文，便于直接用于论文排版。若某个失败样本在目标 Gamma 下无法生成 BVP，脚本会保留状态文件和可生成的图，不中断整批任务。

## roi_gamma_mechanism_analysis

原独立项目现为本仓库的 ROI 机制模块，读取 pre_roi_gamma 表。输出七张统计表、报告、run_manifest.json、resolved_config.yaml 和八类图表。结构与字段详见 [roi_mechanism.md](roi_mechanism.md)。聚合文件保留 dataset 标签，各数据集内部检验独立计算。
