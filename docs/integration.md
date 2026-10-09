# 项目整合与迁移

2026-10-09，以 Gamma_rPPG_Analysis 为主仓库整合 ROI_gamma_mechanism_analysis。

## 备份与成果保留

整合前完整备份位于服务器 `~/Project_backups/gamma_integration_20261009/original_projects.tar.gz`。相邻的 `.sha256`、`original_files.sha256` 和 `archive_contents.txt` 分别记录压缩包校验、原文件校验和归档清单。备份包括原代码、配置、文档、缓存与历史日志，不提交到 GitHub。

真实视频、数据集、`~/Result/GammaBrightness_rPPG_v5` 及其机制分析结果不移动、不删除。验证输出写入备份目录中的独立 validation 子目录。原 ROI 项目目录保留启动兼容入口；完整原目录保存在备份目录中。

## 导入与运行迁移

| 原入口 | 新实现 |
|---|---|
| src.run_gamma_experiment | gamma_rppg.experiments.pre_roi |
| src.run_post_roi_analysis | gamma_rppg.experiments.post_roi |
| src.quick_test_methods | gamma_rppg.experiments.quick_test |
| src.summarize_results | gamma_rppg.analysis.summary |
| src.analyse_pre_roi_sensitivity | gamma_rppg.analysis.sensitivity |
| src.config_io / src.result_store | gamma_rppg.common.config / result_store |
| src.gamma_signal_processing / src.image_math | gamma_rppg.signal.processing / image_math |
| src.pure_directory_dataset | gamma_rppg.datasets.pure |
| ROI run_analysis.py | gamma_rppg.analysis.roi_mechanism.cli |

兼容模块委托给唯一实现，不复制实验逻辑；包含私有辅助函数的旧导入也保留。现有 Shell 文件名、实验 YAML 文件名和 CSV schema 保持可用。

`full_experiment.yaml`、`quick_test.yaml`、`default.yaml` 是配置别名。共同数据路径提取到 `datasets.yaml`，快速配置只覆盖样本范围。配置支持明确的环境占位符；默认路径保持现有服务器布局，GPU 编号由 1 修正为实际的 0。

Holm 校正的四份实现合并到 `common/statistics.py`。实验模块保留返回 list 的 API，ROI 模块保留返回 ndarray 的 API；有效 p 值结果不变，缺失值保持 NaN 并不计入有效检验族。均值与中位数 bootstrap 的估计对象、种子和实现保留各自语义，没有混合。

ROI 分析原来保留所有已读数据集的未使用缓存，现已移除，逐数据集处理时释放不再需要的表。

## 文档整理

原总览、框架、使用顺序合并到 README 和 architecture 文档；ROI README 的研究定义与图表说明保留在 roi_mechanism 文档，设计说明保留在 ROI_MECHANISM_DESIGN 文档。旧构建验证、v4 迁移和 ROI 验证记录移入 `docs/archive/`，不作为当前验收结论。有效方法说明、配置、测试和图表生成程序均保留。
