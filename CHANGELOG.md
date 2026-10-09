# Changelog

## 0.7.0 — 2026-10-09

- 将 ROI 机制分析纳入 gamma_rppg.analysis.roi_mechanism，保留原 CLI 和结果 schema。
- 按公共工具、数据、信号、实验、分析、诊断、图表划分模块；src 仅提供兼容转发。
- 合并配置加载、四份 Holm 校正和依赖列表，集中数据路径，保留旧配置别名。
- 默认 GPU 编号由 1 修正为本服务器的 0；移除未使用的跨数据集缓存。
- 更新主 README、配置与输出说明，历史文档归档，补充忽略规则与整合测试。
- 原项目完整备份，既有数据集与研究结果保持在仓库之外。

## v5

- PURE 正式实验改为 AVI 优先且默认禁止 PNG 回退；
- 新增独立 `check_pure_avi_integrity.py`，逐帧检查 AVI，不进入主循环；
- pre-ROI 结果保存到独立 `pre_roi_gamma/`；
- 新增固定原图 ROI 的 `run_post_roi_analysis.py`；
- post-ROI 结果和 pre/post 比较统一保存到 `post_roi_analyse/`；
- 新增差分中的差分和配对 Wilcoxon-Holm 比较；
- 扩展 ROI RGB、luma、空间、时间和有效像素特征；
- 新增正向敏感、稳定、负向和非单调响应分类；
- 新增多算法一致性、Spearman、组间检验和诊断包；
- Shell 入口自动保存完整日志；
- 增加 v5 专用配置、文档和测试。
