# 从 v4 迁移

1. 将 v4 的真实数据路径复制到 v5 的三个相关 YAML；
2. PURE 改为 `prefer_avi: true`、`allow_frames_fallback: false`；
3. 先运行 AVI 完整性检查；
4. v4 的主结果不可直接写入 v5 目录，因为 v5 增加了新的 ROI 指标和输出分层；
5. v5 的主结果位于 `pre_roi_gamma/`，post-ROI 结果位于 `post_roi_analyse/`；
6. v4 的 `signal_only` 不作为 v5 的严格 post-ROI 结果。v5 使用原图固定 ROI，并在 ROI 像素级执行 Gamma；
7. 如需参考 v4 结果，应只作历史对照，不与 v5 任务键混写。
