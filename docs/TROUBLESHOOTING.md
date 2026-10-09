# 故障排查

## PURE 样本数量不足

先执行：

```bash
bash scripts/check_pure_avi_integrity.sh
bash scripts/inspect_pure_structure.sh
```

正式配置禁止 PNG 回退。缺少 AVI 的样本不会进入主实验，应在诊断报告中处理。

## AVI 存在但无法解码

查看 `diagnostics/PURE_AVI_CHECK/pure_avi_integrity.csv` 中的 `avi_opened`、`reported_frame_count`、`decoded_frame_count` 和 `issues`。主实验不会自动修复或替换该文件。

## post-ROI 缓存不一致

删除对应 `post_roi_analyse/roi_cache` 文件，或提升 `roi_cache.version`。缓存会检查版本、FPS 和最低帧数。

## 重新运行

- `--resume`：跳过已有任务；
- `--retry-errors`：只重试失败；
- `--overwrite`：删除所选数据集结果后重跑。

## 查看完整终端输出

所有 Shell 入口使用 `tee` 自动保存日志，并启用 `pipefail`。日志位于对应模块的 `logs/`。

## 敏感样本没有 ROI 图

只有主实验预设的质量示例或启用的信号诊断样本会生成 ROI 图。调整 `sensitivity_cases.max_signal_diagnostic_cases` 后重新运行敏感性分析即可；不必重跑全部主实验，但目标案例会重新提取信号。
