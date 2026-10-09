# v5 构建与验证记录

## 已完成检查

- 所有 `src/` 与 `tests/` Python 文件通过 `compileall` 语法检查。
- 所有 `scripts/*.sh` 通过 `bash -n` 语法检查。
- 所有 `configs/*.yaml` 可由 PyYAML 正常解析。
- 当前构建环境中执行测试：10 项通过，2 项跳过。
- 跳过项需要完整的 pyVHR 运行环境，用于验证 PURE 数据集适配器和 post-ROI 与 pyVHR `holistic_mean` 的集成语义。
- 使用合成 AVI 验证了独立 AVI 完整性检查脚本的实际逐帧解码统计。
- 使用合成结果验证了 pre-ROI 敏感性分类、相关性表、组间检验、图表及案例目录生成。
- 使用合成结果验证了完整配对汇总与统计报告生成。

## 尚未执行

当前构建环境不包含真实 UBFC2、LGI_PPGI、PURE 数据集，也不包含用户指定的完整 pyVHR-GPU、CUDA、MediaPipe 环境，因此未执行真实数据的端到端 GPU 实验。

正式运行前建议依次执行：

```bash
bash scripts/check_pure_avi_integrity.sh
bash scripts/quick_test_all.sh
bash scripts/run_pre_roi_all_datasets.sh
bash scripts/summarize_pre_roi.sh
bash scripts/analyse_pre_roi_sensitivity.sh
bash scripts/run_post_roi_analysis.sh
```
