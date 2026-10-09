# pre_roi_gamma 正式主实验

## 流程

```text
原始帧 → 整帧 Gamma → Gamma 帧 FaceMesh → Gamma 帧 ROI → RGB 时间序列 → BVP → BPM → 指标
```

Gamma 同时可能改变人脸检测、关键点、ROI 几何、有效像素和颜色信号，因此该模式衡量完整处理链的综合效应。

## PURE 输入

正式配置使用：

```yaml
prefer_avi: true
allow_frames_fallback: false
```

主程序只检查 AVI 路径是否存在，不逐帧扫描完整性，以避免把诊断开销加入正式实验。正式实验前应单独执行 `scripts/check_pure_avi_integrity.sh`。

## 运行

```bash
bash scripts/run_pre_roi_all_datasets.sh
bash scripts/run_pre_roi_pure.sh
bash scripts/run_pre_roi_ubfc2.sh
bash scripts/run_pre_roi_lgi_ppgi.sh
```

第二个参数可指定 `--resume`、`--retry-errors` 或 `--overwrite`。默认断点续跑。

## 输出

每个数据集独立保存 `raw_results.csv`、`brightness_metrics.csv`、完整配对统计、成功率、统计检验、质量控制和图表。PURE 主均值覆盖六类运动中的全部视频，运动宏平均和受试者宏平均只作补充。
