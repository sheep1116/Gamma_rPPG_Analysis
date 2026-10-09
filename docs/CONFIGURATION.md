# 配置说明

## 正式主实验

`configs/full_pre_roi_experiment.yaml`

关键项：

```yaml
experiment:
  output_subdir: pre_roi_gamma
  transform_mode: pre_roi_gamma
  gamma_values: [1.0, ..., 2.8]

datasets:
  PURE:
    prefer_avi: true
    allow_frames_fallback: false
```

## post-ROI

`configs/post_roi_analysis.yaml`

```yaml
post_roi:
  roi_cache:
    enabled: true
    version: fixed_mask_v1
```

修改 ROI 定义或关键点处理后必须提升缓存版本。

## AVI 检查

`configs/pure_avi_integrity.yaml` 控制期望 FPS、时长容差、是否逐帧解码和 JSON 信号采样率。

## 敏感性分析

`configs/sensitivity_analysis.yaml` 控制绝对/相对改善阈值、相邻 Gamma 支持、最低窗口数、相关性 bootstrap 次数和诊断包数量。

## 输出路径

YAML 是唯一来源：

```yaml
output:
  root: /home/yangjicheng/Result
  experiment_name: GammaBrightness_rPPG_v5
  folders:
    pre_roi_gamma: pre_roi_gamma
    post_roi_analyse: post_roi_analyse
    diagnostics: diagnostics
    pure_avi_check: diagnostics/PURE_AVI_CHECK
```

## 公共配置和便携路径

`datasets.yaml` 是三类数据集的路径来源；full_pre_roi_experiment、post_roi_analysis、pure_avi_integrity 继承该文件，quick_pre_roi_test 只覆盖样本范围。`extends` 路径相对于当前 YAML 文件解析，字典递归合并，列表整体替换。通过 `load_config` 读取配置，不能只用 `yaml.safe_load` 代替完整配置解析。

`${GAMMA_DATA_ROOT:-/tank/数据集}` 和 `${GAMMA_RESULT_ROOT:-~/Result}` 分别读取环境变量或默认路径。缺少没有默认值的变量会报错。GPU 默认编号为 0。

本地配置示例（不提交）：

```yaml
extends: full_pre_roi_experiment.yaml
output:
  root: /different/results
experiment:
  cuda_device: 0
```

保存为 `configs/experiment.local.yaml` 并将它作为 Shell 入口第一个参数。

ROI 机制配置为 `roi_mechanism.yaml`。`default.yaml` 是它的兼容别名；full_experiment 和 quick_test 分别是正式/快速 pre-ROI 配置的兼容别名。ROI 配置保留独立的 result_root、output_subdir、core_features 和 mechanism_features，分析的 result_root 指整个结果根目录，不能指单一数据集子目录。
