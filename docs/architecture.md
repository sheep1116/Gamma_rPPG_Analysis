# 整合项目架构

## 研究问题

1. Gamma 在人脸检测和 ROI 提取之前时，对完整 rPPG 流程有什么影响；
2. 固定原图 ROI 后只变换 ROI 像素，结果是否与 pre-ROI 显著不同；
3. 哪些视频从 Gamma 中获得稳定正向效果，其原始 ROI 颜色和亮度是否存在共同规律。

## 模块边界

| 模块 | 入口 | 作用 | 是否影响主实验 |
|---|---|---|---|
| PURE AVI 检查 | `check_pure_avi_integrity.py` | 数据完整性诊断 | 否 |
| pre-ROI | `run_gamma_experiment.py` | 正式主实验 | 主实验 |
| pre-ROI 敏感性 | `analyse_pre_roi_sensitivity.py` | ROI 规律与正向案例 | 只读主结果 |
| post-ROI | `run_post_roi_analysis.py` | 固定 ROI 控制实验 | 否，独立目录 |
| 模式比较 | 同上 `--stage compare` | pre/post 配对检验 | 只读两类结果 |

## 数据集口径

- UBFC2、LGI_PPGI、PURE 独立保存和统计；
- PURE 六类运动来源统一进入 PURE 主均值；
- 运动宏平均和受试者宏平均只作补充；
- 不创建跨数据集合并结果。

## 推荐研究顺序

1. 运行 PURE AVI 检查并修复明显异常；
2. 运行 pre-ROI 主实验；
3. 生成主统计和图表；
4. 运行 pre-ROI 敏感性及 ROI 特征分析；
5. 独立运行 post-ROI；
6. 对共同完整样本执行 pre/post 比较；
7. 根据统计结果和案例诊断形成结论。

## 解释边界

- pre-ROI 的改善可能来自检测、ROI 几何和颜色信号的共同变化；
- post-ROI 的改善只能来自固定 ROI 内的像素变换和有效像素筛选；
- 若两者 Gamma 效应接近，颜色信号变化可能占主导；
- 若两者显著不同，检测和 ROI 几何因素可能不可忽略；
- 相关性和组间检验用于发现规律，不等于因果证明；
- v5 不训练预测模型。

## 代码组织与调用关系

公共配置、原子 CSV 存储和多重校正位于 `gamma_rppg.common`。数据适配和像素数学分别位于 datasets 和 signal；视频实验通过它们提取 RGB/BVP/BPM。summary 与 sensitivity 读取实验 CSV；ROI 机制模块继续读取 sensitivity、response、changes、brightness 和 raw 表，不反向调用实验。

```text
数据集 → datasets + signal → experiments → pre_roi_gamma / post_roi_analyse
                                               ↓
                                     summary + sensitivity
                                               ↓
                                  analysis.roi_mechanism
                                               ↓
                                    tables / figures / reports
```

`figures.cases` 按已有结果选择少量典型案例，再按需提取视频信号。诊断模块独立检查视频完整性。CPU 统计模块不导入 GPU 实验模块，旧 src 包仅转发，不再存放实现。

逐通道变换公式仍为 `I_out = 255 * (I_in / 255)^gamma`。它作用于编码 RGB 像素，不等同于相机响应反演或标准 sRGB 线性化。ROI 机制估计对象与多重检验族见 [ROI_MECHANISM_DESIGN.md](ROI_MECHANISM_DESIGN.md)。
