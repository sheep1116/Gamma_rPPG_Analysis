# post_roi_analyse 固定 ROI 控制实验

## 研究目的

该模块只回答：Gamma 在 ROI 前与 ROI 后是否有显著区别。它不执行敏感视频分类或复杂 ROI 规律分析。

## 严格流程

```text
原始帧 → 原图 FaceMesh → 原图 ROI 掩膜 → 取 ROI 原始像素
       → ROI 像素逐像素 Gamma → 阈值筛选 → RGB 均值 → rPPG
```

同一视频所有 Gamma 使用完全相同的原图关键点和 ROI。不能先求 ROI 均值再做 Gamma，因为一般有：

\[
E[X^\gamma]\neq E[X]^\gamma
\]

## ROI 缓存

缓存位于：

```text
post_roi_analyse/roi_cache/<dataset>/*.npz
```

包含每帧关键点、检测有效标记、FPS、帧数和缓存版本。若修改 ROI 定义，应更改 `post_roi.roi_cache.version`，避免复用旧缓存。

## 运行

```bash
bash scripts/run_post_roi_analysis.sh
```

默认依次执行 post-ROI 实验、基础汇总和 pre/post 比较。也可分开执行：

```bash
bash scripts/run_post_roi_analysis.sh configs/post_roi_analysis.yaml experiment --resume
bash scripts/run_post_roi_analysis.sh configs/post_roi_analysis.yaml compare --resume
```

## 比较口径

只使用两种模式共同成功、且覆盖全部 Gamma 的“样本—方法”完整配对集合。每种模式先相对自身 Gamma=1.0 计算变化：

\[
\Delta_{pre}=MAE_{pre,\gamma}-MAE_{pre,1}
\]

\[
\Delta_{post}=MAE_{post,\gamma}-MAE_{post,1}
\]

再计算：

\[
D=\Delta_{pre}-\Delta_{post}
\]

`D<0` 表示 pre-ROI 的 Gamma 改善更强；`D>0` 表示 post-ROI 更强；接近 0 表示两种位置的 Gamma 效应近似。
