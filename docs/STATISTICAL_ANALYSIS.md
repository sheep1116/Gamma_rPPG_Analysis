# 统计分析

## 主实验

每个数据集和方法分别计算均值、中位数、标准差、视频级 bootstrap 95% 置信区间、Friedman 检验、Gamma 与基准的配对 Wilcoxon 检验及 Holm 校正。主趋势优先采用覆盖全部 Gamma 的完整配对样本。

当 Wilcoxon 差值中存在零时，代码显式采用正态近似，避免 SciPy 重复输出“精确 p 值无法处理零差值”的警告。

## pre/post 比较

只使用两种模式共同成功并覆盖全部 Gamma 的样本—方法组合。报告绝对指标差异和相对各自 Gamma=1.0 基线的差分中的差分。对 MAE、SNR 执行配对 Wilcoxon和 Holm 校正。

## ROI 特征

- 基线 ROI 特征与最佳 MAE 改善：Spearman 相关、bootstrap 95% CI、Holm 校正；
- Gamma 引起的 ROI 特征变化与 MAE 变化：Spearman 相关；
- 响应组间：Kruskal–Wallis、Mann–Whitney U、Cliff's delta 和 Holm 校正。

不同数据集不合并进行统计推断。
