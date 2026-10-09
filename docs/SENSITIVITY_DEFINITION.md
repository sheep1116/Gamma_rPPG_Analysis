# Gamma 敏感性定义

对每个 `sample_uid + method`，以 Gamma=1.0 为基准：

\[
Improvement_{abs}=MAE_{1.0}-\min_\gamma MAE_\gamma
\]

\[
Improvement_{rel}=\frac{MAE_{1.0}-\min_\gamma MAE_\gamma}{MAE_{1.0}}
\]

默认 `positive_sensitive` 同时要求：

- 绝对 MAE 至少下降 0.2 BPM；
- 相对 MAE 至少下降 5%；
- 最佳 Gamma 邻近的至少一个 Gamma 也优于基准；
- 估算有效窗口数不少于 10。

其他标签：

- `stable`：所有 Gamma 的 MAE 变化均在稳定阈值内；
- `negative_sensitive`：非基准 Gamma 持续恶化；
- `non_monotonic`：存在明显转折，但不满足稳定正向条件；
- `invalid`：样本或窗口不足，不能参与敏感性结论。

跨方法共识标签：

- `multi_method_positive`；
- `single_method_positive`；
- `mixed_response`；
- `mostly_negative`；
- `mostly_stable`。

重点人工检查 `multi_method_positive`，因为多个算法一致受益更可能来自视频本身的亮度和颜色属性。
