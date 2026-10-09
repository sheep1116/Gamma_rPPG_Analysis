# Project design notes

## Primary estimands

1. Baseline association:
   `Spearman(baseline ROI feature, best MAE benefit)`.
2. Positive-response separation:
   `positive_sensitive` versus `stable` using Mann–Whitney U and Cliff's delta.
3. Within-video mechanism:
   for each video, `Spearman(delta ROI feature, MAE benefit)` across Gamma values, followed by a video-level summary.
4. Failure susceptibility:
   baseline ROI features of videos that ever fail at Gamma > 1 compared with videos that never fail.

## Why no predictive model

The goal is interpretable mechanism analysis, not out-of-sample classification. The project therefore does not train machine-learning prediction models.

## Multiplicity

Holm correction is applied separately inside each dataset–method family. This keeps the inferential unit aligned with the algorithm-specific hypothesis.

## Repeated measures

Gamma points from the same video are not treated as independent in the main mechanism statistic. Correlations are first calculated inside each video and then summarized across videos.
