import pandas as pd

from gamma_rppg.analysis.sensitivity import classify_response, cross_method_consensus


def test_positive_sensitive_requires_stable_improvement():
    frame = pd.DataFrame({
        "gamma": [1.0, 1.2, 1.4, 1.6],
        "MAE": [4.0, 3.7, 3.5, 3.6],
    })
    result = classify_response(frame, 1.0, 0.2, 0.05, True, 0.2)
    assert result["response_type"] == "positive_sensitive"
    assert result["best_gamma"] == 1.4


def test_cross_method_consensus():
    labels = pd.DataFrame([
        {"dataset": "PURE", "sample_uid": "s1", "method": "POS", "response_type": "positive_sensitive", "relative_mae_improvement": .1, "best_gamma": 1.2},
        {"dataset": "PURE", "sample_uid": "s1", "method": "CHROM", "response_type": "positive_sensitive", "relative_mae_improvement": .2, "best_gamma": 1.4},
        {"dataset": "PURE", "sample_uid": "s1", "method": "GREEN", "response_type": "stable", "relative_mae_improvement": .0, "best_gamma": 1.0},
    ])
    result = cross_method_consensus(labels).iloc[0]
    assert result["cross_method_consensus"] == "multi_method_positive"
    assert result["positive_method_count"] == 2


def test_incomplete_gamma_set_is_not_eligible():
    from gamma_rppg.analysis.sensitivity import build_labels
    raw = pd.DataFrame([
        {"dataset": "PURE", "sample_uid": "s1", "method": "POS", "transform_mode": "pre_roi_gamma", "status": "ok", "gamma": 1.0, "MAE": 4.0, "SNR": 1.0, "fps": 30, "num_frames": 900},
        {"dataset": "PURE", "sample_uid": "s1", "method": "POS", "transform_mode": "pre_roi_gamma", "status": "ok", "gamma": 1.2, "MAE": 3.0, "SNR": 2.0, "fps": 30, "num_frames": 900},
        {"dataset": "PURE", "sample_uid": "s2", "method": "POS", "transform_mode": "pre_roi_gamma", "status": "ok", "gamma": 1.0, "MAE": 4.0, "SNR": 1.0, "fps": 30, "num_frames": 900},
        {"dataset": "PURE", "sample_uid": "s2", "method": "POS", "transform_mode": "pre_roi_gamma", "status": "method_error", "gamma": 1.2, "MAE": None, "SNR": None, "fps": 30, "num_frames": 900},
    ])
    config = {
        "analysis": {"baseline_gamma": 1.0},
        "experiment": {"wsize": 8, "step": 1},
        "sensitivity": {
            "min_absolute_mae_improvement": 0.2,
            "min_relative_mae_improvement": 0.05,
            "require_adjacent_gamma_support": False,
            "stable_mae_epsilon": 0.2,
            "min_valid_windows": 10,
        },
    }
    labels, _ = build_labels(raw, config)
    result = labels.set_index("sample_uid")
    assert bool(result.loc["s1", "gamma_complete"]) is True
    assert bool(result.loc["s2", "gamma_complete"]) is False
    assert result.loc["s2", "response_type"] == "invalid"
