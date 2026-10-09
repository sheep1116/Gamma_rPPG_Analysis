import pandas as pd

from gamma_rppg.analysis.summary import complete_case_rows, normalize_raw, success_rate_table


def synthetic_raw():
    rows = []
    for sample in ["s1", "s2"]:
        for gamma in [1.0, 1.2, 1.4]:
            status = "ok"
            if sample == "s2" and gamma == 1.4:
                status = "method_error"
            rows.append({
                "dataset": "PURE", "sample_uid": sample, "gamma": gamma, "method": "POS",
                "transform_mode": "pre_roi_gamma", "status": status, "MAE": gamma,
                "RMSE": gamma, "MAX": gamma, "PCC": 0.5, "CCC": 0.5, "SNR": 1.0,
            })
    return pd.DataFrame(rows)


def test_complete_cases_require_all_gammas():
    raw, duplicates = normalize_raw(synthetic_raw())
    assert duplicates == 0
    ok = raw[raw.status == "ok"]
    paired, sample_table = complete_case_rows(ok, [1.0, 1.2, 1.4])
    assert set(paired.sample_uid.unique()) == {"s1"}
    complete = sample_table.set_index("sample_uid")["complete"].to_dict()
    assert bool(complete["s1"]) is True
    assert bool(complete["s2"]) is False


def test_success_rate_uses_union_sample_count():
    raw, _ = normalize_raw(synthetic_raw())
    table = success_rate_table(raw, [1.0, 1.2, 1.4])
    last = table[table.gamma == 1.4].iloc[0]
    assert last.dataset_method_sample_count == 2
    assert last.success_count == 1
    assert last.success_rate == 0.5
