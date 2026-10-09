import pandas as pd

from gamma_rppg.figures.cases import select_explicit_cases


def test_explicit_case_selection_uses_only_configured_case(tmp_path):
    dataset_dir = tmp_path / "UBFC2"
    dataset_dir.mkdir()
    pd.DataFrame([
        {
            "sample_uid": "UBFC2/subject8/vid.avi",
            "method": "GREEN",
            "gamma": 1.0,
            "MAE": 28.69,
            "fps": 29.88,
            "input_path": "/data/subject8/vid.avi",
            "status": "ok",
            "error_message": "",
        },
        {
            "sample_uid": "UBFC2/subject8/vid.avi",
            "method": "GREEN",
            "gamma": 2.6,
            "MAE": 8.87,
            "fps": 29.88,
            "input_path": "/data/subject8/vid.avi",
            "status": "ok",
            "error_message": "",
        },
        {
            "sample_uid": "UBFC2/subject9/vid.avi",
            "method": "POS",
            "gamma": 1.0,
            "MAE": 1.0,
            "fps": 29.88,
            "input_path": "/data/subject9/vid.avi",
            "status": "ok",
            "error_message": "",
        },
    ]).to_csv(dataset_dir / "raw_results.csv", index=False)

    config = {
        "analysis": {"baseline_gamma": 1.0},
        "case_mechanism": {
            "explicit_cases": [{
                "dataset": "UBFC2",
                "sample_uid": "UBFC2/subject8/vid.avi",
                "method": "GREEN",
                "category": "positive",
                "target_gamma": 2.6,
            }],
        },
    }

    cases = select_explicit_cases(dataset_dir, config)

    assert len(cases) == 1
    assert cases[0].sample_uid == "UBFC2/subject8/vid.avi"
    assert cases[0].target_gamma == 2.6
    assert cases[0].base_mae == 28.69
    assert cases[0].target_mae == 8.87
