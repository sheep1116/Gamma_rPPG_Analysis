from pathlib import Path

from gamma_rppg.common.result_store import AtomicCSVStore


def test_upsert_does_not_duplicate(tmp_path: Path):
    path = tmp_path / "results.csv"
    columns = ["dataset", "sample_uid", "gamma", "method", "status", "MAE"]
    keys = ["dataset", "sample_uid", "gamma", "method"]
    store = AtomicCSVStore(path, columns, keys)
    base = {"dataset": "PURE", "sample_uid": "PURE/A/01", "gamma": 1.0, "method": "POS"}
    store.upsert({**base, "status": "error", "MAE": ""})
    store.upsert({**base, "status": "ok", "MAE": 2.5})
    reloaded = AtomicCSVStore(path, columns, keys)
    assert len(reloaded) == 1
    assert reloaded.get(base)["status"] == "ok"
