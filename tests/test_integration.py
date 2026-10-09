import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gamma_rppg.common.config import load_config
from gamma_rppg.common.statistics import holm_adjust, holm_adjust_list


def test_config_aliases_preserve_effective_config():
    for alias, canonical in [("full_experiment", "full_pre_roi_experiment"), ("quick_test", "quick_pre_roi_test"), ("default", "roi_mechanism")]:
        assert load_config(f"configs/{alias}.yaml") == load_config(f"configs/{canonical}.yaml")
    full = load_config("configs/full_pre_roi_experiment.yaml")
    quick = load_config("configs/quick_pre_roi_test.yaml")
    assert full["datasets"]["PURE"]["prefer_avi"] is True
    assert full["datasets"]["PURE"]["allow_frames_fallback"] is False
    assert len(full["datasets"]["PURE"]["sources"]) == 6
    for name in full["datasets"]:
        assert full["datasets"][name]["video_indices"] == "all"
        assert quick["datasets"][name]["video_indices"] == "0:1"
    assert full["experiment"]["cuda_device"] == 0


def test_config_extends_merges_mappings_and_replaces_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMMA_TEST_PATH", "/portable/data")
    base = tmp_path / "base.yaml"
    base.write_text('paths: {data: "${GAMMA_TEST_PATH}"}\nrun: {methods: [POS, CHROM], seed: 4}\n', encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text('extends: base.yaml\nrun: {methods: [GREEN]}\n', encoding="utf-8")
    assert load_config(child) == {"paths": {"data": "/portable/data"}, "run": {"methods": ["GREEN"], "seed": 4}}


def test_config_rejects_cycles_and_missing_required_environment(tmp_path, monkeypatch):
    path = tmp_path / "a.yaml"
    path.write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular"):
        load_config(path)
    monkeypatch.delenv("GAMMA_TEST_UNSET", raising=False)
    path.write_text('path: "${GAMMA_TEST_UNSET}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unset"):
        load_config(path)


def test_holm_shares_finite_family_and_retains_missing_values():
    p = [0.2, np.nan, 0.01, 0.04]
    np.testing.assert_allclose(holm_adjust(p), [0.2, np.nan, 0.03, 0.08], equal_nan=True)
    assert holm_adjust_list([]) == []
    for module in ("gamma_rppg.analysis.summary", "gamma_rppg.analysis.sensitivity"):
        assert importlib.import_module(module).holm_adjust is holm_adjust_list


def test_legacy_import_preserves_private_helpers():
    old = importlib.import_module("src.summarize_results")
    new = importlib.import_module("gamma_rppg.analysis.summary")
    assert old.normalize_raw is new.normalize_raw
    assert old._legacy_sample_uid is new._legacy_sample_uid


def test_cpu_entrypoints_do_not_import_pyvhr():
    command = "import sys; import gamma_rppg.analysis.roi_mechanism.cli; assert 'pyVHR' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)
    for entry in (["-m", "gamma_rppg.analysis.roi_mechanism.cli"], ["run_analysis.py"], ["-m", "src.summarize_results"]):
        completed = subprocess.run([sys.executable, *entry, "--help"], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert "--config" in completed.stdout
