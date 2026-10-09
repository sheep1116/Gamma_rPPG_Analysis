from gamma_rppg.common.config import load_config, resolve_mode_root


def test_v5_module_paths_are_isolated():
    config = load_config("configs/full_pre_roi_experiment.yaml")
    pre = resolve_mode_root(config, "pre_roi_gamma")
    post = resolve_mode_root(config, "post_roi_analyse")
    diagnostics = resolve_mode_root(config, "diagnostics")
    assert pre.name == "pre_roi_gamma"
    assert post.name == "post_roi_analyse"
    assert diagnostics.name == "diagnostics"
    assert len({pre, post, diagnostics}) == 3
