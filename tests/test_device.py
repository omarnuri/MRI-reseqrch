"""The device choice has to reach the tools, not just the config object.

`gpu_required` was assigned by the CLI and read by nothing, so `--no-gpu` changed
nothing at all. The segmentation stages skipped anyway — because the CLIs were not
on PATH — which looks identical from the outside and is not the same thing.
"""

from __future__ import annotations

import spinelab.cli as cli
from spinelab.config import Config
from spinelab.stages.seg_spineps import _spineps_cmd


class TestResolveDevice:
    def test_explicit_cpu_is_honoured_even_with_a_gpu_present(self):
        assert Config(device="cpu").resolve_device() == "cpu"

    def test_explicit_cuda_is_not_second_guessed(self):
        # A user who says cuda gets cuda: silently downgrading would turn a
        # 4-minute run into a 40-minute one with no explanation.
        assert Config(device="cuda").resolve_device() == "cuda"

    def test_gpu_is_an_accepted_spelling_of_cuda(self):
        assert Config(device="gpu").resolve_device() == "cuda"

    def test_auto_answers_with_something_usable(self):
        assert Config(device="auto").resolve_device() in ("cuda", "cpu")

    def test_auto_falls_back_to_cpu_without_torch(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_torch)
        assert Config(device="auto").resolve_device() == "cpu"


class TestSpinepsCommand:
    def test_cpu_adds_the_flag_that_stops_it_loading_cuda(self, tmp_path):
        cmd = _spineps_cmd(tmp_path / "t2.nii.gz", "cpu")
        assert "-cpu" in cmd

    def test_cuda_does_not_pass_the_cpu_flag(self, tmp_path):
        assert "-cpu" not in _spineps_cmd(tmp_path / "t2.nii.gz", "cuda")

    def test_the_flags_match_the_installed_spineps_cli(self, tmp_path):
        # Verified against spineps 2.0.0's entrypoint parser. `-der` does not
        # exist; the option is `-der_name`, and an unknown flag makes argparse
        # exit before a single model is loaded.
        cmd = _spineps_cmd(tmp_path / "t2.nii.gz", "cuda")
        assert cmd[:2] == ["spineps", "sample"]
        for flag in ("-i", "-der_name", "-model_semantic", "-model_instance",
                     "-model_labeling", "-ignore_bids_filter"):
            assert flag in cmd, flag
        assert "-der" not in cmd

    def test_the_model_ids_are_the_ones_spineps_can_download(self, tmp_path):
        # spineps/utils/auto_download.py maps exactly these keys to release zips.
        cmd = _spineps_cmd(tmp_path / "t2.nii.gz", "cuda")
        assert cmd[cmd.index("-model_semantic") + 1] == "t2w"
        assert cmd[cmd.index("-model_instance") + 1] == "instance"
        assert cmd[cmd.index("-model_labeling") + 1] == "t2w_labeling"


class TestCliPlumbing:
    def _captured_config(self, argv, monkeypatch, tmp_path) -> Config:
        seen = {}

        def fake_run(config):
            seen["config"] = config
            (config.results_dir).mkdir(parents=True, exist_ok=True)
            (config.results_dir / "summary.json").write_text("{}", encoding="utf-8")
            return {}

        monkeypatch.setattr(cli, "run_pipeline", fake_run)
        cli.main(["run", "--work", str(tmp_path / "w"), *argv])
        return seen["config"]

    def test_device_reaches_the_config(self, monkeypatch, tmp_path):
        config = self._captured_config(["--device", "cpu"], monkeypatch, tmp_path)
        assert config.resolve_device() == "cpu"

    def test_no_gpu_now_means_something(self, monkeypatch, tmp_path):
        config = self._captured_config(["--no-gpu"], monkeypatch, tmp_path)
        assert config.device == "cpu", "--no-gpu used to be silently ignored"

    def test_default_is_auto(self, monkeypatch, tmp_path):
        assert self._captured_config([], monkeypatch, tmp_path).device == "auto"

    def test_timeout_applies_to_every_segmentation_tool(self, monkeypatch, tmp_path):
        # A CPU run of the same model is roughly ten times slower, so the default
        # 2400 s would report a timeout for a process that is working fine.
        config = self._captured_config(["--timeout", "9000"], monkeypatch, tmp_path)
        assert config.timeout_spineps_s == 9000
        assert config.timeout_tss_s == 9000
        assert config.timeout_ts_s == 9000

    def test_timeouts_keep_their_defaults_when_not_given(self, monkeypatch, tmp_path):
        config = self._captured_config([], monkeypatch, tmp_path)
        assert config.timeout_spineps_s == 2400
