"""Environment setup: state is checked, never assumed.

The bug this guards against cost two Colab sessions: a marker file written without
looking at pip's exit status made the setup report success while every segmentation
tool was missing, and the pipeline then skipped all sixteen stages.
"""

from __future__ import annotations

import pytest

import spinelab.envsetup as envsetup


class TestCheck:
    def test_core_modules_are_seen_as_present(self):
        state = envsetup.check(segmentation=False)
        assert state["modules"]["nibabel"]["importable"] is True
        assert state["modules"]["pydicom"]["importable"] is True
        assert state["ready"] is True
        assert state["missing_modules"] == []

    def test_an_already_imported_c_extension_is_not_reimported(self):
        # numpy refuses to load twice in one process, and in Colab it is always
        # imported before this runs. Purging and re-importing reported healthy
        # tools as broken.
        import numpy  # noqa: F401

        state = envsetup.check(segmentation=False)
        assert state["modules"]["numpy"]["importable"] is True
        assert "cannot load module more than once" not in state["modules"]["numpy"]["detail"]

    def test_a_recommended_module_does_not_block_readiness(self):
        # SimpleITK missing means `register` skips; it must not stop the pipeline.
        state = envsetup.check(segmentation=False)
        assert "SimpleITK" in state["modules"]
        assert state["ready"] is True

    def test_absent_segmentation_stack_is_reported_missing(self):
        # None of the GPU tools are installed in the test environment.
        state = envsetup.check(segmentation=True)
        assert state["ready"] is False
        assert "spineps" in state["missing_modules"]
        assert state["modules"]["spineps"]["importable"] is False
        assert "ModuleNotFoundError" in state["modules"]["spineps"]["detail"]

    def test_binaries_are_probed(self):
        state = envsetup.check(segmentation=False)
        assert set(state["binaries"]) == set(envsetup.BINARIES)

    def test_gpu_line_is_always_a_string(self):
        assert isinstance(envsetup.check(segmentation=False)["gpu"], str)

    def test_required_modules_depends_on_the_profile(self):
        assert "spineps" in envsetup.required_modules(True)
        assert "spineps" not in envsetup.required_modules(False)
        assert "nibabel" in envsetup.required_modules(False)
        assert "SimpleITK" not in envsetup.required_modules(True)


class TestInstall:
    def test_nothing_is_installed_when_everything_is_present(self, monkeypatch):
        calls = []
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: calls.append(cmd) or True)
        lines: list[str] = []
        state = envsetup.install(segmentation=False, log=lines.append)
        assert calls == []
        assert state["ready"] is True
        assert any("already importable" in line for line in lines)

    def test_a_failed_install_is_reported_not_hidden(self, monkeypatch):
        # pip fails for everything; the result must be "NOT READY", never success.
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: False)
        lines: list[str] = []
        state = envsetup.install(segmentation=True, apt=False, log=lines.append)
        text = "\n".join(lines)
        assert state["ready"] is False
        assert "NOT READY" in text
        assert "Restart session" in text
        assert "Do not start the pipeline" in text

    def test_combined_failure_retries_package_by_package(self, monkeypatch):
        attempts: list[list[str]] = []

        def fake_run(cmd, log):
            attempts.append(cmd)
            return False

        monkeypatch.setattr(envsetup, "_run", fake_run)
        envsetup.install(segmentation=True, apt=False, log=lambda *_: None)
        installed_args = [" ".join(c) for c in attempts]
        # One combined attempt with all four, then one attempt per package.
        assert any(all(p.split(">=")[0] in a for p in envsetup.SEGMENTATION_PACKAGES)
                   for a in installed_args)
        for package in envsetup.SEGMENTATION_PACKAGES:
            assert any(a.endswith(package) for a in installed_args), package

    def test_force_reinstalls_even_when_present(self, monkeypatch):
        seen: list[list[str]] = []
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: seen.append(cmd) or True)
        envsetup.install(segmentation=False, force=True, apt=False, log=lambda *_: None)
        assert seen, "force must install regardless of what is importable"
        assert any("--force-reinstall" in " ".join(cmd) for cmd in seen)

    def test_apt_can_be_skipped(self, monkeypatch):
        seen: list[list[str]] = []
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: seen.append(cmd) or True)
        envsetup.install(segmentation=False, force=True, apt=False, log=lambda *_: None)
        assert not any("apt-get" in " ".join(cmd) for cmd in seen)


class TestAcvlConflict:
    """The conflict that actually blocked every install.

        spineps 2.0.0 -> acvl-utils==0.2
        nnunetv2 2.8.1 -> acvl-utils>=0.2.6,<0.3

    pip cannot satisfy both, so it must never be shown both at once.
    """

    def test_spineps_is_never_resolved_together_with_nnunet(self):
        joint = " ".join(envsetup.SEGMENTATION_PACKAGES).lower()
        assert "spineps" not in joint, (
            "SPINEPS in the combined install re-creates the acvl-utils conflict")

    def test_the_pinned_acvl_version_is_one_nnunet_accepts(self):
        # nnU-Net has required >=0.2.6 since 2.7.0; anything older re-breaks it.
        assert envsetup.ACVL_UTILS == "acvl-utils==0.2.6"

    def test_spineps_dependencies_exclude_acvl_utils(self):
        assert not any("acvl" in dep.lower() for dep in envsetup.SPINEPS_DEPS), (
            "supplying acvl-utils by hand would reinstate the pin we are overriding")

    def test_spineps_dependencies_cover_its_metadata(self):
        # --no-deps means pip does not read requires_dist, so this list is the only
        # thing standing between us and an ImportError deep inside segmentation.
        for name in ("TPTBox", "antspyx", "monai", "pytorch-lightning", "nnunetv2",
                     "torchmetrics", "einops", "rich", "tqdm", "TypeSaveArgParse"):
            assert any(dep.lower().startswith(name.lower()) for dep in envsetup.SPINEPS_DEPS), name

    def test_spineps_is_installed_with_no_deps(self, monkeypatch):
        seen: list[list[str]] = []
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: seen.append(cmd) or True)
        envsetup.install(segmentation=True, apt=False, log=lambda *_: None)
        spineps_calls = [c for c in seen if any("SPINEPS" in a for a in c)]
        assert spineps_calls, "SPINEPS must still be installed"
        assert all("--no-deps" in c for c in spineps_calls)

    def test_acvl_is_installed_before_anything_that_needs_it(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(envsetup, "_run",
                            lambda cmd, log: order.append(" ".join(cmd)) or True)
        envsetup.install(segmentation=True, apt=False, log=lambda *_: None)
        acvl = next(i for i, c in enumerate(order) if "acvl-utils" in c)
        nnunet = next(i for i, c in enumerate(order) if "nnunetv2" in c)
        assert acvl < nnunet, "acvl-utils is source-only; building it later hides its errors"

    def test_a_broken_pin_is_reported_by_symbol_name(self, monkeypatch):
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: True)
        monkeypatch.setattr(envsetup, "acvl_symbols",
                            lambda: ["acvl_utils.cropping_and_padding.padding.pad_nd_image"])
        monkeypatch.setattr(envsetup, "_probe", lambda module: (True, "stub"))
        lines: list[str] = []
        state = envsetup.install(segmentation=True, apt=False, log=lines.append)
        text = "\n".join(lines)
        assert state["ready"] is False, "importable modules are not enough if the pin is real"
        assert "pad_nd_image" in text
        assert "nnunetv2==2.5.2" in text, "the fallback that does satisfy both must be named"
        assert "ready — the pipeline can run" not in text

    def test_symbol_check_is_skipped_when_spineps_is_absent(self):
        # Reporting missing symbols for a tool that is simply not installed would
        # bury the real message ("spineps cannot be imported") under noise.
        state = envsetup.check(segmentation=True)
        assert state["modules"]["spineps"]["importable"] is False
        assert state["missing_acvl_symbols"] == []


class TestSmokeTest:
    """--no-deps means an importable package can still have a dead entry point."""

    def test_a_crashing_cli_makes_the_environment_not_ready(self, monkeypatch):
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: True)
        monkeypatch.setattr(envsetup, "_probe", lambda module: (True, "stub"))
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        monkeypatch.setattr(envsetup.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(envsetup, "smoke_test", lambda binaries: {
            "spineps": "starts but crashes: ModuleNotFoundError: No module named 'monai'",
            "totalspineseg": "ok", "dcm2niix": "ok"})
        lines: list[str] = []
        state = envsetup.install(segmentation=True, force=True, apt=False, log=lines.append)
        text = "\n".join(lines)
        assert state["ready"] is False
        assert state["crashing_binaries"] == ["spineps"]
        assert "monai" in text
        assert "SPINEPS_DEPS" in text, "the fix has to be named, not guessed at"
        assert "ready — the pipeline can run" not in text

    def test_a_traceback_is_what_counts_as_broken_not_the_exit_status(self, monkeypatch):
        # dcm2niix answers -h with a non-zero status and is perfectly healthy.
        class Proc:
            stdout = "usage: dcm2niix [options]"
            stderr = ""
            returncode = 1

        monkeypatch.setattr(envsetup.subprocess, "run", lambda *a, **k: Proc())
        assert envsetup.smoke_test({"dcm2niix": "/usr/bin/dcm2niix"}) == {"dcm2niix": "ok"}

    def test_a_traceback_is_reported_with_its_last_line(self, monkeypatch):
        class Proc:
            stdout = ""
            stderr = ('Traceback (most recent call last):\n'
                      '  File "/x/spineps", line 5, in <module>\n'
                      "ModuleNotFoundError: No module named 'antspyx'\n")
            returncode = 1

        monkeypatch.setattr(envsetup.subprocess, "run", lambda *a, **k: Proc())
        verdict = envsetup.smoke_test({"spineps": "/usr/bin/spineps"})["spineps"]
        assert "crashes" in verdict
        assert "antspyx" in verdict

    def test_binaries_that_are_absent_are_not_smoke_tested(self):
        assert envsetup.smoke_test({"spineps": None}) == {}

    def test_check_does_not_run_binaries_unless_asked(self, monkeypatch):
        # The default `check()` is used by the CLI on every run; spawning three
        # subprocesses there would make `--check-only` far from instant.
        monkeypatch.setattr(envsetup, "smoke_test",
                            lambda binaries: pytest.fail("must not be called"))
        state = envsetup.check(segmentation=True)
        assert state["smoke"] == {}
        assert state["crashing_binaries"] == []


class TestTorchWarning:
    def test_an_excluded_torch_is_announced_before_pip_replaces_it(self, monkeypatch):
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: True)
        monkeypatch.setattr(envsetup, "_probe",
                            lambda module: (True, "2.9.1" if module == "torch" else "stub"))
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        lines: list[str] = []
        envsetup.install(segmentation=True, force=True, apt=False, log=lines.append)
        text = "\n".join(lines)
        assert "2.9" in text and "replace" in text

    def test_a_supported_torch_produces_no_warning(self, monkeypatch):
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: True)
        monkeypatch.setattr(envsetup, "_probe",
                            lambda module: (True, "2.13.0" if module == "torch" else "stub"))
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        lines: list[str] = []
        envsetup.install(segmentation=True, force=True, apt=False, log=lines.append)
        assert "replace" not in "\n".join(lines)


class TestCliContract:
    def test_check_only_exits_nonzero_when_not_ready(self, capsys):
        from spinelab.cli import main

        code = main(["setup", "--check-only"])
        out = capsys.readouterr().out
        assert code == 1                    # segmentation stack absent here
        assert "missing:" in out
        assert "spineps" in out

    def test_check_only_without_segmentation_is_ready(self, capsys):
        from spinelab.cli import main

        code = main(["setup", "--check-only", "--no-segmentation"])
        assert code == 0
        assert "ready" in capsys.readouterr().out
