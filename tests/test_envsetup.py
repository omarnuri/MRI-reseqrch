"""Environment setup: state is checked, never assumed.

The bug this guards against cost two Colab sessions: a marker file written without
looking at pip's exit status made the setup report success while every segmentation
tool was missing, and the pipeline then skipped all sixteen stages.
"""

from __future__ import annotations

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
