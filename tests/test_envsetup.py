"""Environment setup: state is checked, never assumed.

The bug this guards against cost two Colab sessions: a marker file written without
looking at pip's exit status made the setup report success while every segmentation
tool was missing, and the pipeline then skipped all sixteen stages.
"""

from __future__ import annotations

import sys

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


class TestNumpyConsistency:
    """The Colab failure that made half the stack unimportable.

        ImportError: cannot import name '_center' from 'numpy._core.umath'

    numpy's own _core/strings.py imports `_center`, and _core/umath.py only
    re-exports it from 2.1 onward. A tree mixing the two cannot import either — and
    it is not a wrong version, so no version check would have caught it.
    """

    def test_a_healthy_numpy_reports_no_problem(self):
        assert envsetup.numpy_problem() == ""

    def test_a_mixed_tree_is_detected_and_quoted(self, monkeypatch):
        def broken(module):
            if module in envsetup.NUMPY_CONSISTENCY_MODULES:
                raise ImportError(
                    "cannot import name '_center' from 'numpy._core.umath'")
            return object()

        monkeypatch.setattr(envsetup.importlib, "import_module", broken)
        problem = envsetup.numpy_problem()
        assert "_center" in problem
        assert "numpy" in problem

    def test_an_unrelated_failure_is_not_blamed_on_numpy(self, monkeypatch):
        # A MemoryError while importing is not a mixed file set, and saying it is
        # would send the reader off to reinstall numpy for nothing.
        def other(module):
            raise MemoryError("not enough memory")

        monkeypatch.setattr(envsetup.importlib, "import_module", other)
        assert envsetup.numpy_problem() == ""

    def test_a_mixed_numpy_makes_the_environment_not_ready(self, monkeypatch):
        monkeypatch.setattr(envsetup, "numpy_problem", lambda: "numpy.strings: boom")
        state = envsetup.check(segmentation=False)
        assert state["ready"] is False
        assert state["numpy_problem"] == "numpy.strings: boom"

    def test_a_numpy_below_the_floor_is_the_reported_problem(self, monkeypatch):
        """Colab preinstalls 2.0.2, and that alone breaks the stack.

        Verified on an A100 runtime: with 2.0.2, nnU-Net could not resolve a trainer
        class and SPINEPS was unimportable; `pip install -U 'numpy>=2.1'` turned the
        same environment from NOT READY into ready.
        """
        import types

        monkeypatch.setitem(sys.modules, "numpy",
                            types.SimpleNamespace(__version__="2.0.2"))
        problem = envsetup.numpy_problem()
        assert "2.0.2" in problem
        assert "2.1" in problem

    def test_a_numpy_above_the_floor_is_accepted(self, monkeypatch):
        import types

        monkeypatch.setitem(sys.modules, "numpy",
                            types.SimpleNamespace(__version__="2.5.1"))
        # The submodule check still runs against the real numpy on this machine.
        assert envsetup.numpy_problem() == ""

    def test_the_repair_moves_numpy_forward(self, monkeypatch):
        # An earlier version of this reinstalled the *same* version, on the theory
        # that the file set was mixed rather than old. That was wrong: both symptoms
        # are one cause — numpy too old for packages built against 2.1+.
        seen: list[list[str]] = []
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: seen.append(cmd) or True)
        assert envsetup.repair_numpy(lambda *_: None) is True
        joined = " ".join(seen[0])
        assert envsetup.NUMPY_REQUIREMENT in joined
        assert "--no-deps" in joined, "a numpy move must not drag its dependents along"
        assert "--force-reinstall" not in joined, (
            "pip should be free to leave an already-adequate numpy alone")

    def test_the_repair_runs_before_the_stack_is_installed(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(envsetup, "_run",
                            lambda cmd, log: order.append(" ".join(cmd)) or True)
        monkeypatch.setattr(envsetup, "numpy_problem", lambda: "numpy 2.0.2 is too old")
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        envsetup.install(segmentation=True, apt=False, log=lambda *_: None)
        numpy_at = next(i for i, c in enumerate(order)
                        if envsetup.NUMPY_REQUIREMENT in c)
        stack_at = next(i for i, c in enumerate(order) if "nnunetv2" in c)
        assert numpy_at < stack_at, "the stack must not be built onto an old numpy"

    def test_a_repair_demands_a_restart_before_the_pipeline(self, monkeypatch):
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: True)
        monkeypatch.setattr(envsetup, "numpy_problem", lambda: "numpy.strings: boom")
        monkeypatch.setattr(envsetup, "_probe", lambda module: (True, "stub"))
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        monkeypatch.setattr(envsetup, "smoke_test", lambda binaries: {})
        lines: list[str] = []
        envsetup.install(segmentation=True, apt=False, log=lines.append)
        text = "\n".join(lines)
        assert "Restart session" in text
        assert "still holds modules loaded from the broken tree" in text

    def test_the_message_names_numpy_rather_than_just_the_symptom(self, monkeypatch):
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: True)
        monkeypatch.setattr(envsetup, "numpy_problem",
                            lambda: "numpy._core.strings: cannot import name '_center'")
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        lines: list[str] = []
        envsetup.install(segmentation=True, apt=False, log=lines.append)
        text = "\n".join(lines)
        assert "_center" in text
        assert envsetup.NUMPY_REQUIREMENT in text
        assert "Restart session" in text, (
            "numpy was already imported when the upgrade landed, so the live module "
            "object is stale until the kernel restarts — the reader has to be told")


class TestNnunetTrainerModules:
    """The fault that survived two attempts at detecting it.

    nnU-Net 2.8.1 imports `auglab` from one of its trainer modules, auglab imports
    `Tensor` from `kornia.core`, and that name was removed in kornia 0.8. Both
    TotalSegmentator and TotalSpineSeg resolve a trainer by name to load a model, so
    the broken module killed them after the weights were already on the GPU.
    """

    def test_a_healthy_environment_reports_nothing(self):
        # No false positives: nnU-Net raises RuntimeError for a name it cannot find,
        # so an earlier probe-by-absent-name flagged healthy environments as broken.
        assert envsetup.nnunet_problem() == ""

    def test_a_module_that_cannot_be_imported_is_named(self, monkeypatch):
        # Walks the real trainer package, with one real module made to fail the way
        # nnUNetTrainerDAExt does when kornia is 0.8.
        pytest.importorskip("nnunetv2")
        real_import = envsetup.importlib.import_module
        broken = envsetup.NNUNET_TRAINER_PACKAGE + ".nnUNetTrainer"

        def fake_import(name, *args, **kwargs):
            if name == broken:
                raise ImportError("cannot import name 'Tensor' from 'kornia.core'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(envsetup.importlib, "import_module", fake_import)
        problem = envsetup.nnunet_problem()
        assert "nnUNetTrainer" in problem
        assert "kornia.core" in problem
        assert "ImportError" in problem

    def test_the_kornia_pin_excludes_the_release_that_dropped_the_symbol(self):
        # kornia.core.Tensor exists up to 0.7.x and is gone in 0.8. Verified on the
        # A100 runtime: 0.8.3 fails, 0.7.4 imports the whole chain.
        pin = next(p for p in envsetup.STALE_ON_COLAB if p.startswith("kornia"))
        assert "<0.8" in pin, "kornia >= 0.8 has no kornia.core.Tensor for auglab"

    def test_a_broken_trainer_module_blocks_readiness(self, monkeypatch):
        monkeypatch.setattr(envsetup, "nnunet_problem",
                            lambda: "nnUNetTrainerDAExt: ImportError: kornia.core")
        monkeypatch.setattr(envsetup, "_probe", lambda module: (True, "stub"))
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        monkeypatch.setattr(envsetup, "numpy_problem", lambda: "")
        state = envsetup.check(segmentation=True)
        assert state["ready"] is False
        assert "nnUNetTrainerDAExt" in state["nnunet_problem"]

    def test_the_message_says_the_failure_is_mid_inference(self, monkeypatch):
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: True)
        monkeypatch.setattr(envsetup, "nnunet_problem",
                            lambda: "nnUNetTrainerDAExt: ImportError: kornia.core")
        monkeypatch.setattr(envsetup, "numpy_problem", lambda: "")
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        # check() only asks about trainer modules when nnunetv2 itself imports, and
        # it does not in the test environment.
        monkeypatch.setattr(envsetup, "_probe", lambda module: (True, "stub"))
        lines: list[str] = []
        envsetup.install(segmentation=True, apt=False, log=lines.append)
        text = "\n".join(lines)
        assert "cannot resolve a trainer class" in text
        assert "die mid-inference" in text, (
            "the reader needs to know this is not an import-time failure")


class TestSmokeTest:
    """--no-deps means an importable package can still have a dead entry point."""

    def test_a_crashing_cli_makes_the_environment_not_ready(self, monkeypatch):
        monkeypatch.setattr(envsetup, "_run", lambda cmd, log: True)
        monkeypatch.setattr(envsetup, "_probe", lambda module: (True, "stub"))
        monkeypatch.setattr(envsetup, "acvl_symbols", lambda: [])
        monkeypatch.setattr(envsetup, "tool_path", lambda name: f"/usr/bin/{name}")
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
