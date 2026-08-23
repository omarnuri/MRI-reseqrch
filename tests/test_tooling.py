"""Finding the tools, and not mistaking their exit noise for a failure.

Both bugs here came out of the first working local install:

* every segmentation stage skipped with "not on PATH" while `spineps.exe` sat in the
  same directory as the interpreter that was asking — `venv/python -m spinelab`
  does not put `Scripts/` on PATH, only `activate` does;
* SPINEPS prints a citation banner from an atexit callback, and on a cp1251 console
  that raises UnicodeEncodeError *after* the work is done. The traceback is the last
  thing in stderr, so it became the reported reason for a successful run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from spinelab.utils import ATEXIT_MARKER, child_env, clean_reason, strip_atexit_noise, tool_path

SPINEPS_ATEXIT_NOISE = f"""\
------------------------- Thank you for using SPINEPS -------------------------
{ATEXIT_MARKER}: <function print_citation_reminder at 0x0>
Traceback (most recent call last):
  File "spineps/utils/citation_reminder.py", line 42, in print_citation_reminder
    console.rule()
UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-78
"""


class TestToolPath:
    def test_a_script_beside_the_interpreter_is_found(self, tmp_path, monkeypatch):
        scripts = tmp_path / "Scripts"
        scripts.mkdir()
        exe = scripts / ("fake-tool.exe" if sys.platform == "win32" else "fake-tool")
        exe.write_text("", encoding="utf-8")
        exe.chmod(0o755)
        monkeypatch.setattr(sys, "executable", str(scripts / "python.exe"))
        assert tool_path("fake-tool") is not None

    def test_path_is_still_consulted_as_a_fallback(self, tmp_path, monkeypatch):
        # On Colab the tools land in /usr/local/bin, which is on PATH and is not
        # beside the interpreter.
        elsewhere = tmp_path / "bin"
        elsewhere.mkdir()
        exe = elsewhere / ("other-tool.exe" if sys.platform == "win32" else "other-tool")
        exe.write_text("", encoding="utf-8")
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", str(elsewhere))
        monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python.exe"))
        assert tool_path("other-tool") is not None

    def test_a_missing_tool_is_still_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path))
        assert tool_path("definitely-not-installed-anywhere") is None


class TestChildEnv:
    def test_utf8_is_forced_so_rich_cannot_crash_on_a_legacy_console(self, monkeypatch):
        # The default only applies when the parent has not chosen an encoding, and
        # some shells do (Claude Code exports "utf-8:surrogateescape"). Clearing it
        # keeps this test about the default rather than about the developer's shell.
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        assert child_env()["PYTHONIOENCODING"] == "utf-8"

    def test_extra_variables_are_merged(self):
        env = child_env(SPINEPS_SEGMENTOR_MODELS="/weights/spineps")
        assert env["SPINEPS_SEGMENTOR_MODELS"] == "/weights/spineps"
        assert "PATH" in env, "the parent environment must be inherited, not replaced"

    def test_an_explicit_encoding_choice_is_respected(self, monkeypatch):
        monkeypatch.setenv("PYTHONIOENCODING", "latin-1")
        assert child_env()["PYTHONIOENCODING"] == "latin-1"


class TestRunTool:
    """A run that dies mid-inference has to leave evidence behind.

    With `capture_output=True` the tool's entire output sits in a pipe until it
    exits. A local SPINEPS run was killed between two model phases after 47 minutes
    and left no log line, no stage marker and no explanation — the output died with
    the pipe.
    """

    def test_output_is_on_disk_before_the_process_exits(self, tmp_path):
        import subprocess
        import sys

        log = tmp_path / "tool.log"
        marker = tmp_path / "seen.txt"
        # The child writes a line, then waits until the test has read the log —
        # proving the line is readable while the process is still alive.
        script = (
            "import sys, time, pathlib\n"
            "print('phase 1 done', flush=True)\n"
            "p = pathlib.Path(sys.argv[1])\n"
            "for _ in range(200):\n"
            "    if p.exists(): break\n"
            "    time.sleep(0.05)\n"
            "print('phase 2 done', flush=True)\n"
        )
        import threading

        from spinelab.utils import run_tool

        def watch():
            for _ in range(200):
                if log.exists() and "phase 1 done" in log.read_text(encoding="utf-8"):
                    marker.write_text("ok", encoding="utf-8")
                    return
                import time as t
                t.sleep(0.05)

        watcher = threading.Thread(target=watch)
        watcher.start()
        proc = run_tool([sys.executable, "-c", script, str(marker)],
                        log_path=log, timeout=60)
        watcher.join()
        assert marker.exists(), "the log was not readable until the process had exited"
        assert proc.returncode == 0
        assert "phase 1 done" in proc.stdout and "phase 2 done" in proc.stdout
        assert isinstance(proc, subprocess.CompletedProcess)

    def test_the_command_is_recorded_at_the_top(self, tmp_path):
        import sys

        from spinelab.utils import run_tool

        log = tmp_path / "tool.log"
        run_tool([sys.executable, "-c", "pass"], log_path=log, timeout=60)
        assert log.read_text(encoding="utf-8").startswith("$ ")

    def test_stderr_is_merged_so_the_order_survives(self, tmp_path):
        import sys

        from spinelab.utils import run_tool

        script = ("import sys\n"
                  "print('to stdout', flush=True)\n"
                  "print('to stderr', file=sys.stderr, flush=True)\n")
        proc = run_tool([sys.executable, "-c", script],
                        log_path=tmp_path / "tool.log", timeout=60)
        assert "to stdout" in proc.stdout and "to stderr" in proc.stdout
        assert proc.stderr == "", "callers read stderr-or-stdout; both must not duplicate"

    def test_a_nonzero_exit_is_reported_with_its_output(self, tmp_path):
        import sys

        from spinelab.utils import run_tool

        proc = run_tool([sys.executable, "-c", "raise SystemExit(3)"],
                        log_path=tmp_path / "tool.log", timeout=60)
        assert proc.returncode == 3

    def test_a_timeout_still_raises_so_the_stage_can_report_it(self, tmp_path):
        import subprocess
        import sys

        import pytest as _pytest

        from spinelab.utils import run_tool

        with _pytest.raises(subprocess.TimeoutExpired):
            run_tool([sys.executable, "-c", "import time; time.sleep(30)"],
                     log_path=tmp_path / "tool.log", timeout=1)


class TestToolEnvSurvivesThirdPartyImports:
    """`import totalspineseg` runs, at module level:

        os.environ["nnUNet_results"] = './nnUNet_results'

    Our own environment probe imports it to read its version, so the absolute cache
    path set moments earlier became a path relative to the working directory — and a
    tool that trusted os.environ would have written its weights into the repository.
    """

    def test_the_cache_locations_win_over_a_clobbered_environ(self, tmp_path, monkeypatch):
        from spinelab.config import Config

        config = Config(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
        exported = config.export_env()
        expected = exported["nnUNet_results"]

        monkeypatch.setenv("nnUNet_results", "./nnUNet_results")  # what the import does
        assert child_env()["nnUNet_results"] == expected
        assert Path(child_env()["nnUNet_results"]).is_absolute()

    def test_explicit_arguments_still_win_over_the_recorded_values(self, tmp_path):
        from spinelab.config import Config

        Config(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache").export_env()
        env = child_env(nnUNet_results="/somewhere/deliberate")
        assert env["nnUNet_results"] == "/somewhere/deliberate"

    def test_unrelated_variables_are_untouched(self, tmp_path, monkeypatch):
        from spinelab.config import Config

        Config(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache").export_env()
        monkeypatch.setenv("SOME_USER_VARIABLE", "keep me")
        assert child_env()["SOME_USER_VARIABLE"] == "keep me"


class TestAtexitNoise:
    def test_an_atexit_traceback_is_dropped(self):
        cleaned = strip_atexit_noise(f"real work happened\n{SPINEPS_ATEXIT_NOISE}")
        assert "real work happened" in cleaned
        assert "Traceback" not in cleaned
        assert "UnicodeEncodeError" not in cleaned

    def test_the_reason_reported_is_the_real_error_not_the_exit_noise(self):
        # This is the whole point: the genuine failure is above the banner.
        text = ("ERROR: could not load model t2w\n" + SPINEPS_ATEXIT_NOISE)
        assert clean_reason(text) == "ERROR: could not load model t2w"

    def test_output_without_an_atexit_block_is_untouched(self):
        assert strip_atexit_noise("plain output\n") == "plain output\n"

    def test_the_citation_banner_alone_is_not_a_reason(self):
        # Nothing failed here, but if it had, "Thank you for using SPINEPS" is not
        # the explanation.
        assert clean_reason(SPINEPS_ATEXIT_NOISE, "no output produced") == "no output produced"

    def test_a_decorative_rule_is_not_a_reason(self):
        assert clean_reason("real error here\n" + "-" * 60) == "real error here"

    def test_a_short_line_of_dashes_is_not_treated_as_a_rule(self):
        # `--iso` and similar must survive: only long runs of rule characters go.
        assert clean_reason("bad flag: --iso") == "bad flag: --iso"

    def test_a_real_traceback_still_survives(self):
        # Only *ignored atexit* exceptions are noise. A genuine crash must remain.
        text = ('Traceback (most recent call last):\n'
                "RuntimeError: CUDA out of memory\n")
        assert "RuntimeError: CUDA out of memory" in strip_atexit_noise(text)
        assert clean_reason(text) == "RuntimeError: CUDA out of memory"
