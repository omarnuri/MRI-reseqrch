"""Drive is optional. The run that died proving otherwise is what these guard.

`ValueError: mount failed` in the first cell aborted everything downstream, and the
cell had already created the cache directory under the unmounted mount point — which
is what makes the *next* mount fail too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spinelab import colab


@pytest.fixture()
def unmounted(tmp_path):
    """A mount point that does not exist: Drive was never mounted."""
    return str(tmp_path / "drive")


@pytest.fixture()
def mounted(tmp_path):
    """A mount point with MyDrive inside it: Drive is usable."""
    point = tmp_path / "drive"
    (point / "MyDrive").mkdir(parents=True)
    return str(point)


class TestDriveDetection:
    def test_a_missing_mountpoint_is_not_mounted(self, unmounted):
        assert colab.drive_is_mounted(unmounted) is False

    def test_mydrive_inside_the_mountpoint_counts_as_mounted(self, mounted):
        assert colab.drive_is_mounted(mounted) is True

    def test_a_leftover_stub_is_not_mistaken_for_a_mount(self, tmp_path):
        # The exact shape of the poisoning: a real directory tree at the mount
        # point, created by a previous cell after the mount had already failed.
        stub = tmp_path / "drive"
        (stub / "MyDrive2" / "spinelab-cache").mkdir(parents=True)
        assert colab.drive_is_mounted(str(stub)) is False
        assert colab.stub_at_mountpoint(str(stub)) is True

    def test_an_empty_directory_is_not_a_stub(self, tmp_path):
        # Drive mounts happily into an empty directory, so this is not a problem.
        empty = tmp_path / "drive"
        empty.mkdir()
        assert colab.stub_at_mountpoint(str(empty)) is False


class TestCacheResolution:
    def test_an_unmounted_drive_path_falls_back_to_local(self, unmounted, tmp_path):
        resolved, note = colab.resolve_cache_dir(
            f"{unmounted}/MyDrive/spinelab-cache",
            mountpoint=unmounted, fallback=str(tmp_path / "local"))
        assert resolved == tmp_path / "local"
        assert "not mounted" in note
        assert "re-downloaded" in note, "the cost of the fallback has to be stated"

    def test_a_mounted_drive_path_is_kept(self, mounted, tmp_path):
        preferred = f"{mounted}/MyDrive/spinelab-cache"
        resolved, note = colab.resolve_cache_dir(preferred, mountpoint=mounted,
                                                fallback=str(tmp_path / "local"))
        assert resolved == Path(preferred)
        assert note == ""

    def test_a_path_outside_drive_is_never_second_guessed(self, unmounted, tmp_path):
        own = tmp_path / "somewhere-else"
        resolved, note = colab.resolve_cache_dir(own, mountpoint=unmounted,
                                                fallback=str(tmp_path / "local"))
        assert resolved == own
        assert note == ""


class TestPrepare:
    def test_nothing_is_created_under_an_unmounted_mountpoint(self, unmounted, tmp_path):
        """The bug that made a failed mount permanent for the whole session."""
        colab.prepare(f"{unmounted}/MyDrive/spinelab-cache", mount=False,
                      mountpoint=unmounted, fallback=str(tmp_path / "local"),
                      log=lambda *_: None)
        assert not Path(unmounted).exists(), (
            "creating the cache under an unmounted Drive leaves a stub that makes "
            "every later drive.mount() fail with 'directory not empty'")
        assert (tmp_path / "local").is_dir()

    def test_a_failed_mount_does_not_raise(self, unmounted, tmp_path, monkeypatch):
        monkeypatch.setattr(colab, "in_colab", lambda: True)
        state = colab.prepare(f"{unmounted}/MyDrive/spinelab-cache",
                              mountpoint=unmounted, fallback=str(tmp_path / "local"),
                              log=lambda *_: None)
        assert state["drive_mounted"] is False
        assert state["cache_dir"] == str(tmp_path / "local")

    def test_the_reason_a_mount_failed_is_reported(self, unmounted, tmp_path, monkeypatch):
        monkeypatch.setattr(colab, "in_colab", lambda: True)
        lines: list[str] = []
        colab.prepare(f"{unmounted}/MyDrive/spinelab-cache", mountpoint=unmounted,
                      fallback=str(tmp_path / "local"), log=lines.append)
        text = "\n".join(lines)
        assert "not mounted" in text
        assert "unavailable" in text or "mount failed" in text

    def test_a_mounted_drive_is_used(self, mounted, tmp_path):
        state = colab.prepare(f"{mounted}/MyDrive/spinelab-cache", mount=False,
                              mountpoint=mounted, fallback=str(tmp_path / "local"),
                              log=lambda *_: None)
        assert Path(state["cache_dir"]) == Path(mounted, "MyDrive", "spinelab-cache")
        assert Path(state["cache_dir"]).is_dir()

    def test_cache_sizes_are_reported_per_tool(self, tmp_path):
        weights = tmp_path / "weights" / "spineps"
        weights.mkdir(parents=True)
        (weights / "model.pth").write_bytes(b"x" * 2000)
        sizes = colab.cache_sizes(tmp_path)
        assert set(sizes) == set(colab.WEIGHT_CACHES)
        assert sizes["spineps"] == pytest.approx(2000 / 1e9)
        assert sizes["totalsegmentator"] == 0.0


class TestMountDrive:
    def test_off_colab_it_declines_instead_of_importing_google(self, unmounted, monkeypatch):
        monkeypatch.setattr(colab, "in_colab", lambda: False)
        mounted, reason = colab.mount_drive(unmounted, log=lambda *_: None)
        assert mounted is False
        assert reason == "not running on Colab"

    def test_a_stub_mountpoint_is_explained_with_the_fix(self, tmp_path, monkeypatch):
        stub = tmp_path / "drive"
        (stub / "leftover").mkdir(parents=True)
        monkeypatch.setattr(colab, "in_colab", lambda: True)
        lines: list[str] = []
        colab.mount_drive(str(stub), log=lines.append)
        text = "\n".join(lines)
        assert "will not mount" in text
        assert "rm -rf" in text, "the user needs the actual command, not a hint"
