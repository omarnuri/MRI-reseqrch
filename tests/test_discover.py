"""Study discovery: nobody should have to type a path, and nothing may be guessed."""

from __future__ import annotations

import pytest

from spinelab.discover import discover, is_placeholder


class TestPlaceholders:
    def test_recognises_the_notebook_default(self):
        assert is_placeholder("/content/drive/MyDrive/mri/study.zip") is True
        assert is_placeholder("") is True
        assert is_placeholder(None) is True

    def test_a_real_path_is_not_a_placeholder(self):
        assert is_placeholder("/content/drive/MyDrive/mri/2026-07-27-thoracic.zip") is False


class TestExplicit:
    def test_an_existing_path_wins(self, tmp_path):
        archive = tmp_path / "study.zip"
        archive.write_bytes(b"x")
        found = discover(str(archive), allow_repo_lookup=False)
        assert found.source == str(archive)
        assert found.how == "explicit path"

    def test_a_url_is_taken_at_face_value(self):
        found = discover("https://example.org/study.zip", allow_repo_lookup=False)
        assert found.source == "https://example.org/study.zip"
        assert "URL" in found.how

    def test_a_missing_explicit_path_falls_through(self, tmp_path):
        found = discover(str(tmp_path / "absent.zip"), allow_repo_lookup=False)
        assert found.source is None


class TestEnvironment:
    def test_env_var_is_used(self, tmp_path, monkeypatch):
        archive = tmp_path / "from_env.zip"
        archive.write_bytes(b"x")
        monkeypatch.setenv("SPINELAB_STUDY", str(archive))
        found = discover(None, allow_repo_lookup=False)
        assert found.source == str(archive)
        assert "environment" in found.how

    def test_env_var_pointing_nowhere_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPINELAB_STUDY", str(tmp_path / "nope.zip"))
        found = discover(None, allow_repo_lookup=False)
        assert found.source is None


class TestPointerFile:
    def test_pointer_file_in_the_cache_is_read(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPINELAB_STUDY", raising=False)
        cache = tmp_path / "cache"
        cache.mkdir()
        archive = tmp_path / "study.zip"
        archive.write_bytes(b"x")
        (cache / "study_source.txt").write_text(f"{archive}\n", encoding="utf-8")
        found = discover(None, cache_dir=cache, allow_repo_lookup=False)
        assert found.source == str(archive)
        assert "pointer file" in found.how

    def test_a_url_in_the_pointer_file_works_too(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPINELAB_STUDY", raising=False)
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "study_source.txt").write_text("https://example.org/s.zip", encoding="utf-8")
        found = discover(None, cache_dir=cache, allow_repo_lookup=False)
        assert found.source == "https://example.org/s.zip"

    def test_an_empty_pointer_file_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPINELAB_STUDY", raising=False)
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "study_source.txt").write_text("\n", encoding="utf-8")
        found = discover(None, cache_dir=cache, allow_repo_lookup=False)
        assert found.source is None


class TestNothingIsGuessed:
    def test_no_candidates_reports_why(self, monkeypatch):
        monkeypatch.delenv("SPINELAB_STUDY", raising=False)
        found = discover(None, allow_repo_lookup=False)
        assert found.source is None
        assert found.how == "nothing found"

    def test_ambiguity_refuses_instead_of_picking(self, tmp_path, monkeypatch):
        # Two archives in the same searched location: the pipeline must not choose.
        import spinelab.discover as mod

        monkeypatch.delenv("SPINELAB_STUDY", raising=False)
        (tmp_path / "a.zip").write_bytes(b"x")
        (tmp_path / "b.zip").write_bytes(b"x")
        pattern = str(tmp_path / "*.zip").replace("\\", "/")
        monkeypatch.setattr(mod, "LOCAL_PATTERNS", (pattern,))
        found = discover(None, allow_repo_lookup=False)
        assert found.source is None
        assert "ambiguous" in found.how
        assert len(found.candidates) == 2

    def test_a_single_match_in_a_searched_location_is_used(self, tmp_path, monkeypatch):
        import spinelab.discover as mod

        monkeypatch.delenv("SPINELAB_STUDY", raising=False)
        (tmp_path / "only.zip").write_bytes(b"x")
        pattern = str(tmp_path / "*.zip").replace("\\", "/")
        monkeypatch.setattr(mod, "LOCAL_PATTERNS", (pattern,))
        found = discover(None, allow_repo_lookup=False)
        assert found.source.endswith("only.zip")


class TestCheckoutFallback:
    """The archive is committed to this repository, so the checkout is searched too.

    Without this the only way to reach it was the GitHub API lookup, which downloads
    the same 36 MB over the network — and in Colab the sparse checkout can simply be
    told to include it.
    """

    def test_an_archive_in_the_checkout_is_found(self, tmp_path, monkeypatch):
        import spinelab.discover as mod

        (tmp_path / "study.zip").write_bytes(b"x")
        monkeypatch.setattr(mod, "repo_root", lambda: tmp_path)
        monkeypatch.setattr(mod, "LOCAL_PATTERNS", ())
        found = discover(None, allow_repo_lookup=False)
        assert found.source.endswith("study.zip")

    def test_drive_wins_over_the_checkout(self, tmp_path, monkeypatch):
        # A study the operator put on Drive is the one they meant; the copy in the
        # repository is a historical accident, not a choice.
        import spinelab.discover as mod

        checkout, drive = tmp_path / "repo", tmp_path / "drive"
        checkout.mkdir(), drive.mkdir()
        (checkout / "in-repo.zip").write_bytes(b"x")
        (drive / "on-drive.zip").write_bytes(b"x")
        monkeypatch.setattr(mod, "repo_root", lambda: checkout)
        monkeypatch.setattr(mod, "LOCAL_PATTERNS",
                            (str(drive / "*.zip").replace("\\", "/"),))
        assert discover(None, allow_repo_lookup=False).source.endswith("on-drive.zip")

    def test_the_checkout_pattern_is_last(self, monkeypatch):
        import spinelab.discover as mod

        patterns = mod.local_patterns()
        assert patterns[:len(mod.LOCAL_PATTERNS)] == mod.LOCAL_PATTERNS
        assert patterns[-1].endswith("*.zip")


class TestRepositoryLookup:
    """These must never touch the network — urlopen is always replaced."""

    @staticmethod
    def _fake_listing(monkeypatch, entries):
        import io
        import json
        import urllib.request

        class _Response(io.StringIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: _Response(json.dumps(entries)))

    def test_single_archive_yields_a_raw_url(self, monkeypatch):
        import spinelab.discover as mod

        monkeypatch.setattr(mod, "_git_remote", lambda: "https://github.com/o/r.git")
        self._fake_listing(monkeypatch, [
            {"type": "file", "name": "README.md", "size": 10},
            {"type": "file", "name": "study one.zip", "size": 35_000_000,
             "download_url": "https://raw.example/study%20one.zip"},
            {"type": "dir", "name": "docs"},
        ])
        url, reason = mod._repo_archive_url()
        assert url == "https://raw.example/study%20one.zip"
        assert "only .zip" in reason and "35 MB" in reason

    def test_refuses_when_the_repository_holds_several_archives(self, monkeypatch):
        import spinelab.discover as mod

        monkeypatch.setattr(mod, "_git_remote", lambda: "https://github.com/o/r.git")
        self._fake_listing(monkeypatch, [
            {"type": "file", "name": "a.zip", "size": 1},
            {"type": "file", "name": "b.zip", "size": 2},
        ])
        url, reason = mod._repo_archive_url()
        assert url is None
        assert "2 archives" in reason

    def test_unreachable_repository_is_reported_not_raised(self, monkeypatch):
        import urllib.request

        import spinelab.discover as mod

        monkeypatch.setattr(mod, "_git_remote", lambda: "https://github.com/o/r.git")
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("private repo")))
        url, reason = mod._repo_archive_url()
        assert url is None
        assert "unavailable" in reason

    def test_non_github_remote_is_reported(self, monkeypatch):
        import spinelab.discover as mod

        monkeypatch.setattr(mod, "_git_remote", lambda: "git@gitlab.com:o/r.git")
        url, reason = mod._repo_archive_url()
        assert url is None
        assert "not GitHub" in reason

    def test_no_remote_is_reported(self, monkeypatch):
        import spinelab.discover as mod

        monkeypatch.setattr(mod, "_git_remote", lambda: None)
        url, reason = mod._repo_archive_url()
        assert url is None
        assert "no git remote" in reason


class TestIngestUsesDiscovery:
    def test_ingest_skip_message_explains_the_three_ways_to_fix_it(self, tmp_path, monkeypatch):
        import spinelab.discover as mod
        from spinelab.config import Config
        from spinelab.pipeline import run_pipeline

        monkeypatch.delenv("SPINELAB_STUDY", raising=False)
        monkeypatch.setattr(mod, "LOCAL_PATTERNS", ())
        monkeypatch.setattr(mod, "_repo_archive_url", lambda: (None, "lookup disabled in test"))
        cfg = Config(work_dir=tmp_path / "work",
                     dicom_source="/content/drive/MyDrive/mri/study.zip")
        results = run_pipeline(cfg, log=lambda *_: None)
        reason = results["ingest"].reason
        assert "no study found" in reason
        assert "Drive" in reason and "study_source.txt" in reason and "SPINELAB_STUDY" in reason


@pytest.mark.parametrize("value", ["", "none", "NULL", "/PATH/TO/STUDY.ZIP"])
def test_placeholder_matching_is_case_insensitive(value):
    assert is_placeholder(value) is True
