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


def _dicom_zip(path, members=("study/IM1.dcm",)):
    """A zip whose members look like DICOM to `contains_dicom`."""
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name in members:
            zf.writestr(name, b"\0" * 128 + b"DICM" + b"payload")
    return path


def _junk_zip(path):
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("notes.txt", "nothing to do with a spine")
    return path


class TestContainsDicom:
    """Deciding by content, not by file name.

    The real Drive root held nine archives — coursework, a screen recording and two
    copies of the study — and a name-based rule would be wrong in both directions.
    """

    def test_an_archive_with_the_dicm_magic_is_recognised(self, tmp_path):
        from spinelab.discover import contains_dicom

        assert contains_dicom(_dicom_zip(tmp_path / "unnamed.zip", ("a/b/00001",))) is True

    def test_a_dcm_extension_counts_even_without_the_preamble(self, tmp_path):
        import zipfile

        from spinelab.discover import contains_dicom

        path = tmp_path / "noheader.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("scan/slice1.dcm", "no preamble here")
        assert contains_dicom(path) is True

    def test_an_unrelated_archive_is_rejected(self, tmp_path):
        from spinelab.discover import contains_dicom

        assert contains_dicom(_junk_zip(tmp_path / "screen-recording.zip")) is False

    def test_a_corrupt_archive_is_not_a_crash(self, tmp_path):
        from spinelab.discover import contains_dicom

        broken = tmp_path / "truncated.zip"
        broken.write_bytes(b"PK\x03\x04 not really a zip")
        assert contains_dicom(broken) is False


class TestAmbiguity:
    def test_dicom_content_resolves_a_cluttered_folder(self, tmp_path, monkeypatch):
        import spinelab.discover as mod

        _dicom_zip(tmp_path / "study.zip")
        _junk_zip(tmp_path / "10 easy.zip")
        _junk_zip(tmp_path / "coursework.zip")
        monkeypatch.setattr(mod, "LOCAL_PATTERNS",
                            (str(tmp_path / "*.zip").replace("\\", "/"),))
        found = discover(None, allow_repo_lookup=False)
        assert found.source.endswith("study.zip")

    def test_two_dicom_archives_are_still_refused(self, tmp_path, monkeypatch):
        # Two copies of a study is exactly the case where guessing is wrong.
        import spinelab.discover as mod

        _dicom_zip(tmp_path / "study.zip")
        _dicom_zip(tmp_path / "study (1).zip")
        monkeypatch.setattr(mod, "LOCAL_PATTERNS",
                            (str(tmp_path / "*.zip").replace("\\", "/"),))
        found = discover(None, allow_repo_lookup=False)
        assert found.source is None
        assert "ambiguous" in found.how
        assert len(found.candidates) == 2

    def test_ambiguity_does_not_veto_a_later_location(self, tmp_path, monkeypatch):
        """The bug this fixes: a cluttered Drive root blocked every other place.

        Nine archives matched /content/drive/MyDrive/*.zip, discovery gave up there,
        and the copy in the checkout — unambiguous — was never even looked at.
        """
        import spinelab.discover as mod

        drive, checkout = tmp_path / "drive", tmp_path / "checkout"
        drive.mkdir(), checkout.mkdir()
        _dicom_zip(drive / "study.zip")
        _dicom_zip(drive / "study (1).zip")
        _dicom_zip(checkout / "in-checkout.zip")
        monkeypatch.setattr(mod, "LOCAL_PATTERNS",
                            (str(drive / "*.zip").replace("\\", "/"),))
        monkeypatch.setattr(mod, "repo_root", lambda: checkout)
        found = discover(None, allow_repo_lookup=False)
        assert found.source.endswith("in-checkout.zip")

    def test_the_candidates_are_reported_so_one_can_be_chosen(self, tmp_path, monkeypatch):
        import spinelab.discover as mod

        _dicom_zip(tmp_path / "a.zip")
        _dicom_zip(tmp_path / "b.zip")
        monkeypatch.setattr(mod, "LOCAL_PATTERNS",
                            (str(tmp_path / "*.zip").replace("\\", "/"),))
        found = discover(None, allow_repo_lookup=False)
        assert all(c.endswith(".zip") for c in found.candidates)

    def test_the_skip_message_lists_the_candidates(self, tmp_path, monkeypatch):
        import spinelab.discover as mod
        from spinelab.config import Config
        from spinelab.pipeline import run_pipeline

        archives = tmp_path / "drive"
        archives.mkdir()
        _dicom_zip(archives / "one.zip")
        _dicom_zip(archives / "two.zip")
        monkeypatch.setattr(mod, "LOCAL_PATTERNS",
                            (str(archives / "*.zip").replace("\\", "/"),))
        monkeypatch.setattr(mod, "_repo_archive_url", lambda: (None, "disabled in test"))
        cfg = Config(work_dir=tmp_path / "work", dicom_source="")
        results = run_pipeline(cfg, log=lambda *_: None)
        reason = results["ingest"].reason
        assert "one.zip" in reason and "two.zip" in reason
        assert "DICOM_PATH" in reason, "the reader needs to know where to put the answer"


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
