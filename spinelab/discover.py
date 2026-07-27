"""Find the study without anyone typing a path.

Hard-coding the archive's URL as a default would bake the patient's name into the
repository's source and history (it is in the filename), and would break the moment
the repository goes private. So instead the study is discovered:

1. whatever the caller passed, if it exists;
2. ``SPINELAB_STUDY`` in the environment;
3. a one-line ``study_source.txt`` in the weights cache — on Drive, outside the
   repository, so it is set once and never committed;
4. the usual Drive and Colab locations;
5. the single ``.zip`` in the root of the repository this code came from, resolved
   through the GitHub API from ``git remote`` — no name written down anywhere.

Ambiguity is never resolved by guessing: if more than one candidate matches, all of
them are reported and nothing is chosen.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .runlog import event, get_logger

log = get_logger(__name__)

#: Values that mean "nobody filled this in".
PLACEHOLDERS = {
    "", "none", "null",
    "/content/drive/mydrive/mri/study.zip",
    "/path/to/study.zip",
}

#: Searched in order; the first pattern that yields exactly one file wins.
LOCAL_PATTERNS = (
    "/content/drive/MyDrive/mri/*.zip",
    "/content/drive/MyDrive/mri/**/*.zip",
    "/content/drive/MyDrive/spinelab-data/*.zip",
    "/content/drive/MyDrive/*.zip",
    "/content/*.zip",
)

SOURCE_FILE = "study_source.txt"


def repo_root() -> Path:
    """The checkout this package is running from."""
    return Path(__file__).resolve().parent.parent


def local_patterns() -> tuple[str, ...]:
    """Search patterns, Drive first and the source checkout last.

    The checkout is included because the study archive is committed to this
    repository (which is what docs/PRIVACY.md is about). Finding it there costs a
    glob; the alternative was the GitHub API fallback below, which downloads the
    same 36 MB over the network. It is deliberately last, so a copy on Drive — or
    anything the operator chose explicitly — always wins.
    """
    return LOCAL_PATTERNS + (str(repo_root() / "*.zip"),)


@dataclass
class Discovery:
    source: str | None
    how: str
    candidates: list[str]

    def to_dict(self) -> dict:
        return {"source": self.source, "how": self.how, "candidates": self.candidates}


def is_placeholder(value: str | None) -> bool:
    return not value or str(value).strip().lower() in PLACEHOLDERS


def discover(explicit: str | None = None, *, cache_dir: str | Path | None = None,
             allow_repo_lookup: bool = True) -> Discovery:
    """Locate the study. Returns a Discovery; `source` is None when undecidable."""
    if explicit and not is_placeholder(explicit):
        if str(explicit).startswith(("http://", "https://")):
            return Discovery(str(explicit), "explicit URL", [])
        if Path(explicit).exists():
            return Discovery(str(explicit), "explicit path", [])
        log.warning("explicit dicom_source does not exist: %s — falling back to discovery",
                    explicit)

    env_value = os.environ.get("SPINELAB_STUDY")
    if env_value and not is_placeholder(env_value):
        if env_value.startswith(("http://", "https://")) or Path(env_value).exists():
            return Discovery(env_value, "SPINELAB_STUDY environment variable", [])

    if cache_dir:
        pointer = Path(cache_dir) / SOURCE_FILE
        if pointer.exists():
            value = pointer.read_text(encoding="utf-8").strip().splitlines()
            value = value[0].strip() if value else ""
            if value and not is_placeholder(value):
                return Discovery(value, f"pointer file {pointer}", [])

    for pattern in local_patterns():
        # glob.glob, not Path("/").glob: the latter rejects absolute patterns on
        # Windows outright, so the same code could not be exercised locally.
        matches = sorted(m for m in glob.glob(pattern, recursive=True) if Path(m).is_file())
        if len(matches) == 1:
            return Discovery(matches[0], f"found in {pattern}", matches)
        if len(matches) > 1:
            log.warning("several archives match %s: %s", pattern, matches)
            return Discovery(None, f"ambiguous: {len(matches)} archives match {pattern}",
                             matches)

    # The repository lookup is the only step that touches the network. It is
    # skippable by environment variable so that test runs and offline sessions
    # cannot stall on it.
    if allow_repo_lookup and not os.environ.get("SPINELAB_NO_NETWORK"):
        url, detail = _repo_archive_url()
        if url:
            return Discovery(url, detail, [url])

    return Discovery(None, "nothing found", [])


def resolve_for_config(config) -> str | None:
    """Fill in `config.dicom_source` by discovery, logging how it was decided."""
    found = discover(config.dicom_source, cache_dir=config.cache_dir)
    event("study_discovery", **found.to_dict())
    if found.source:
        log.info("study source: %s (%s)", found.source, found.how)
        config.dicom_source = found.source
    else:
        log.error("could not locate a study: %s", found.how)
    return found.source


def _repo_archive_url() -> tuple[str | None, str]:
    """The single .zip in the root of this checkout's GitHub repository.

    Uses the git remote, so nothing about the patient is written down here. Returns
    (None, reason) when the repository is private, unreachable, or holds more than
    one archive.
    """
    remote = _git_remote()
    if not remote:
        return None, "no git remote to look up"
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", remote)
    if not match:
        return None, f"remote is not GitHub: {remote}"
    owner, repo = match.group(1), match.group(2)

    try:
        import json
        import urllib.parse
        import urllib.request

        api = f"https://api.github.com/repos/{owner}/{repo}/contents/"
        # Short timeout on purpose: this is a convenience, and a stalled lookup
        # must never hold up a run.
        with urllib.request.urlopen(api, timeout=10) as response:  # noqa: S310
            entries = json.load(response)
    except Exception as exc:  # noqa: BLE001
        return None, f"repository listing unavailable ({str(exc)[:60]})"

    archives = [e for e in entries
                if e.get("type") == "file" and str(e.get("name", "")).lower().endswith(".zip")]
    if len(archives) != 1:
        return None, f"{len(archives)} archives in the repository root — cannot choose"
    entry = archives[0]
    url = entry.get("download_url") or (
        f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
        + urllib.parse.quote(entry["name"]))
    size_mb = (entry.get("size") or 0) / 1e6
    return url, f"the only .zip in {owner}/{repo} ({size_mb:.0f} MB)"


def _git_remote() -> str | None:
    for cwd in (Path(__file__).resolve().parent.parent, Path.cwd()):
        try:
            out = subprocess.run(["git", "remote", "get-url", "origin"], cwd=cwd,
                                 capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:  # noqa: BLE001
            continue
    return None
