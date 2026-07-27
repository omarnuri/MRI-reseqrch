"""Shared test setup.

The suite must run offline, on any machine, in seconds. Study discovery can look
up the GitHub repository it came from, which is useful in Colab and unacceptable in
a test run: it made the suite hang for minutes on a slow connection. It is disabled
here for every test; the tests that cover that lookup call it directly with
`urlopen` replaced.

Discovery also searches the checkout it is running from, because the study archive is
committed to this repository. That must not leak into tests: whether a developer
happens to have a .zip in their working tree would otherwise change what the suite
does. Every test sees an empty checkout unless it says otherwise.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("SPINELAB_NO_NETWORK", "1")
    monkeypatch.delenv("SPINELAB_STUDY", raising=False)


@pytest.fixture(autouse=True)
def _no_study_in_the_checkout(monkeypatch, tmp_path_factory):
    import spinelab.discover as discover

    empty = tmp_path_factory.mktemp("empty-checkout")
    monkeypatch.setattr(discover, "repo_root", lambda: empty)
