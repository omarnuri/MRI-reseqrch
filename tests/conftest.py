"""Shared test setup.

The suite must run offline, on any machine, in seconds. Study discovery can look
up the GitHub repository it came from, which is useful in Colab and unacceptable in
a test run: it made the suite hang for minutes on a slow connection. It is disabled
here for every test; the tests that cover that lookup call it directly with
`urlopen` replaced.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("SPINELAB_NO_NETWORK", "1")
    monkeypatch.delenv("SPINELAB_STUDY", raising=False)
