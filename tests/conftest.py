from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def anydataset_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "anydataset-home"))
