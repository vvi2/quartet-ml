"""Smoke test: confirms the package installs and imports cleanly."""

import quartet_ml


def test_import() -> None:
    assert quartet_ml is not None
