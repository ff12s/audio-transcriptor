import sys
from pathlib import Path

import pytest

from meeting_notes.config import Settings


def test_model_dir_default_depends_on_platform() -> None:
    expected = (
        Path("D:/models/whisper")
        if sys.platform == "win32"
        else Path.home() / ".cache" / "meeting-notes"
    )
    assert Settings.model_fields["model_dir"].default == expected


def test_engine_defaults() -> None:
    fields = Settings.model_fields
    assert fields["engine"].default == "gigaam"
    assert fields["gigaam_model"].default == "gigaam-v3-e2e-rnnt"
    assert fields["device"].default == "auto"
    assert fields["model"].default == "large-v3"


def test_engine_and_gigaam_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEETING_NOTES_ENGINE", "whisper")
    monkeypatch.setenv("MEETING_NOTES_GIGAAM_MODEL", "gigaam-v3-rnnt")
    settings = Settings()
    assert settings.engine == "whisper"
    assert settings.gigaam_model == "gigaam-v3-rnnt"


def test_model_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEETING_NOTES_MODEL_DIR", str(tmp_path))
    assert Settings().model_dir == tmp_path
