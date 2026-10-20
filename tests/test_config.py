from pathlib import Path

import pytest

from meeting_notes.config import Settings


def test_model_dir_defaults_to_d_drive() -> None:
    assert Settings.model_fields["model_dir"].default == Path("D:/models/whisper")


def test_model_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEETING_NOTES_MODEL_DIR", str(tmp_path))
    assert Settings().model_dir == tmp_path
