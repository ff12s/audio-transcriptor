import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from meeting_notes import cli
from meeting_notes.errors import MeetingNotesError
from meeting_notes.models import Segment, Transcript
from meeting_notes.track import TrackWriter
from tests.conftest import FakeTranscriber

runner = CliRunner()


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)


class FakeRecorder:
    def __init__(self, me: Path, them: Path) -> None:
        self.me, self.them = me, them
        self.recorded = False

    def record_until_interrupted(self, on_tick: Callable[[float], None] | None = None) -> float:
        self.recorded = True
        _write_wav(self.me)
        _write_wav(self.them)
        if on_tick is not None:
            on_tick(1.0)
        return 1.0


class Env:
    def __init__(self) -> None:
        self.recorders: list[FakeRecorder] = []
        self.found: list[str] = []


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Env:
    state = Env()
    transcriber = FakeTranscriber([Segment(start=1.0, end=2.0, text=" Привет.")], duration=2.0)

    def build_recorder(me: Path, them: Path, pids: list[int], on_warning: Any) -> FakeRecorder:
        state.recorders.append(FakeRecorder(me, them))
        return state.recorders[-1]

    def find_pids(app: str) -> list[int]:
        state.found.append(app)
        return [10]

    monkeypatch.setenv("MEETING_NOTES_GLOSSARY_PATH", str(tmp_path / "glossary.txt"))
    monkeypatch.setattr(cli, "build_transcriber", lambda *_args: transcriber)
    monkeypatch.setattr(cli, "build_recorder", build_recorder)
    monkeypatch.setattr(cli, "find_capture_pids", find_pids)
    monkeypatch.setattr(cli, "windows_build", lambda: 22631)
    return state


def _invoke(*args: str) -> Any:
    return runner.invoke(cli.record_app, list(args))


def test_record_then_transcribe(tmp_path: Path, env: Env) -> None:
    base = tmp_path / "standup"
    result = _invoke(str(base), "--app", "chrome")
    assert result.exit_code == 0, result.output
    assert env.found == ["chrome"]
    text = (tmp_path / "standup.md").read_text(encoding="utf-8")
    assert "**Я:** Привет." in text
    assert "**Собеседники:** Привет." in text


def test_missing_app_flag(tmp_path: Path, env: Env) -> None:
    result = _invoke(str(tmp_path / "standup"))
    assert result.exit_code == 1
    assert "--app" in result.output
    assert env.recorders == []


def test_app_not_found_creates_nothing(
    tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(app: str) -> list[int]:
        raise MeetingNotesError(f"Процесс {app!r} не найден.")

    monkeypatch.setattr(cli, "find_capture_pids", missing)
    result = _invoke(str(tmp_path / "standup"), "--app", "teams")
    assert result.exit_code == 1
    assert "не найден" in result.output
    assert list(tmp_path.iterdir()) == []
    assert env.recorders == []


def test_missing_directory_before_recording(tmp_path: Path, env: Env) -> None:
    result = _invoke(str(tmp_path / "nope" / "standup"), "--app", "chrome")
    assert result.exit_code == 1
    assert "Каталог не существует" in result.output
    assert env.recorders == []


def test_existing_md_before_recording(tmp_path: Path, env: Env) -> None:
    (tmp_path / "standup.md").write_text("old", encoding="utf-8")
    result = _invoke(str(tmp_path / "standup"), "--app", "chrome")
    assert result.exit_code == 1
    assert "--overwrite" in result.output
    assert env.recorders == []
    assert (tmp_path / "standup.md").read_text(encoding="utf-8") == "old"


def test_existing_tracks_before_recording(tmp_path: Path, env: Env) -> None:
    (tmp_path / "standup.me.wav").write_bytes(b"keep")
    result = _invoke(str(tmp_path / "standup"), "--app", "chrome", "--overwrite")
    assert result.exit_code == 1
    assert "--transcribe-only" in result.output
    assert env.recorders == []
    assert (tmp_path / "standup.me.wav").read_bytes() == b"keep"


def test_old_windows(tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "windows_build", lambda: 19045)
    result = _invoke(str(tmp_path / "standup"), "--app", "chrome")
    assert result.exit_code == 1
    assert "Windows 11" in result.output
    assert env.recorders == []


def test_mic_missing_message(tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_mic(*_args: object) -> None:
        raise MeetingNotesError("Микрофон не найден: нет устройства.")

    monkeypatch.setattr(cli, "build_recorder", no_mic)
    result = _invoke(str(tmp_path / "standup"), "--app", "chrome")
    assert result.exit_code == 1
    assert "Микрофон не найден" in result.output
    assert "Traceback" not in result.output
    assert list(tmp_path.iterdir()) == []


def test_transcription_failure_keeps_tracks(
    tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failing(FakeTranscriber):
        def transcribe(self, audio: Path) -> Transcript:
            raise MeetingNotesError("Не хватило видеопамяти.")

    monkeypatch.setattr(cli, "build_transcriber", lambda *_args: Failing([], 0.0))
    base = tmp_path / "standup"
    result = _invoke(str(base), "--app", "chrome")
    assert result.exit_code == 1
    assert "Не хватило видеопамяти" in result.output
    assert "--transcribe-only" in result.output
    # Подсказку копируют из терминала, поэтому путь и команда должны быть целыми, без переносов.
    assert str(tmp_path / "standup.me.wav") in result.output
    assert str(tmp_path / "standup.them.wav") in result.output
    command = f'uv run record --transcribe-only "{base}"'
    assert any(command in line for line in result.output.splitlines())
    assert (tmp_path / "standup.me.wav").exists()
    assert (tmp_path / "standup.them.wav").exists()


def test_transcribe_only(tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "windows_build", lambda: 10)
    _write_wav(tmp_path / "standup.me.wav")
    _write_wav(tmp_path / "standup.them.wav")
    result = _invoke(str(tmp_path / "standup"), "--transcribe-only")
    assert result.exit_code == 0, result.output
    assert env.recorders == []
    assert (tmp_path / "standup.md").exists()


def test_transcribe_only_without_tracks(tmp_path: Path, env: Env) -> None:
    _write_wav(tmp_path / "standup.me.wav")
    result = _invoke(str(tmp_path / "standup"), "--transcribe-only")
    assert result.exit_code == 1
    assert "standup.them.wav" in result.output.replace("\n", "")
    assert env.recorders == []


def test_unwritable_track_is_user_facing(
    tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    monkeypatch.setenv("MEETING_NOTES_GLOSSARY_PATH", str(tmp_path / "glossary.txt"))
    monkeypatch.setattr(cli, "find_capture_pids", lambda _app: [10])
    monkeypatch.setattr(cli, "windows_build", lambda: 22631)
    monkeypatch.setattr(cli, "MicSource", lambda: object())

    def denied(self: object, path: Path) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(TrackWriter, "__init__", denied)
    result = _invoke(str(tmp_path / "standup"), "--app", "chrome")
    assert result.exit_code == 1
    assert "Не удалось создать файл" in result.output
    assert "Traceback" not in result.output


def test_recording_failure_mentions_partial_tracks(
    tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(FakeRecorder):
        def record_until_interrupted(self, on_tick: Callable[[float], None] | None = None) -> float:
            raise MeetingNotesError("Не удалось записать дорожку: диск полон.")

    monkeypatch.setattr(cli, "build_recorder", lambda me, them, pids, w: Broken(me, them))
    result = _invoke(str(tmp_path / "standup"), "--app", "chrome")
    assert result.exit_code == 1
    assert "диск полон" in result.output
    assert "Частично записано" in result.output


def test_foreign_recording_error_still_prints_partial_tracks(
    tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom(FakeRecorder):
        def record_until_interrupted(self, on_tick: Callable[[float], None] | None = None) -> float:
            raise RuntimeError("драйвер упал")

    monkeypatch.setattr(cli, "build_recorder", lambda me, them, pids, w: Boom(me, them))
    result = _invoke(str(tmp_path / "standup"), "--app", "chrome")
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert str(tmp_path / "standup.me.wav") in result.output
    assert str(tmp_path / "standup.them.wav") in result.output
    assert "Частично записано" in result.output


def test_foreign_transcription_error_still_prints_hint(
    tmp_path: Path, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom(FakeTranscriber):
        def transcribe(self, audio: Path) -> Transcript:
            raise RuntimeError("ctranslate2 упал")

    monkeypatch.setattr(cli, "build_transcriber", lambda *_args: Boom([], 0.0))
    result = _invoke(str(tmp_path / "standup"), "--app", "chrome")
    assert result.exit_code != 0
    assert "--transcribe-only" in result.output
