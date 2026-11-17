import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from meeting_notes import cli
from meeting_notes.asr.gigaam import GigaamTranscriber
from meeting_notes.asr.transcriber import WhisperTranscriber
from meeting_notes.config import Settings
from tests.conftest import FakeTranscriber

runner = CliRunner()
# Сохраняем настоящую фабрику до того, как фикстура её подменит.
_real_build_transcriber = cli.build_transcriber

_ENV = (
    "MEETING_NOTES_ENGINE",
    "MEETING_NOTES_MODEL",
    "MEETING_NOTES_GIGAAM_MODEL",
    "MEETING_NOTES_DEVICE",
    "MEETING_NOTES_LANGUAGE",
    "MEETING_NOTES_COMPUTE_TYPE",
)


@pytest.fixture(autouse=True)
def calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_transcriber: FakeTranscriber
) -> list[tuple[Settings, str | None]]:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEETING_NOTES_GLOSSARY_PATH", str(tmp_path / "glossary.txt"))
    recorded: list[tuple[Settings, str | None]] = []

    def build(settings: Settings, prompt: str | None, _on_progress: Any) -> FakeTranscriber:
        recorded.append((settings, prompt))
        return fake_transcriber

    monkeypatch.setattr(cli, "build_transcriber", build)
    return recorded


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"x")
    return path


def test_whisper_without_glossary_file_warns_about_empty_glossary(audio: Path) -> None:
    result = runner.invoke(cli.app, [str(audio), "--engine", "whisper"])

    assert result.exit_code == 0, result.output
    assert "Глоссарий пуст или не найден" in result.output


def test_whisper_run_prints_engine_and_whisper_model(audio: Path) -> None:
    result = runner.invoke(cli.app, [str(audio), "--engine", "whisper", "--model", "tiny"])

    assert result.exit_code == 0, result.output
    assert "Движок: whisper, модель: tiny" in result.output


def test_model_option_with_whisper_engine_sets_whisper_model(
    audio: Path, calls: list[tuple[Settings, str | None]]
) -> None:
    result = runner.invoke(cli.app, [str(audio), "--engine", "whisper", "--model", "tiny"])

    assert result.exit_code == 0, result.output
    ((settings, _prompt),) = calls
    assert settings.model == "tiny"
    assert settings.gigaam_model == "gigaam-v3-e2e-rnnt"


def test_model_option_without_engine_sets_gigaam_model(
    audio: Path, calls: list[tuple[Settings, str | None]]
) -> None:
    result = runner.invoke(cli.app, [str(audio), "--model", "gigaam-v3-rnnt"])

    assert result.exit_code == 0, result.output
    ((settings, _prompt),) = calls
    assert settings.gigaam_model == "gigaam-v3-rnnt"
    assert settings.model == "large-v3"
    assert "Движок: gigaam, модель: gigaam-v3-rnnt" in result.output


def test_unknown_engine_from_environment_is_rejected(
    audio: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[Settings, str | None]]
) -> None:
    monkeypatch.setenv("MEETING_NOTES_ENGINE", "foo")

    result = runner.invoke(cli.app, [str(audio)])

    assert result.exit_code == 1, result.output
    assert "MEETING_NOTES_ENGINE" in result.output.replace("\n", "")
    assert calls == []


def test_whisper_model_name_is_rejected_under_default_engine(
    audio: Path, calls: list[tuple[Settings, str | None]]
) -> None:
    result = runner.invoke(cli.app, [str(audio), "--model", "large-v3"])

    assert result.exit_code == 1, result.output
    assert "--engine whisper" in result.output.replace("\n", "")
    assert calls == []


def test_non_russian_language_is_rejected_under_default_engine(
    audio: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[Settings, str | None]]
) -> None:
    monkeypatch.setenv("MEETING_NOTES_LANGUAGE", "en")

    result = runner.invoke(cli.app, [str(audio)])

    assert result.exit_code == 1, result.output
    assert calls == []


def test_record_rejects_ctranslate2_device_string_under_default_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorders: list[str] = []

    def build_recorder(*_args: Any) -> Any:
        recorders.append("called")
        raise AssertionError("build_recorder must not be reached")

    monkeypatch.setattr(cli, "windows_build", lambda: 22631)
    monkeypatch.setattr(cli, "find_capture_pids", lambda _app: [10])
    monkeypatch.setattr(cli, "build_recorder", build_recorder)
    monkeypatch.setenv("MEETING_NOTES_DEVICE", "cuda:0")

    result = runner.invoke(cli.record_app, [str(tmp_path / "standup"), "--app", "chrome"])

    assert result.exit_code == 1, result.output
    assert recorders == []


def test_factory_builds_whisper_transcriber_for_whisper_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # None в sys.modules: любой импорт (в том числе уже загруженного модуля) даёт ImportError.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    settings = Settings(engine="whisper", device="cpu")

    transcriber = _real_build_transcriber(settings, None, lambda _done, _total: None)

    assert isinstance(transcriber, WhisperTranscriber)


def test_factory_builds_gigaam_transcriber_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnx_asr", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    transcriber = _real_build_transcriber(Settings(), None, lambda _done, _total: None)

    assert isinstance(transcriber, GigaamTranscriber)
