from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.conftest import FakeTranscriber

runner = CliRunner()

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
) -> list[tuple[Any, Any]]:
    from meeting_notes import cli

    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEETING_NOTES_GLOSSARY_PATH", str(tmp_path / "glossary.txt"))
    recorded: list[tuple[Any, Any]] = []

    def build(settings: Any, prompt: Any, _on_progress: Any) -> FakeTranscriber:
        recorded.append((settings, prompt))
        return fake_transcriber

    monkeypatch.setattr(cli, "build_transcriber", build)
    return recorded


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"x")
    return path


def _patch_record(monkeypatch: pytest.MonkeyPatch, build: int) -> list[str]:
    from meeting_notes import cli

    recorders: list[str] = []

    def build_recorder(*_args: Any) -> Any:
        recorders.append("called")
        raise AssertionError("build_recorder must not be reached")

    monkeypatch.setattr(cli, "windows_build", lambda: build)
    monkeypatch.setattr(cli, "find_capture_pids", lambda _app: [10])
    monkeypatch.setattr(cli, "build_recorder", build_recorder)
    return recorders


def test_default_engine_is_gigaam_without_prompt(
    tmp_path: Path, calls: list[tuple[Any, Any]]
) -> None:
    from meeting_notes import cli

    result = runner.invoke(cli.app, [str(_audio(tmp_path))])

    assert result.exit_code == 0, result.output
    ((settings, prompt),) = calls
    assert settings.engine == "gigaam"
    assert settings.gigaam_model == "gigaam-v3-e2e-rnnt"
    assert prompt is None


def test_gigaam_run_ignores_glossary_and_prints_engine_line(tmp_path: Path) -> None:
    from meeting_notes import cli

    (tmp_path / "glossary.txt").write_text("Kubernetes\nTerraform\n", encoding="utf-8")

    result = runner.invoke(cli.app, [str(_audio(tmp_path))])

    assert result.exit_code == 0, result.output
    assert "Глоссарий" not in result.output
    assert "Whisper" not in result.output
    assert "Движок: gigaam" in result.output


def test_whisper_engine_normalises_auto_device_and_builds_glossary_prompt(
    tmp_path: Path, calls: list[tuple[Any, Any]]
) -> None:
    from meeting_notes import cli
    from meeting_notes.glossary import build_prompt

    (tmp_path / "glossary.txt").write_text("Kubernetes\nTerraform\n", encoding="utf-8")

    result = runner.invoke(cli.app, [str(_audio(tmp_path)), "--engine", "whisper"])

    assert result.exit_code == 0, result.output
    ((settings, prompt),) = calls
    assert settings.engine == "whisper"
    assert settings.device == "cuda"
    assert settings.model == "large-v3"
    assert prompt == build_prompt(["Kubernetes", "Terraform"])


def test_whisper_engine_keeps_explicit_ctranslate2_device_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[Any, Any]]
) -> None:
    from meeting_notes import cli

    monkeypatch.setenv("MEETING_NOTES_ENGINE", "whisper")
    monkeypatch.setenv("MEETING_NOTES_DEVICE", "cuda:0")

    result = runner.invoke(cli.app, [str(_audio(tmp_path))])

    assert result.exit_code == 0, result.output
    ((settings, _prompt),) = calls
    assert settings.device == "cuda:0"


def test_unknown_engine_is_rejected_before_transcription(
    tmp_path: Path, calls: list[tuple[Any, Any]]
) -> None:
    from meeting_notes import cli

    audio = _audio(tmp_path)

    result = runner.invoke(cli.app, [str(audio), "--engine", "foo"])

    assert result.exit_code == 1, result.output
    assert "foo" in result.output
    assert "gigaam" in result.output
    assert "whisper" in result.output
    assert "Traceback" not in result.output
    assert calls == []
    assert not audio.with_suffix(".md").exists()


def test_ctranslate2_device_string_is_rejected_under_default_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[Any, Any]]
) -> None:
    from meeting_notes import cli

    monkeypatch.setenv("MEETING_NOTES_DEVICE", "cuda:0")

    result = runner.invoke(cli.app, [str(_audio(tmp_path))])

    assert result.exit_code == 1, result.output
    flat = result.output.replace("\n", "")
    assert "MEETING_NOTES_DEVICE" in flat
    assert "--engine whisper" in flat
    assert calls == []


def test_record_rejects_unknown_engine_before_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes import cli

    recorders = _patch_record(monkeypatch, 22631)

    result = runner.invoke(
        cli.record_app, [str(tmp_path / "standup"), "--app", "chrome", "--engine", "foo"]
    )

    assert result.exit_code == 1, result.output
    assert recorders == []
    assert list(tmp_path.glob("*.wav")) == []
    assert list(tmp_path.glob("*.md")) == []


def test_record_reports_settings_error_before_windows_build_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes import cli

    recorders = _patch_record(monkeypatch, 19045)

    result = runner.invoke(
        cli.record_app, [str(tmp_path / "standup"), "--app", "chrome", "--engine", "foo"]
    )

    assert result.exit_code == 1, result.output
    assert "foo" in result.output
    assert "Windows 11" not in result.output
    assert recorders == []
