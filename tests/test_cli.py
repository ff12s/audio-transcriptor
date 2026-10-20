from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meeting_notes import cli
from meeting_notes.models import Transcript
from tests.conftest import FakeTranscriber

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_transcriber: FakeTranscriber
) -> None:
    monkeypatch.setenv("MEETING_NOTES_GLOSSARY_PATH", str(tmp_path / "glossary.txt"))
    monkeypatch.setattr(cli, "build_transcriber", lambda *_args: fake_transcriber)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"x")
    return path


def test_writes_markdown(audio: Path) -> None:
    result = runner.invoke(cli.app, [str(audio)])
    assert result.exit_code == 0, result.output
    text = audio.with_suffix(".md").read_text(encoding="utf-8")
    assert 'source: "meeting.mp3"' in text
    assert 'title: "meeting"' in text
    assert "[00:00] Привет, начинаем. Первый вопрос." in text


def test_refuses_without_overwrite(audio: Path, fake_transcriber: FakeTranscriber) -> None:
    md = audio.with_suffix(".md")
    md.write_text("old", encoding="utf-8")
    result = runner.invoke(cli.app, [str(audio)])
    assert result.exit_code == 1
    assert "--overwrite" in result.output
    assert "Traceback" not in result.output
    assert md.read_text(encoding="utf-8") == "old"
    assert fake_transcriber.calls == []


def test_overwrite_replaces_file(audio: Path) -> None:
    md = audio.with_suffix(".md")
    md.write_text("old", encoding="utf-8")
    result = runner.invoke(cli.app, [str(audio), "--overwrite"])
    assert result.exit_code == 0, result.output
    assert md.read_text(encoding="utf-8").startswith("---\n")


def test_overrides_applied(audio: Path) -> None:
    result = runner.invoke(
        cli.app,
        [str(audio), "--title", "Ретро", "--date", "2026-10-20", "--participants", "Анна, ,Борис"],
    )
    assert result.exit_code == 0, result.output
    text = audio.with_suffix(".md").read_text(encoding="utf-8")
    assert 'title: "Ретро"' in text
    assert "date: 2026-10-20" in text
    assert 'participants: ["Анна", "Борис"]' in text


def test_missing_file_is_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, [str(tmp_path / "nope.mp3")])
    assert result.exit_code == 2


def test_progress_finishes_at_100_percent(
    audio: Path, fake_transcriber: FakeTranscriber, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build(
        _settings: object, _prompt: object, on_progress: Callable[[float, float], None]
    ) -> FakeTranscriber:
        class Reporting(FakeTranscriber):
            def transcribe(self, audio: Path) -> Transcript:
                on_progress(25.0, self.duration)
                return super().transcribe(audio)

        return Reporting(fake_transcriber.segments, fake_transcriber.duration)

    monkeypatch.setattr(cli, "build_transcriber", build)
    result = runner.invoke(cli.app, [str(audio)])
    assert result.exit_code == 0, result.output
    assert "100%" in result.output


def test_write_failure_saves_fallback_without_traceback(
    audio: Path, fake_transcriber: FakeTranscriber, monkeypatch: pytest.MonkeyPatch
) -> None:
    md = audio.with_suffix(".md")

    class Racing(FakeTranscriber):
        def transcribe(self, audio: Path) -> Transcript:
            md.write_text("appeared meanwhile", encoding="utf-8")
            return super().transcribe(audio)

    racing = Racing(fake_transcriber.segments, fake_transcriber.duration)
    monkeypatch.setattr(cli, "build_transcriber", lambda *_args: racing)
    result = runner.invoke(cli.app, [str(audio)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert md.read_text(encoding="utf-8") == "appeared meanwhile"
    (fallback,) = audio.parent.glob("meeting.*.md")
    assert "[00:00] Привет" in fallback.read_text(encoding="utf-8")
