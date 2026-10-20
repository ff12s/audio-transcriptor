import datetime
import types
from pathlib import Path

import pytest

from meeting_notes.errors import MeetingNotesError, OutputExistsError
from meeting_notes.pipeline import _FALLBACK_ATTEMPTS, MetaOverrides, TranscriptionPipeline
from meeting_notes.renderer import TranscriptRenderer
from tests.conftest import FakeTranscriber


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"x")
    return path


def _run(audio: Path, fake: FakeTranscriber, writer_exc: Exception) -> None:
    def failing_writer(path: Path, text: str, overwrite: bool) -> None:
        raise writer_exc

    pipeline = TranscriptionPipeline(fake, TranscriptRenderer(), failing_writer)
    pipeline.run(audio, MetaOverrides(), overwrite=False)


@pytest.mark.parametrize("exc", [OSError("disk full"), OutputExistsError("race")])
def test_writer_failure_saves_fallback(
    audio: Path, fake_transcriber: FakeTranscriber, exc: Exception
) -> None:
    with pytest.raises(MeetingNotesError) as info:
        _run(audio, fake_transcriber, exc)
    (fallback,) = audio.parent.glob("meeting.*.md")
    assert "[00:00] Привет, начинаем." in fallback.read_text(encoding="utf-8")
    assert str(exc) in str(info.value)
    assert str(fallback) in str(info.value)
    assert not audio.with_suffix(".md").exists()


def test_fallback_failure_reports_original_error(
    audio: Path, fake_transcriber: FakeTranscriber, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_fallback(path: Path, text: str, overwrite: bool) -> None:
        raise OSError("still broken")

    monkeypatch.setattr("meeting_notes.pipeline.write_output", broken_fallback)
    with pytest.raises(MeetingNotesError) as info:
        _run(audio, fake_transcriber, OSError("disk full"))
    message = str(info.value)
    assert message.index("disk full") < message.index("still broken")
    assert "тоже не записан" in message
    assert "не сохранён" in message
    assert list(audio.parent.glob("meeting.*.md")) == []


class _FrozenDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> "_FrozenDatetime":
        return cls(2026, 1, 1, 12, 0, 0)


def test_fallback_name_collision_picks_free_name(
    audio: Path, fake_transcriber: FakeTranscriber, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "meeting_notes.pipeline.datetime", types.SimpleNamespace(datetime=_FrozenDatetime)
    )
    taken = [audio.with_name("meeting.20260101-120000.md")]
    taken.append(audio.with_name("meeting.20260101-120000-1.md"))
    for path in taken:
        path.write_text("занято", encoding="utf-8")
    with pytest.raises(MeetingNotesError) as info:
        _run(audio, fake_transcriber, OSError("disk full"))
    free = audio.with_name("meeting.20260101-120000-2.md")
    assert str(free) in str(info.value)
    assert "Привет, начинаем." in free.read_text(encoding="utf-8")
    assert [p.read_text(encoding="utf-8") for p in taken] == ["занято", "занято"]


def test_fallback_gives_up_after_bounded_attempts(
    audio: Path, fake_transcriber: FakeTranscriber, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[Path] = []

    def always_taken(path: Path, text: str, overwrite: bool) -> None:
        attempts.append(path)
        raise OutputExistsError("taken")

    monkeypatch.setattr("meeting_notes.pipeline.write_output", always_taken)
    with pytest.raises(MeetingNotesError, match="не сохранён"):
        _run(audio, fake_transcriber, OSError("disk full"))
    assert len(attempts) == _FALLBACK_ATTEMPTS
    assert len(set(attempts)) == _FALLBACK_ATTEMPTS
