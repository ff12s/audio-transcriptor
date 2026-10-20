import datetime
import types
from pathlib import Path

import pytest

from meeting_notes.errors import MeetingNotesError, OutputExistsError
from meeting_notes.models import Segment, Transcript, TranscriptMeta
from meeting_notes.pipeline import (
    _FALLBACK_ATTEMPTS,
    CallTranscriptionPipeline,
    MetaOverrides,
    TranscriptionPipeline,
    check_call_outputs,
)
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


class _PerPathTranscriber:
    def __init__(self, by_name: dict[str, list[Segment]], duration: float = 30.0) -> None:
        self.by_name = by_name
        self.duration = duration
        self.calls: list[Path] = []

    def transcribe(self, audio: Path) -> Transcript:
        self.calls.append(audio)
        meta = TranscriptMeta(
            date=datetime.date(2020, 1, 1),
            title=audio.stem,
            participants=[],
            source=audio.name,
            duration=self.duration,
        )
        return Transcript(meta=meta, segments=self.by_name.get(audio.name, []))


def _call_pipeline(transcriber: _PerPathTranscriber) -> CallTranscriptionPipeline:
    return CallTranscriptionPipeline(transcriber, TranscriptRenderer())


def test_call_md_labels_and_order(tmp_path: Path) -> None:
    base = tmp_path / "standup"
    fake = _PerPathTranscriber(
        {
            "standup.me.wav": [Segment(start=5, end=6, text="м")],
            "standup.them.wav": [Segment(start=1, end=2, text="т")],
        }
    )
    md = _call_pipeline(fake).run(base, MetaOverrides(), overwrite=False)
    text = md.read_text(encoding="utf-8")
    assert md == tmp_path / "standup.md"
    assert text.index("[00:01] **Собеседники:** т") < text.index("[00:05] **Я:** м")
    assert 'source: "standup"' in text
    assert 'title: "standup"' in text
    assert fake.calls == [tmp_path / "standup.me.wav", tmp_path / "standup.them.wav"]


def test_call_overrides_and_longest_duration(tmp_path: Path) -> None:
    fake = _PerPathTranscriber({})
    md = _call_pipeline(fake).run(
        tmp_path / "standup", MetaOverrides(title="Стендап"), overwrite=False
    )
    text = md.read_text(encoding="utf-8")
    assert 'title: "Стендап"' in text
    assert 'duration: "00:00:30"' in text


def test_call_base_with_dot_keeps_name(tmp_path: Path) -> None:
    md = _call_pipeline(_PerPathTranscriber({})).run(
        tmp_path / "call.v2", MetaOverrides(), overwrite=False
    )
    assert md == tmp_path / "call.v2.md"


def test_call_one_track_empty(tmp_path: Path) -> None:
    fake = _PerPathTranscriber({"standup.them.wav": [Segment(start=1, end=2, text="т")]})
    md = _call_pipeline(fake).run(tmp_path / "standup", MetaOverrides(), overwrite=False)
    body = md.read_text(encoding="utf-8").split("---\n")[-1]
    assert body.strip() == "[00:01] **Собеседники:** т"


def test_call_both_empty(tmp_path: Path) -> None:
    md = _call_pipeline(_PerPathTranscriber({})).run(
        tmp_path / "standup", MetaOverrides(), overwrite=False
    )
    text = md.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert text.split("---\n")[-1].strip() == ""


def test_call_existing_md_refused_before_transcribing(tmp_path: Path) -> None:
    (tmp_path / "standup.md").write_text("old", encoding="utf-8")
    fake = _PerPathTranscriber({})
    with pytest.raises(OutputExistsError):
        _call_pipeline(fake).run(tmp_path / "standup", MetaOverrides(), overwrite=False)
    assert fake.calls == []


def test_call_writer_failure_saves_fallback(tmp_path: Path) -> None:
    def failing_writer(path: Path, text: str, overwrite: bool) -> None:
        raise OSError("disk full")

    fake = _PerPathTranscriber({"standup.me.wav": [Segment(start=0, end=1, text="м")]})
    pipeline = CallTranscriptionPipeline(fake, TranscriptRenderer(), failing_writer)
    with pytest.raises(MeetingNotesError, match="disk full"):
        pipeline.run(tmp_path / "standup", MetaOverrides(), overwrite=False)
    (fallback,) = tmp_path.glob("standup.*.md")
    assert "**Я:** м" in fallback.read_text(encoding="utf-8")


def test_check_outputs_refuses_existing_wav(tmp_path: Path) -> None:
    (tmp_path / "standup.me.wav").write_bytes(b"x")
    with pytest.raises(OutputExistsError, match="--transcribe-only"):
        check_call_outputs(tmp_path / "standup", overwrite=True, recording=True)


def test_check_outputs_wav_ok_when_not_recording(tmp_path: Path) -> None:
    (tmp_path / "standup.them.wav").write_bytes(b"x")
    check_call_outputs(tmp_path / "standup", overwrite=False, recording=False)


def test_check_outputs_md_overwrite(tmp_path: Path) -> None:
    (tmp_path / "standup.md").write_text("old", encoding="utf-8")
    check_call_outputs(tmp_path / "standup", overwrite=True, recording=True)
    with pytest.raises(OutputExistsError, match="--overwrite"):
        check_call_outputs(tmp_path / "standup", overwrite=False, recording=True)
