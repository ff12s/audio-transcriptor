import datetime
from pathlib import Path
from types import TracebackType

import pytest

from meeting_notes.models import Segment, Transcript, TranscriptMeta


class FakeTranscriber:
    def __init__(self, segments: list[Segment], duration: float) -> None:
        self.segments = segments
        self.duration = duration
        self.calls: list[Path] = []

    def __enter__(self) -> "FakeTranscriber":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def transcribe(self, audio: Path) -> Transcript:
        self.calls.append(audio)
        meta = TranscriptMeta(
            date=datetime.date(2020, 1, 1),
            title=audio.stem,
            participants=[],
            source=audio.name,
            duration=self.duration,
        )
        return Transcript(meta=meta, segments=self.segments)


@pytest.fixture
def fake_transcriber() -> FakeTranscriber:
    return FakeTranscriber(
        [
            Segment(start=0.0, end=14.0, text=" Привет, начинаем."),
            Segment(start=14.0, end=25.0, text=" Первый вопрос."),
        ],
        duration=40.0,
    )
