import datetime

import pytest

from meeting_notes.models import Segment, Transcript, TranscriptMeta
from meeting_notes.renderer import TranscriptRenderer, format_timecode


@pytest.mark.parametrize(
    ("seconds", "with_hours", "expected"),
    [
        (0, False, "00:00"),
        (14.9, False, "00:14"),
        (125, False, "02:05"),
        (3599, False, "59:59"),
        (3600, True, "01:00:00"),
        (3735.7, True, "01:02:15"),
        (5, True, "00:00:05"),
    ],
)
def test_format_timecode(seconds: float, with_hours: bool, expected: str) -> None:
    assert format_timecode(seconds, with_hours) == expected


def make(duration: float, segments: list[Segment], participants: list[str]) -> Transcript:
    meta = TranscriptMeta(
        date=datetime.date(2026, 10, 20),
        title='Планёрка "A"',
        participants=participants,
        source="rec.mp3",
        duration=duration,
    )
    return Transcript(meta=meta, segments=segments)


def test_render_frontmatter_and_body() -> None:
    t = make(
        3735.0,
        [Segment(start=0, end=30, text="Привет"), Segment(start=3700, end=3730, text="Пока")],
        ["Анна", "Борис"],
    )
    assert TranscriptRenderer().render(t) == (
        "---\n"
        "date: 2026-10-20\n"
        'title: "Планёрка \\"A\\""\n'
        'participants: ["Анна", "Борис"]\n'
        'source: "rec.mp3"\n'
        'duration: "01:02:15"\n'
        "---\n\n"
        "[00:00:00] Привет\n\n"
        "[01:01:40] Пока\n"
    )


def test_short_recording_uses_mm_ss_and_empty_participants() -> None:
    t = make(90.0, [Segment(start=65, end=70, text="x")], [])
    out = TranscriptRenderer().render(t)
    assert "participants: []" in out
    assert 'duration: "00:01:30"' in out
    assert "\n[01:05] x\n" in out
