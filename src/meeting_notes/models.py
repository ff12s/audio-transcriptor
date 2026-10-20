import datetime

from pydantic import BaseModel

SPEAKER_ME = "Я"
SPEAKER_THEM = "Собеседники"


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class TranscriptMeta(BaseModel):
    date: datetime.date
    title: str
    participants: list[str]
    source: str
    duration: float  # секунды


class Transcript(BaseModel):
    meta: TranscriptMeta
    segments: list[Segment]
