import datetime

from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptMeta(BaseModel):
    date: datetime.date
    title: str
    participants: list[str]
    source: str
    duration: float  # секунды


class Transcript(BaseModel):
    meta: TranscriptMeta
    segments: list[Segment]
