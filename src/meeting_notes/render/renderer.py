import json

from meeting_notes.models import Transcript
from meeting_notes.render.paragraphs import merge_segments


def format_timecode(seconds: float, with_hours: bool) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if with_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    # Без часов минуты накапливают часы, чтобы не терять информацию.
    return f"{hours * 60 + minutes:02d}:{secs:02d}"


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _label(speaker: str | None) -> str:
    return f"**{speaker}:** " if speaker else ""


class TranscriptRenderer:
    def render(self, t: Transcript) -> str:
        meta = t.meta
        with_hours = meta.duration >= 3600
        participants = ", ".join(_quote(p) for p in meta.participants)
        frontmatter = [
            "---",
            f"date: {meta.date.isoformat()}",
            f"title: {_quote(meta.title)}",
            f"participants: [{participants}]",
            f"source: {_quote(meta.source)}",
            f'duration: "{format_timecode(meta.duration, with_hours=True)}"',
            "---",
        ]
        body = [
            f"[{format_timecode(p.start, with_hours)}] {_label(p.speaker)}{p.text}"
            for p in merge_segments(t.segments)
        ]
        return "\n".join(frontmatter) + "\n\n" + "\n\n".join(body) + "\n"
