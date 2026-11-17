from meeting_notes.models import Segment


def merge_segments(
    segments: list[Segment], min_s: float = 20.0, max_s: float = 30.0
) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if merged:
            cur = merged[-1]
            if (
                cur.speaker == seg.speaker
                and cur.end - cur.start < min_s
                and seg.end - cur.start <= max_s
            ):
                merged[-1] = Segment(
                    start=cur.start, end=seg.end, text=f"{cur.text} {text}", speaker=cur.speaker
                )
                continue
        merged.append(Segment(start=seg.start, end=seg.end, text=text, speaker=seg.speaker))
    return merged


def interleave(*tracks: list[Segment]) -> list[Segment]:
    # sorted стабилен: при равном start порядок дорожек сохраняется.
    return sorted((seg for track in tracks for seg in track), key=lambda s: s.start)
