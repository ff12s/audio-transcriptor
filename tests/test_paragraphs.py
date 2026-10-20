from meeting_notes.models import Segment
from meeting_notes.paragraphs import merge_segments


def seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


def test_short_segments_merge_until_min_span() -> None:
    segs = [seg(0, 10, "a"), seg(10, 19, "b"), seg(19, 25, "c"), seg(25, 30, "d")]
    result = merge_segments(segs)
    assert result == [seg(0, 25, "a b c"), seg(25, 30, "d")]


def test_does_not_exceed_max_span() -> None:
    segs = [seg(0, 15, "a"), seg(15, 35, "b")]
    assert merge_segments(segs) == segs


def test_long_single_segment_is_own_paragraph() -> None:
    segs = [seg(0, 45, "long"), seg(45, 50, "next")]
    assert merge_segments(segs) == [seg(0, 45, "long"), seg(45, 50, "next")]


def test_drops_empty_and_strips_text() -> None:
    segs = [seg(0, 5, "  a "), seg(5, 8, "   "), seg(8, 12, " b")]
    assert merge_segments(segs) == [seg(0, 12, "a b")]


def test_empty_input() -> None:
    assert merge_segments([]) == []
