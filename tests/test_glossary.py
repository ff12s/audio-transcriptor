from pathlib import Path

from meeting_notes.glossary import (
    PROMPT_WARN_CHARS,
    build_prompt,
    is_prompt_too_long,
    load_glossary,
)


def test_load_skips_comments_and_blanks(tmp_path: Path) -> None:
    f = tmp_path / "g.txt"
    f.write_text("# comment\n\n  Kubernetes  \n#another\nJira\n", encoding="utf-8")
    assert load_glossary(f) == ["Kubernetes", "Jira"]


def test_load_missing_file(tmp_path: Path) -> None:
    assert load_glossary(tmp_path / "nope.txt") == []


def test_build_prompt() -> None:
    assert build_prompt(["a", "b", "c"]) == "Глоссарий: a, b, c."
    assert build_prompt([]) is None


def test_long_prompt_detected() -> None:
    assert not is_prompt_too_long("x" * PROMPT_WARN_CHARS)
    assert is_prompt_too_long("x" * (PROMPT_WARN_CHARS + 1))
