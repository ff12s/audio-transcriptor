from pathlib import Path

# Whisper учитывает ~224 токена промпта и молча отбрасывает остальное;
# для русского это грубо 600 символов.
PROMPT_WARN_CHARS = 600


def load_glossary(path: Path) -> list[str]:
    if not path.is_file():
        return []
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            terms.append(term)
    return terms


def build_prompt(terms: list[str]) -> str | None:
    if not terms:
        return None
    return f"Глоссарий: {', '.join(terms)}."


def is_prompt_too_long(prompt: str) -> bool:
    return len(prompt) > PROMPT_WARN_CHARS
