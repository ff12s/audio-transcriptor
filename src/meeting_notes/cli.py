import datetime
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TimeElapsedColumn

from meeting_notes.config import Settings
from meeting_notes.errors import MeetingNotesError
from meeting_notes.glossary import build_prompt, is_prompt_too_long, load_glossary
from meeting_notes.pipeline import MetaOverrides, TranscriptionPipeline
from meeting_notes.renderer import TranscriptRenderer
from meeting_notes.transcriber import Transcriber, WhisperTranscriber

app = typer.Typer(add_completion=False)
err = Console(stderr=True)
out = Console()


def build_transcriber(
    settings: Settings,
    initial_prompt: str | None,
    on_progress: Callable[[float, float], None],
) -> AbstractContextManager[Transcriber]:
    # Единая точка подмены Whisper в тестах.
    return WhisperTranscriber(settings, initial_prompt, on_progress)


def _split_participants(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


@app.command()
def main(
    file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Аудио- или видеофайл."),
    ],
    title: Annotated[str | None, typer.Option(help="Заголовок (по умолчанию имя файла).")] = None,
    date: Annotated[
        datetime.datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Дата встречи, YYYY-MM-DD."),
    ] = None,
    participants: Annotated[str | None, typer.Option(help="Участники через запятую.")] = None,
    model: Annotated[
        str | None, typer.Option(help="Модель Whisper (по умолчанию из конфига).")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Перезаписать существующий .md.")] = False,
) -> None:
    """Транскрибировать запись встречи в Markdown рядом с исходным файлом."""
    try:
        settings = Settings()
        if model is not None:
            settings.model = model

        terms = load_glossary(settings.glossary_path)
        if not terms:
            missing = escape(str(settings.glossary_path))
            err.print(f"[yellow]Глоссарий пуст или не найден: {missing}[/yellow]")
        prompt = build_prompt(terms)
        if prompt is not None and is_prompt_too_long(prompt):
            err.print(
                "[yellow]Глоссарий длинный: Whisper использует только ~224 токена промпта, "
                "остальное будет отброшено.[/yellow]"
            )

        overrides = MetaOverrides(
            title=title,
            date=date.date() if date else None,
            participants=_split_participants(participants),
        )
        columns = (
            *Progress.get_default_columns()[:1],
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with Progress(*columns, console=out) as progress:
            task = progress.add_task("Транскрипция", total=None)

            def on_progress(done: float, total: float) -> None:
                progress.update(task, total=total, completed=min(done, total))

            with build_transcriber(settings, prompt, on_progress) as transcriber:
                pipeline = TranscriptionPipeline(transcriber, TranscriptRenderer())
                result = pipeline.run(file, overrides, overwrite)
            # Последний сегмент VAD обычно заканчивается раньше конца файла.
            final = progress.tasks[task].total or 1
            progress.update(task, total=final, completed=final)
    except MeetingNotesError as exc:
        err.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None
    out.print(f"Готово: {escape(str(result))}", soft_wrap=True)
