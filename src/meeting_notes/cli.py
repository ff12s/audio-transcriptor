import datetime
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TimeElapsedColumn

from meeting_notes.asr import (
    GigaamTranscriber,
    Transcriber,
    WhisperTranscriber,
    build_prompt,
    check_config,
    is_prompt_too_long,
    load_glossary,
)
from meeting_notes.capture import AppSource, CallRecorder, MicSource
from meeting_notes.config import Settings
from meeting_notes.errors import MeetingNotesError
from meeting_notes.pipeline import (
    CallTranscriptionPipeline,
    MetaOverrides,
    TranscriptionPipeline,
    call_paths,
    check_call_outputs,
)
from meeting_notes.processes import check_windows_build, current_windows_build, find_capture_pids
from meeting_notes.renderer import TranscriptRenderer, format_timecode
from meeting_notes.track import TrackWriter

app = typer.Typer(add_completion=False)
record_app = typer.Typer(add_completion=False)
# Без подсветки: Rich выделяет числа в пути и команде повтора ANSI-кодами, и их нельзя
# скопировать; при FORCE_COLOR он включает цвет даже без терминала.
err = Console(stderr=True, highlight=False)
out = Console(highlight=False)


def build_transcriber(
    settings: Settings,
    initial_prompt: str | None,
    on_progress: Callable[[float, float], None],
) -> AbstractContextManager[Transcriber]:
    # Единая точка подмены движка (Whisper или GigaAM) в тестах.
    if settings.engine == "whisper":
        return WhisperTranscriber(settings, initial_prompt, on_progress)
    return GigaamTranscriber(
        settings.gigaam_model, settings.model_dir, settings.device, on_progress
    )


_ENGINES = ("gigaam", "whisper")


def _load_settings(engine: str | None, model: str | None) -> Settings:
    settings = Settings()
    if engine is not None:
        settings.engine = engine
    if settings.engine not in _ENGINES:
        raise MeetingNotesError(
            f"Неизвестный движок {settings.engine!r}: допустимы {' и '.join(_ENGINES)} "
            "(--engine или MEETING_NOTES_ENGINE)."
        )
    if settings.engine == "gigaam":
        if model is not None:
            settings.gigaam_model = model
        check_config(settings.gigaam_model, settings.device, settings.language)
    else:
        if model is not None:
            settings.model = model
        # ctranslate2 не знает "auto" в нашем смысле: для Whisper это cuda.
        if settings.device == "auto":
            settings.device = "cuda"
    return settings


def windows_build() -> int:
    # Точка подмены в тестах.
    return current_windows_build()


def _open_track(path: Path) -> TrackWriter:
    try:
        return TrackWriter(path)
    except OSError as exc:
        # Нет прав на каталог или недопустимое имя: показываем сообщением, а не стектрейсом.
        raise MeetingNotesError(f"Не удалось создать файл {path}: {exc}") from exc


def build_recorder(
    me_path: Path, them_path: Path, pids: list[int], on_warning: Callable[[str], None]
) -> CallRecorder:
    # Точка подмены в тестах. Источники создаются раньше файлов: без микрофона запись не
    # начинается и пустых дорожек после себя не оставляет.
    mic = MicSource()
    apps = [AppSource(pid) for pid in pids]
    me = _open_track(me_path)
    try:
        them = _open_track(them_path)
    except BaseException:
        me.close()
        me_path.unlink(missing_ok=True)
        raise
    return CallRecorder(me, them, mic, list(apps), on_warning=on_warning)


def _split_participants(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def _transcribe(settings: Settings, run: Callable[[Transcriber], Path]) -> Path:
    prompt = None
    if settings.engine == "whisper":
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
    name = settings.gigaam_model if settings.engine == "gigaam" else settings.model
    out.print(f"Движок: {escape(settings.engine)}, модель: {escape(name)}")

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
            result = run(transcriber)
        # Последний сегмент VAD обычно заканчивается раньше конца файла.
        final = progress.tasks[task].total or 1
        progress.update(task, total=final, completed=final)
    return result


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
    engine: Annotated[
        str | None, typer.Option(help="Движок: gigaam или whisper (по умолчанию из конфига).")
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Модель выбранного движка (по умолчанию из конфига).")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Перезаписать существующий .md.")] = False,
) -> None:
    """Транскрибировать запись встречи в Markdown рядом с исходным файлом."""
    overrides = MetaOverrides(
        title=title,
        date=date.date() if date else None,
        participants=_split_participants(participants),
    )
    try:
        settings = _load_settings(engine, model)
        result = _transcribe(
            settings,
            lambda transcriber: TranscriptionPipeline(transcriber, TranscriptRenderer()).run(
                file, overrides, overwrite
            ),
        )
    except MeetingNotesError as exc:
        err.print(f"[red]{escape(str(exc))}[/red]", soft_wrap=True)
        raise typer.Exit(code=1) from None
    out.print(f"Готово: {escape(str(result))}", soft_wrap=True)


def _record(me_wav: Path, them_wav: Path, pids: list[int]) -> None:
    def on_warning(message: str) -> None:
        err.print(f"[yellow]{escape(message)}[/yellow]", soft_wrap=True)

    recorder = build_recorder(me_wav, them_wav, pids, on_warning)
    out.print("Запись... Ctrl+C - стоп.")
    shown = -1

    def on_tick(elapsed: float) -> None:
        nonlocal shown
        if int(elapsed) != shown:
            shown = int(elapsed)
            out.print(f"Запись {format_timecode(elapsed, with_hours=True)}", end="\r")

    partial = f"Частично записано: {me_wav}, {them_wav}."
    try:
        duration = recorder.record_until_interrupted(on_tick)
    except MeetingNotesError as exc:
        raise MeetingNotesError(f"{exc}\n{partial}") from exc
    except (Exception, KeyboardInterrupt):
        # Чужая ошибка идёт со стектрейсом, но путь к записи пользователь должен увидеть раньше.
        err.print(escape(partial), soft_wrap=True)
        raise
    out.print(f"\nЗапись остановлена: {format_timecode(duration, with_hours=True)}.")


@record_app.command()
def record(
    base: Annotated[
        Path,
        typer.Argument(help="Путь без расширения: рядом появятся .me.wav, .them.wav и .md."),
    ],
    app_name: Annotated[
        str | None,
        typer.Option("--app", help="Имя процесса приложения (chrome, zoom); .exe не обязателен."),
    ] = None,
    title: Annotated[str | None, typer.Option(help="Заголовок (по умолчанию имя базы).")] = None,
    date: Annotated[
        datetime.datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Дата встречи, YYYY-MM-DD."),
    ] = None,
    participants: Annotated[str | None, typer.Option(help="Участники через запятую.")] = None,
    engine: Annotated[
        str | None, typer.Option(help="Движок: gigaam или whisper (по умолчанию из конфига).")
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Модель выбранного движка (по умолчанию из конфига).")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Перезаписать существующий .md.")] = False,
    transcribe_only: Annotated[
        bool,
        typer.Option(
            "--transcribe-only", help="Не записывать, а транскрибировать готовые дорожки."
        ),
    ] = False,
) -> None:
    """Записать звук приложения и микрофон в две дорожки и транскрибировать их."""
    me_wav, them_wav, _md = call_paths(base)
    overrides = MetaOverrides(
        title=title,
        date=date.date() if date else None,
        participants=_split_participants(participants),
    )
    try:
        # Ошибка настроек не должна всплывать только после часа записи.
        settings = _load_settings(engine, model)
        if not transcribe_only:
            check_windows_build(windows_build())
        check_call_outputs(base, overwrite, recording=not transcribe_only)
        if transcribe_only:
            missing = [str(p) for p in (me_wav, them_wav) if not p.exists()]
            if missing:
                raise MeetingNotesError(f"Нет дорожек для транскрипции: {', '.join(missing)}.")
        else:
            if app_name is None:
                raise MeetingNotesError("Укажите приложение для записи: --app chrome.")
            if not base.parent.is_dir():
                raise MeetingNotesError(f"Каталог не существует: {base.parent}.")
            _record(me_wav, them_wav, find_capture_pids(app_name))
        # Запись - единственная копия звонка: после сбоя говорим, где она и как продолжить.
        hint = (
            f"Дорожки сохранены: {me_wav}, {them_wav}. "
            f'Повторить: uv run record --transcribe-only "{base}"'
        )
        try:
            result = _transcribe(
                settings,
                lambda transcriber: CallTranscriptionPipeline(
                    transcriber, TranscriptRenderer()
                ).run(base, overrides, overwrite),
            )
        except MeetingNotesError as exc:
            raise MeetingNotesError(f"{exc}\n{hint}") from exc
        except (Exception, KeyboardInterrupt):
            # Ctrl+C или чужая ошибка (сбой движка): стектрейс остаётся, подсказка добавляется.
            err.print(escape(hint), soft_wrap=True)
            raise
    except MeetingNotesError as exc:
        err.print(f"[red]{escape(str(exc))}[/red]", soft_wrap=True)
        raise typer.Exit(code=1) from None
    out.print(f"Готово: {escape(str(result))}", soft_wrap=True)
