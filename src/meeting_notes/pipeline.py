import datetime
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from meeting_notes.errors import MeetingNotesError, OutputExistsError
from meeting_notes.output import output_path_for, write_output
from meeting_notes.renderer import TranscriptRenderer
from meeting_notes.transcriber import Transcriber

_FALLBACK_ATTEMPTS = 100


@dataclass(frozen=True)
class MetaOverrides:
    title: str | None = None
    date: datetime.date | None = None
    participants: list[str] | None = None

    def as_update(self) -> dict[str, object]:
        fields = {"title": self.title, "date": self.date, "participants": self.participants}
        return {k: v for k, v in fields.items() if v is not None}


class TranscriptionPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        renderer: TranscriptRenderer,
        writer: Callable[[Path, str, bool], None] = write_output,
    ) -> None:
        self._transcriber = transcriber
        self._renderer = renderer
        self._writer = writer

    def run(self, audio: Path, overrides: MetaOverrides, overwrite: bool) -> Path:
        out = output_path_for(audio)
        if out.exists() and not overwrite:
            # Падаем до загрузки модели: распознавание занимает минуты.
            raise OutputExistsError(
                f"Файл уже существует: {out}. Укажите --overwrite для перезаписи."
            )
        transcript = self._transcriber.transcribe(audio)
        transcript = transcript.model_copy(
            update={"meta": transcript.meta.model_copy(update=overrides.as_update())}
        )
        text = self._renderer.render(transcript)
        try:
            self._writer(out, text, overwrite)
        except (OutputExistsError, OSError) as exc:
            self._save_fallback(out, text, exc)
        return out

    @staticmethod
    def _save_fallback(out: Path, text: str, cause: Exception) -> NoReturn:
        # Транскрипция занимает минуты — результат нельзя терять из-за сбоя записи.
        stem = f"{out.stem}.{datetime.datetime.now():%Y%m%d-%H%M%S}"
        for attempt in range(_FALLBACK_ATTEMPTS):
            suffix = f"-{attempt}" if attempt else ""
            fallback = out.with_name(f"{stem}{suffix}.md")
            try:
                write_output(fallback, text, overwrite=False)
            except OutputExistsError:
                continue
            except OSError as fallback_exc:
                raise MeetingNotesError(
                    f"Не удалось записать {out}: {cause}. Запасной файл {fallback} тоже не "
                    f"записан: {fallback_exc}. Транскрипт не сохранён."
                ) from cause
            raise MeetingNotesError(
                f"Не удалось записать {out}: {cause}. Транскрипт сохранён в {fallback}"
            ) from cause
        raise MeetingNotesError(
            f"Не удалось записать {out}: {cause}. Свободное имя запасного файла "
            f"{out.parent / stem}*.md не найдено. Транскрипт не сохранён."
        ) from cause
