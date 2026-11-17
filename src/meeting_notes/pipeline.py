import datetime
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from meeting_notes.asr import Transcriber
from meeting_notes.errors import MeetingNotesError, OutputExistsError
from meeting_notes.models import SPEAKER_ME, SPEAKER_THEM, Segment, Transcript
from meeting_notes.render import TranscriptRenderer, interleave, output_path_for, write_output

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
        _write_or_fallback(self._writer, out, text, overwrite)
        return out


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


def _write_or_fallback(
    writer: Callable[[Path, str, bool], None], out: Path, text: str, overwrite: bool
) -> None:
    try:
        writer(out, text, overwrite)
    except (OutputExistsError, OSError) as exc:
        _save_fallback(out, text, exc)


def call_paths(base: Path) -> tuple[Path, Path, Path]:
    # with_suffix съел бы часть имени базы с точкой («standup.v2»).
    name = base.name
    return (
        base.with_name(f"{name}.me.wav"),
        base.with_name(f"{name}.them.wav"),
        base.with_name(f"{name}.md"),
    )


def check_call_outputs(base: Path, overwrite: bool, recording: bool) -> None:
    me_wav, them_wav, md = call_paths(base)
    if md.exists() and not overwrite:
        raise OutputExistsError(f"Файл уже существует: {md}. Укажите --overwrite для перезаписи.")
    # Дорожки - единственная копия записи: --overwrite их не трогает.
    if recording and (me_wav.exists() or them_wav.exists()):
        raise OutputExistsError(
            f"Дорожки уже есть ({me_wav}, {them_wav}), используйте --transcribe-only "
            "или другое имя."
        )


class CallTranscriptionPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        renderer: TranscriptRenderer,
        writer: Callable[[Path, str, bool], None] = write_output,
    ) -> None:
        self._transcriber = transcriber
        self._renderer = renderer
        self._writer = writer

    def run(self, base: Path, overrides: MetaOverrides, overwrite: bool) -> Path:
        me_wav, them_wav, md = call_paths(base)
        check_call_outputs(base, overwrite, recording=False)
        me = self._transcriber.transcribe(me_wav)
        them = self._transcriber.transcribe(them_wav)
        meta = me.meta.model_copy(
            update={
                "title": base.name,
                "source": base.name,
                "duration": max(me.meta.duration, them.meta.duration),
                **overrides.as_update(),
            }
        )
        transcript = Transcript(
            meta=meta,
            segments=interleave(_labeled(me, SPEAKER_ME), _labeled(them, SPEAKER_THEM)),
        )
        _write_or_fallback(self._writer, md, self._renderer.render(transcript), overwrite)
        return md


def _labeled(transcript: Transcript, speaker: str) -> list[Segment]:
    return [seg.model_copy(update={"speaker": speaker}) for seg in transcript.segments]
