import datetime
import gc
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from meeting_notes import cuda_setup
from meeting_notes.config import Settings
from meeting_notes.errors import CudaOutOfMemoryError, CudaUnavailableError, MeetingNotesError
from meeting_notes.models import Segment, Transcript, TranscriptMeta

_CUDA_MARKERS = ("cuda", "cublas", "cudnn")
_DEVICE_MARKERS = ("compute type", "compute_type", "device")
# Ровно один '/', без диска, ведущей точки и обратных слэшей: иначе опечатка в локальном пути
# ушла бы в Hugging Face и дала бы невнятную ошибку загрузки.
_HF_REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")


def _is_cuda_failure(message: str) -> bool:
    return "out of memory" in message or any(marker in message for marker in _CUDA_MARKERS)


@contextmanager
def _decode_errors(audio: Path) -> Iterator[None]:
    try:
        yield
    except (ValueError, OSError) as exc:
        # Ошибки PyAV наследуют ValueError (InvalidDataError и др.) или OSError.
        raise MeetingNotesError(
            f"Не удалось прочитать или декодировать аудиофайл {audio.name!r}: {exc}."
        ) from exc


def _cuda_failure_error(exc: Exception) -> MeetingNotesError | None:
    message = str(exc).lower()
    if "out of memory" in message:
        return CudaOutOfMemoryError(
            f"Не хватило видеопамяти GPU: {exc}. Закройте другие приложения, "
            "использующие GPU, задайте MEETING_NOTES_COMPUTE_TYPE=int8_float16 "
            "или выберите модель поменьше (--model)."
        )
    if _is_cuda_failure(message):
        return CudaUnavailableError(
            f"Ошибка CUDA: {exc}. Проверьте драйвер NVIDIA и пакеты "
            "nvidia-cublas-cu12 / nvidia-cudnn-cu12 (`uv sync`)."
        )
    return None


@contextmanager
def _cuda_errors() -> Iterator[None]:
    try:
        yield
    except RuntimeError as exc:
        error = _cuda_failure_error(exc)
        if error is None:
            raise
        raise error from exc


def _model_load_error(exc: Exception, settings: Settings) -> MeetingNotesError:
    # Классификация по тексту, а не по классу: сетевой слой huggingface_hub (httpx2) бросает
    # исключения вне OSError, а HFValidationError наследует ValueError, как и ошибки ctranslate2.
    if (cuda_error := _cuda_failure_error(exc)) is not None:
        return cuda_error
    name = settings.model
    if isinstance(exc, ValueError) and any(
        marker in str(exc).lower() for marker in _DEVICE_MARKERS
    ):
        return MeetingNotesError(
            f"Модель {name!r} не удалось запустить с device={settings.device!r}, "
            f"compute_type={settings.compute_type!r}: {exc}. Проверьте "
            "MEETING_NOTES_DEVICE и MEETING_NOTES_COMPUTE_TYPE "
            "(например, для cpu нужен int8)."
        )
    return MeetingNotesError(
        f"Не удалось загрузить модель {name!r} в {settings.model_dir}: "
        f"{type(exc).__name__}: {exc}. Проверьте свободное место на диске, сеть, "
        "имя модели и MEETING_NOTES_MODEL_DIR."
    )


class Transcriber(Protocol):
    def transcribe(self, audio: Path) -> Transcript: ...


class WhisperTranscriber:
    def __init__(
        self,
        settings: Settings,
        initial_prompt: str | None,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> None:
        self._settings = settings
        self._initial_prompt = initial_prompt
        self._on_progress = on_progress
        self._model: Any = None

    def __enter__(self) -> "WhisperTranscriber":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._model = None
        gc.collect()

    def _load_model(self) -> Any:
        if self._model is None:
            if self._settings.device.startswith("cuda"):
                cuda_setup.prepare_cuda()
            # DLL подключаются только в cuda-ветке; импорт идёт после этого, чтобы они были видны.
            from faster_whisper import WhisperModel, available_models

            name = self._settings.model
            known = available_models()
            is_local_dir = Path(name).is_dir()
            if name not in known and not _HF_REPO_ID.fullmatch(name) and not is_local_dir:
                raise MeetingNotesError(
                    f"Неизвестная модель {name!r}. Доступные: {', '.join(known)}. Можно указать "
                    "id репозитория Hugging Face вида org/name или существующий каталог "
                    "локальной модели (несуществующий путь не принимается). "
                    "Проверьте --model / MEETING_NOTES_MODEL."
                )

            model_dir = self._settings.model_dir
            if not is_local_dir:
                try:
                    model_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise MeetingNotesError(
                        f"Не удалось создать каталог моделей {model_dir}: {exc}. "
                        "Проверьте MEETING_NOTES_MODEL_DIR."
                    ) from exc

            try:
                self._model = WhisperModel(
                    name,
                    device=self._settings.device,
                    compute_type=self._settings.compute_type,
                    download_root=str(model_dir),
                )
            except Exception as exc:
                raise _model_load_error(exc, self._settings) from exc
        return self._model

    def _check_language(self, model: Any) -> None:
        language = self._settings.language
        if language not in model.supported_languages:
            raise MeetingNotesError(
                f"Язык {language!r} не поддерживается моделью {self._settings.model}. "
                "Проверьте MEETING_NOTES_LANGUAGE (например ru)."
            )

    def transcribe(self, audio: Path) -> Transcript:
        with _cuda_errors():
            model = self._load_model()
            self._check_language(model)
            with _decode_errors(audio):
                raw_segments, info = model.transcribe(
                    str(audio),
                    language=self._settings.language,
                    vad_filter=True,
                    initial_prompt=self._initial_prompt,
                )
        segments: list[Segment] = []
        # raw_segments — ленивый генератор: распознавание идёт в next(). Колбэк вызывается
        # вне защищённых блоков, иначе его сбой выглядел бы как ошибка декодирования.
        segment_iter = iter(raw_segments)
        while True:
            with _cuda_errors(), _decode_errors(audio):
                raw = next(segment_iter, None)
            if raw is None:
                break
            segments.append(Segment(start=raw.start, end=raw.end, text=raw.text.strip()))
            if self._on_progress is not None:
                self._on_progress(raw.end, info.duration)

        meta = TranscriptMeta(
            date=datetime.date.fromtimestamp(audio.stat().st_mtime),
            title=audio.stem,
            participants=[],
            source=audio.name,
            duration=info.duration,
        )
        return Transcript(meta=meta, segments=segments)
