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

# Сигнатуры CUDA-рантайма и загрузки его DLL; голое «cuda» не годится: оно встречается
# в путях и именах моделей.
_CUDA_SIGNATURES = re.compile(
    r"cuda (failed|error|driver)|cuda_error|no cuda[- ]capable device"
    r"|cudnn\w*\.dll|cudnn\w*64_\d+|cublas\w*\.dll|cublas\w*64_\d+"
    r"|cudnn_status|cublas_status|cuda runtime|could not (load|locate).*(cudnn|cublas)",
    re.IGNORECASE,
)
_DEVICE_MARKERS = ("compute type", "compute_type", "device")
# Ровно один '/', без диска и обратных слэшей, имя владельца не начинается с точки: иначе
# опечатка в локальном пути ушла бы в Hugging Face и дала бы невнятную ошибку загрузки.
_HF_REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")
# Короче этого значение (например, MEETING_NOTES_MODEL_DIR=cuda) стёрло бы из сообщения
# настоящие маркеры ошибки.
_MIN_STRIPPED_LEN = 6


def _without_user_text(message: str, settings: Settings) -> str:
    # Библиотеки подставляют в сообщения путь к моделям и id репозитория, а huggingface_hub
    # кладёт модель в каталог «models--org--name»; в них может встретиться «device» или
    # «cuda», не связанное с причиной сбоя.
    model_dir = str(settings.model_dir)
    spellings = {model_dir, model_dir.replace("\\", "/"), model_dir.replace("/", "\\")}
    spellings.add(settings.model)
    spellings.add("models--" + settings.model.replace("/", "--"))
    for text in sorted(spellings, key=len, reverse=True):
        if len(text) >= _MIN_STRIPPED_LEN:
            message = re.sub(re.escape(text), " ", message, flags=re.IGNORECASE)
    return message


@contextmanager
def _decode_errors(audio: Path) -> Iterator[None]:
    try:
        yield
    except (ValueError, OSError) as exc:
        # Ошибки PyAV наследуют ValueError (InvalidDataError и др.) или OSError.
        raise MeetingNotesError(
            f"Не удалось прочитать или декодировать аудиофайл {audio.name!r}: {exc}."
        ) from exc


def _cuda_failure_error(exc: Exception, settings: Settings) -> MeetingNotesError | None:
    # ctranslate2 сообщает об ошибках CUDA-рантайма как RuntimeError.
    if not (settings.device.startswith("cuda") and isinstance(exc, RuntimeError)):
        return None
    message = _without_user_text(str(exc), settings).lower()
    if "out of memory" in message:
        return CudaOutOfMemoryError(
            f"Не хватило видеопамяти GPU: {exc}. Закройте другие приложения, "
            "использующие GPU, задайте MEETING_NOTES_COMPUTE_TYPE=int8_float16 "
            "или выберите модель поменьше (--model)."
        )
    if _CUDA_SIGNATURES.search(message):
        return CudaUnavailableError(
            f"Ошибка CUDA: {exc}. Проверьте драйвер NVIDIA и пакеты "
            "nvidia-cublas-cu12 / nvidia-cudnn-cu12 (`uv sync`)."
        )
    return None


@contextmanager
def _cuda_errors(settings: Settings) -> Iterator[None]:
    try:
        yield
    except RuntimeError as exc:
        error = _cuda_failure_error(exc, settings)
        if error is None:
            raise
        raise error from exc


def _model_load_error(exc: Exception, settings: Settings) -> MeetingNotesError:
    # Порядок проверок: ValueError про device/compute_type (по тексту без пути и имени модели),
    # затем CUDA/OOM, только для device cuda* и RuntimeError, иначе ошибка загрузки.
    # Исключения загрузки не ограничены OSError (huggingface_hub 2.2.0 ходит через httpx2).
    name = settings.model
    if isinstance(exc, ValueError) and any(
        marker in _without_user_text(str(exc), settings).lower() for marker in _DEVICE_MARKERS
    ):
        return MeetingNotesError(
            f"Модель {name!r} не удалось запустить с device={settings.device!r}, "
            f"compute_type={settings.compute_type!r}: {exc}. Проверьте "
            "MEETING_NOTES_DEVICE и MEETING_NOTES_COMPUTE_TYPE "
            "(например, для cpu нужен int8)."
        )
    if (cuda_error := _cuda_failure_error(exc, settings)) is not None:
        return cuda_error
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
        with _cuda_errors(self._settings):
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
            with _cuda_errors(self._settings), _decode_errors(audio):
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
