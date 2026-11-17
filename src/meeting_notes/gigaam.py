import datetime
import gc
import re
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

from meeting_notes import cuda_setup
from meeting_notes.errors import CudaUnavailableError, MeetingNotesError
from meeting_notes.models import Segment, Transcript, TranscriptMeta
from meeting_notes.transcriber import _decode_errors

_CPU = "CPUExecutionProvider"
_ACCELERATORS = {
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}
DEVICES: tuple[str, ...] = ("auto", "cpu", *_ACCELERATORS)
_MODEL_NAME = re.compile(r"gigaam-[\w.-]+")
_SAMPLE_RATE = 16000


def check_config(model: str, device: str, language: str) -> None:
    if not _MODEL_NAME.fullmatch(model):
        raise MeetingNotesError(
            f"Модель {model!r} не подходит для движка gigaam: имя должно быть вида "
            "gigaam-v3-e2e-rnnt. Для моделей Whisper используйте --engine whisper "
            "(MEETING_NOTES_ENGINE=whisper) либо задайте модель GigaAM через "
            "--model / MEETING_NOTES_GIGAAM_MODEL."
        )
    if device not in DEVICES:
        raise MeetingNotesError(
            f"Устройство {device!r} не поддерживается движком gigaam. Допустимые значения: "
            f"{', '.join(DEVICES)} (MEETING_NOTES_DEVICE). Номер GPU вида cuda:0 не поддерживает "
            "ни один движок; для Whisper выберите --engine whisper (MEETING_NOTES_ENGINE=whisper) "
            "с cuda, cpu или auto."
        )
    if language != "ru":
        raise MeetingNotesError(
            f"Язык {language!r} не поддерживается движком gigaam: он распознаёт только русский. "
            "Проверьте MEETING_NOTES_LANGUAGE или используйте --engine whisper."
        )


def _decode(audio: Path) -> Any:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(audio), sampling_rate=_SAMPLE_RATE)


class GigaamTranscriber:
    def __init__(
        self,
        model: str,
        model_dir: Path,
        device: str,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> None:
        self._model_name = model
        self._model_dir = model_dir
        self._device = device
        self._on_progress = on_progress
        self._model: Any = None

    def __enter__(self) -> "GigaamTranscriber":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._model = None
        gc.collect()

    def _providers(self) -> list[str]:
        import onnxruntime

        available = onnxruntime.get_available_providers()
        if self._device == "cpu":
            return [_CPU]
        if self._device == "auto":
            if _ACCELERATORS["cuda"] not in available:
                return [_CPU]
            try:
                cuda_setup.prepare_cuda()
            except CudaUnavailableError:
                return [_CPU]
            onnxruntime.preload_dlls()
            return [_ACCELERATORS["cuda"], _CPU]
        provider = _ACCELERATORS[self._device]
        if provider not in available:
            raise MeetingNotesError(
                f"Устройство {self._device!r} недоступно: в onnxruntime нет {provider} "
                f"(доступно: {', '.join(available)}). Установите сборку onnxruntime с этим "
                "провайдером или задайте MEETING_NOTES_DEVICE=auto либо cpu."
            )
        if self._device == "cuda":
            cuda_setup.prepare_cuda()
            onnxruntime.preload_dlls()
        return [provider, _CPU]

    def _load_model(self) -> Any:
        if self._model is None:
            vad_dir = self._model_dir / "silero-vad"
            model_path = self._model_dir / self._model_name
            try:
                # Листовые каталоги не создаём: onnx-asr считает существующий каталог
                # готовой моделью и не скачивает в него файлы.
                self._model_dir.mkdir(parents=True, exist_ok=True)
                providers = self._providers()
                import onnx_asr

                vad = onnx_asr.load_vad("silero", vad_dir, providers=[_CPU])
                asr = onnx_asr.load_model(self._model_name, model_path, providers=providers)
                self._model = asr.with_vad(vad)
            except MeetingNotesError:
                raise
            except Exception as exc:
                raise MeetingNotesError(
                    f"Не удалось загрузить модель {self._model_name!r} в {self._model_dir}: "
                    f"{type(exc).__name__}: {exc}. Проверьте свободное место на диске, сеть, "
                    "имя модели и MEETING_NOTES_MODEL_DIR. Если загрузка прервалась, удалите "
                    f"каталоги {model_path} и {vad_dir}. При сбое GPU задайте "
                    "MEETING_NOTES_DEVICE=cpu."
                ) from exc
        return self._model

    def transcribe(self, audio: Path) -> Transcript:
        # Модель грузится до импорта faster_whisper: CUDA DLL должны быть подключены раньше.
        model = self._load_model()
        with _decode_errors(audio):
            waveform = _decode(audio)
        duration = len(waveform) / _SAMPLE_RATE
        segments: list[Segment] = []
        if len(waveform) > 0:
            for result in model.recognize(waveform, sample_rate=_SAMPLE_RATE):
                text = result.text.strip()
                segments.append(Segment(start=result.start, end=result.end, text=text))
                if self._on_progress is not None:
                    self._on_progress(result.end, duration)
        meta = TranscriptMeta(
            date=datetime.date.fromtimestamp(audio.stat().st_mtime),
            title=audio.stem,
            participants=[],
            source=audio.name,
            duration=duration,
        )
        return Transcript(meta=meta, segments=segments)
