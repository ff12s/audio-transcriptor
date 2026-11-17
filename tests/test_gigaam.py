import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from meeting_notes import cuda_setup, gigaam
from meeting_notes.errors import CudaUnavailableError, MeetingNotesError
from meeting_notes.gigaam import DEVICES, GigaamTranscriber, check_config

_MODEL = "gigaam-v3-e2e-rnnt"
_CPU = "CPUExecutionProvider"
_CUDA = "CUDAExecutionProvider"


class _Fakes:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.model_providers: list[list[str]] = []
        self.events_at_load: list[list[str]] = []
        self.recognize_error: BaseException | None = None


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: list[str],
    prepare_error: Exception | None = None,
) -> _Fakes:
    fakes = _Fakes()

    class _Segmenter:
        def recognize(self, _waveform: Any, sample_rate: int = 16000) -> Any:
            if fakes.recognize_error is not None:
                raise fakes.recognize_error
            return iter([SimpleNamespace(start=0.0, end=1.0, text="a")])

    class _Asr:
        def with_vad(self, _vad: Any) -> _Segmenter:
            return _Segmenter()

    def load_vad(name: str, path: Path, **kwargs: Any) -> str:
        return "vad"

    def load_model(name: str, path: Path, **kwargs: Any) -> _Asr:
        fakes.model_providers.append(kwargs["providers"])
        fakes.events_at_load.append(list(fakes.events))
        return _Asr()

    onnx_asr = types.ModuleType("onnx_asr")
    onnx_asr.load_vad = load_vad  # type: ignore[attr-defined]
    onnx_asr.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", onnx_asr)

    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: list(available)  # type: ignore[attr-defined]
    ort.preload_dlls = lambda *_a, **_k: fakes.events.append("preload_dlls")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    def prepare_cuda() -> None:
        fakes.events.append("prepare_cuda")
        if prepare_error is not None:
            raise prepare_error

    monkeypatch.setattr(cuda_setup, "prepare_cuda", prepare_cuda)
    monkeypatch.setattr(gigaam, "_decode", lambda _audio: [0.0] * 16000)
    return fakes


def _audio(tmp_path: Path) -> Path:
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"x")
    return audio


def test_auto_without_cuda_provider_uses_cpu_and_skips_cuda_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = _install(monkeypatch, available=[_CPU])
    with GigaamTranscriber(_MODEL, tmp_path / "models", "auto") as t:
        t.transcribe(_audio(tmp_path))
    assert fakes.model_providers == [[_CPU]]
    assert fakes.events == []


def test_cpu_device_ignores_listed_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = _install(monkeypatch, available=[_CUDA, _CPU])
    with GigaamTranscriber(_MODEL, tmp_path / "models", "cpu") as t:
        t.transcribe(_audio(tmp_path))
    assert fakes.model_providers == [[_CPU]]
    assert fakes.events == []


def test_cuda_device_without_cuda_provider_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, available=[_CPU])
    with (
        GigaamTranscriber(_MODEL, tmp_path / "models", "cuda") as t,
        pytest.raises(MeetingNotesError, match="MEETING_NOTES_DEVICE"),
    ):
        t.transcribe(_audio(tmp_path))


def test_cuda_device_propagates_cuda_unavailable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, available=[_CUDA, _CPU], prepare_error=CudaUnavailableError("no gpu"))
    with (
        GigaamTranscriber(_MODEL, tmp_path / "models", "cuda") as t,
        pytest.raises(CudaUnavailableError, match="no gpu"),
    ):
        t.transcribe(_audio(tmp_path))


def test_cuda_device_prepares_cuda_and_preloads_dlls_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = _install(monkeypatch, available=[_CUDA, _CPU])
    with GigaamTranscriber(_MODEL, tmp_path / "models", "cuda") as t:
        t.transcribe(_audio(tmp_path))
    assert fakes.model_providers == [[_CUDA, _CPU]]
    assert fakes.events_at_load == [["prepare_cuda", "preload_dlls"]]


def test_coreml_device_uses_coreml_provider_with_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = _install(monkeypatch, available=[_CPU, "CoreMLExecutionProvider"])
    with GigaamTranscriber(_MODEL, tmp_path / "models", "coreml") as t:
        t.transcribe(_audio(tmp_path))
    assert fakes.model_providers == [["CoreMLExecutionProvider", _CPU]]
    assert fakes.events == []


@pytest.mark.parametrize("device", ["dml", "coreml"])
def test_accelerator_device_without_its_provider_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    _install(monkeypatch, available=[_CPU])
    with (
        GigaamTranscriber(_MODEL, tmp_path / "models", device) as t,
        pytest.raises(MeetingNotesError) as info,
    ):
        t.transcribe(_audio(tmp_path))
    assert repr(device) in str(info.value)
    assert "MEETING_NOTES_DEVICE=auto" in str(info.value)


def test_unreadable_audio_names_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, available=[_CPU])

    def failing_decode(_audio: Path) -> Any:
        raise ValueError("bad stream")

    monkeypatch.setattr(gigaam, "_decode", failing_decode)
    with (
        GigaamTranscriber(_MODEL, tmp_path / "models", "cpu") as t,
        pytest.raises(MeetingNotesError, match="meeting.mp3"),
    ):
        t.transcribe(_audio(tmp_path))


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()])
def test_recognition_errors_propagate_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    fakes = _install(monkeypatch, available=[_CPU])
    fakes.recognize_error = error
    with (
        GigaamTranscriber(_MODEL, tmp_path / "models", "cpu") as t,
        pytest.raises(type(error)) as info,
    ):
        t.transcribe(_audio(tmp_path))
    assert info.value is error


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "dml", "coreml"])
def test_check_config_accepts_every_device(device: str) -> None:
    check_config(_MODEL, device, "ru")


def test_devices_lists_all_supported_values() -> None:
    assert DEVICES == ("auto", "cpu", "cuda", "dml", "coreml")


@pytest.mark.parametrize("model", ["large-v3", "gigaam-v3/x"])
def test_check_config_rejects_non_gigaam_model(model: str) -> None:
    with pytest.raises(MeetingNotesError) as info:
        check_config(model, "auto", "ru")
    message = str(info.value)
    assert repr(model) in message
    assert "--engine whisper" in message
    assert "MEETING_NOTES_GIGAAM_MODEL" in message


@pytest.mark.parametrize("device", ["cuda:0", "CUDA", "gpu"])
def test_check_config_rejects_unknown_device(device: str) -> None:
    with pytest.raises(MeetingNotesError) as info:
        check_config(_MODEL, device, "ru")
    message = str(info.value)
    assert repr(device) in message
    assert "--engine whisper" in message
    assert "MEETING_NOTES_DEVICE" in message


def test_check_config_rejects_non_russian_language() -> None:
    with pytest.raises(MeetingNotesError) as info:
        check_config(_MODEL, "auto", "en")
    message = str(info.value)
    assert repr("en") in message
    assert "MEETING_NOTES_LANGUAGE" in message
