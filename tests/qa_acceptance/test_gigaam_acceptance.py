import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from meeting_notes import cuda_setup
from meeting_notes.errors import CudaUnavailableError, MeetingNotesError

_MODEL = "gigaam-v3-e2e-rnnt"


class _Fakes:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.model_calls: list[tuple[str, Path, list[str]]] = []
        self.vad_calls: list[tuple[str, Path, list[str]]] = []
        self.recognize_calls = 0


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: list[str],
    waveform: Any = None,
    results: list[Any] | None = None,
    prepare_error: Exception | None = None,
    load_error: Exception | None = None,
) -> _Fakes:
    fakes = _Fakes()
    items = results or []

    class _Segmenter:
        def recognize(self, _waveform: Any, sample_rate: int = 16000) -> Any:
            fakes.recognize_calls += 1
            return iter(items)

    class _Asr:
        def with_vad(self, _vad: Any) -> _Segmenter:
            return _Segmenter()

    def load_vad(name: str, path: Path | None = None, **kwargs: Any) -> str:
        fakes.events.append("load_vad")
        fakes.vad_calls.append((name, Path(str(path)), kwargs["providers"]))
        return "vad"

    def load_model(name: str, path: Path | None = None, **kwargs: Any) -> _Asr:
        fakes.events.append("load_model")
        fakes.model_calls.append((name, Path(str(path)), kwargs["providers"]))
        if load_error is not None:
            raise load_error
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

    fw = types.ModuleType("faster_whisper")
    fw_audio = types.ModuleType("faster_whisper.audio")
    fw_audio.decode_audio = lambda *_a, **_k: waveform  # type: ignore[attr-defined]
    fw.audio = fw_audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", fw_audio)
    return fakes


def _result(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text)


def _audio(tmp_path: Path) -> Path:
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"x")
    return audio


def test_transcribe_maps_results_to_stripped_segments_meta_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes.gigaam import GigaamTranscriber

    waveform = np.zeros(64000, dtype=np.float32)
    _install(
        monkeypatch,
        available=["CPUExecutionProvider"],
        waveform=waveform,
        results=[_result(0.0, 2.5, "  Привет "), _result(2.5, 4.0, "мир ")],
    )
    progress: list[tuple[float, float]] = []
    transcriber = GigaamTranscriber(
        _MODEL, tmp_path / "models", "cpu", lambda done, total: progress.append((done, total))
    )
    with transcriber as t:
        transcript = t.transcribe(_audio(tmp_path))
    expected_duration = len(waveform) / 16000
    assert [(s.start, s.end, s.text) for s in transcript.segments] == [
        (0.0, 2.5, "Привет"),
        (2.5, 4.0, "мир"),
    ]
    assert transcript.meta.title == "meeting"
    assert transcript.meta.source == "meeting.mp3"
    assert transcript.meta.duration == expected_duration
    assert progress == [(2.5, expected_duration), (4.0, expected_duration)]


def test_auto_with_cuda_runs_prepare_then_preload_then_loads_on_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes.gigaam import GigaamTranscriber

    fakes = _install(
        monkeypatch,
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        waveform=np.zeros(16000, dtype=np.float32),
        results=[_result(0.0, 1.0, "a")],
    )
    model_dir = tmp_path / "nested" / "models"
    with GigaamTranscriber(_MODEL, model_dir, "auto") as t:
        t.transcribe(_audio(tmp_path))
    events = fakes.events
    assert events.index("prepare_cuda") < events.index("preload_dlls") < events.index("load_model")
    assert fakes.model_calls == [
        (_MODEL, model_dir / _MODEL, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    ]
    assert fakes.vad_calls == [("silero", model_dir / "silero-vad", ["CPUExecutionProvider"])]
    assert model_dir.is_dir()
    assert not (model_dir / _MODEL).exists()
    assert not (model_dir / "silero-vad").exists()


def test_auto_falls_back_to_cpu_when_cuda_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes.gigaam import GigaamTranscriber

    fakes = _install(
        monkeypatch,
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        waveform=np.zeros(16000, dtype=np.float32),
        results=[_result(0.0, 1.0, "a")],
        prepare_error=CudaUnavailableError("no gpu"),
    )
    with GigaamTranscriber(_MODEL, tmp_path / "models", "auto") as t:
        transcript = t.transcribe(_audio(tmp_path))
    assert fakes.model_calls[0][2] == ["CPUExecutionProvider"]
    assert "preload_dlls" not in fakes.events
    assert [s.text for s in transcript.segments] == ["a"]


def test_dml_uses_dml_provider_with_cpu_fallback_and_skips_cuda_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes.gigaam import GigaamTranscriber

    fakes = _install(
        monkeypatch,
        available=["CPUExecutionProvider", "DmlExecutionProvider"],
        waveform=np.zeros(16000, dtype=np.float32),
        results=[_result(0.0, 1.0, "a")],
    )
    with GigaamTranscriber(_MODEL, tmp_path / "models", "dml") as t:
        t.transcribe(_audio(tmp_path))
    assert fakes.model_calls[0][2] == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert "prepare_cuda" not in fakes.events


def test_model_load_failure_becomes_chained_meeting_notes_error_with_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes.gigaam import GigaamTranscriber

    error = RuntimeError("weights are corrupted")
    _install(
        monkeypatch,
        available=["CPUExecutionProvider"],
        waveform=np.zeros(16000, dtype=np.float32),
        load_error=error,
    )
    model_dir = tmp_path / "models"
    with (
        GigaamTranscriber(_MODEL, model_dir, "cpu") as t,
        pytest.raises(MeetingNotesError) as info,
    ):
        t.transcribe(_audio(tmp_path))
    message = str(info.value)
    assert repr(_MODEL) in message
    assert str(model_dir) in message
    assert "RuntimeError" in message
    assert "weights are corrupted" in message
    assert "MEETING_NOTES_MODEL_DIR" in message
    assert str(model_dir / _MODEL) in message
    assert str(model_dir / "silero-vad") in message
    assert "MEETING_NOTES_DEVICE=cpu" in message
    assert info.value.__cause__ is error


def test_check_config_rejects_ctranslate2_device_string() -> None:
    from meeting_notes.gigaam import check_config

    with pytest.raises(MeetingNotesError) as info:
        check_config(_MODEL, "cuda:0", "ru")
    message = str(info.value)
    assert repr("cuda:0") in message
    for value in ["auto", "cpu", "cuda", "dml", "coreml"]:
        assert value in message
    assert "MEETING_NOTES_DEVICE" in message
    assert "--engine whisper" in message


def test_empty_waveform_skips_recognition_and_gives_empty_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes.gigaam import GigaamTranscriber

    fakes = _install(
        monkeypatch,
        available=["CPUExecutionProvider"],
        waveform=np.zeros(0, dtype=np.float32),
        results=[_result(0.0, 1.0, "unused")],
    )
    with GigaamTranscriber(_MODEL, tmp_path / "models", "cpu") as t:
        transcript = t.transcribe(_audio(tmp_path))
    assert fakes.recognize_calls == 0
    assert transcript.segments == []
    assert transcript.meta.duration == 0.0


def test_model_is_loaded_once_and_reloaded_after_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_notes.gigaam import GigaamTranscriber

    fakes = _install(
        monkeypatch,
        available=["CPUExecutionProvider"],
        waveform=np.zeros(16000, dtype=np.float32),
        results=[_result(0.0, 1.0, "a")],
    )
    audio = _audio(tmp_path)
    transcriber = GigaamTranscriber(_MODEL, tmp_path / "models", "cpu")
    with transcriber as t:
        t.transcribe(audio)
        t.transcribe(audio)
    assert len(fakes.model_calls) == 1
    transcriber.transcribe(audio)
    assert len(fakes.model_calls) == 2
