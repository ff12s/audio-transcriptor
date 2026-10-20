import sys
import types
from pathlib import Path
from typing import Any

import pytest

from meeting_notes import cuda_setup
from meeting_notes.config import Settings
from meeting_notes.errors import CudaOutOfMemoryError, CudaUnavailableError, MeetingNotesError
from meeting_notes.transcriber import WhisperTranscriber

_KNOWN_MODELS = ["tiny", "large-v3"]


class _FakeModel:
    supported_languages = ["ru", "en"]

    def __init__(self, error: Exception) -> None:
        self.error = error

    def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
        raise self.error


@pytest.fixture(autouse=True)
def _isolated_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEETING_NOTES_MODEL_DIR", str(tmp_path / "models"))


@pytest.fixture
def prepare_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(cuda_setup, "prepare_cuda", lambda: calls.append(1))
    return calls


def _transcribe_failing_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, error: Exception
) -> None:
    _install_model(monkeypatch, lambda *_a, **_k: _FakeModel(error))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with WhisperTranscriber(Settings(device=device), None) as t:
        t.transcribe(audio)


def test_cpu_skips_cuda_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_calls: list[int]
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _transcribe_failing_with(tmp_path, monkeypatch, "cpu", RuntimeError("boom"))
    assert prepare_calls == []


def test_cuda_runs_cuda_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_calls: list[int]
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _transcribe_failing_with(tmp_path, monkeypatch, "cuda", RuntimeError("boom"))
    assert prepare_calls == [1]


def test_out_of_memory_has_vram_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_calls: list[int]
) -> None:
    error = RuntimeError("CUDA failed with error Out Of Memory")
    with pytest.raises(CudaOutOfMemoryError) as info:
        _transcribe_failing_with(tmp_path, monkeypatch, "cuda", error)
    assert "int8_float16" in str(info.value)
    assert not isinstance(info.value, CudaUnavailableError)


def test_other_cuda_error_keeps_driver_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_calls: list[int]
) -> None:
    error = RuntimeError("Library cublas64_12.dll is not found")
    with pytest.raises(CudaUnavailableError, match="nvidia-cublas-cu12"):
        _transcribe_failing_with(tmp_path, monkeypatch, "cuda", error)


def _install_model(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = factory  # type: ignore[attr-defined]
    module.available_models = lambda: list(_KNOWN_MODELS)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)


class _FakeInfo:
    duration = 1.0


class _Segment:
    start = 0.0
    end = 1.0
    text = " hi "


class _SegmentsThenError:
    def __init__(self, error: Exception | None = None) -> None:
        self.items: list[_Segment] = [_Segment()]
        self.error = error

    def __iter__(self) -> "_SegmentsThenError":
        return self

    def __next__(self) -> Any:
        if self.items:
            return self.items.pop()
        if self.error is not None:
            raise self.error
        raise StopIteration


class _Model:
    supported_languages = ["ru", "en"]

    def __init__(self, segments: Any) -> None:
        self.segments = segments
        self.transcribe_calls = 0

    def transcribe(self, *_a: Any, **_k: Any) -> Any:
        self.transcribe_calls += 1
        return self.segments, _FakeInfo()


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: Any,
    settings: Settings | None = None,
    on_progress: Any = None,
) -> Any:
    _install_model(monkeypatch, lambda *_a, **_k: model)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with WhisperTranscriber(settings or Settings(device="cpu"), None, on_progress) as t:
        return t.transcribe(audio)


class _FailingSegments:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def __iter__(self) -> "_FailingSegments":
        return self

    def __next__(self) -> Any:
        raise self.error


def test_unsupported_compute_type_has_settings_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValueError("Requested float16 compute type, but the target device does not support it")

    def make_model(*_a: Any, **_k: Any) -> _FakeModel:
        raise error

    _install_model(monkeypatch, make_model)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with (
        WhisperTranscriber(Settings(device="cpu"), None) as t,
        pytest.raises(MeetingNotesError, match="MEETING_NOTES_COMPUTE_TYPE") as info,
    ):
        t.transcribe(audio)
    assert "MEETING_NOTES_DEVICE" in str(info.value)
    assert "int8" in str(info.value)
    assert str(error) in str(info.value)
    assert info.value.__cause__ is error


@pytest.mark.parametrize("failing_step", ["transcribe", "iterate"])
def test_decode_error_is_not_reported_as_compute_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_step: str
) -> None:
    error = ValueError("Invalid data found when processing input")

    class Model:
        supported_languages = ["ru"]

        def transcribe(self, *_a: Any, **_k: Any) -> Any:
            if failing_step == "transcribe":
                raise error
            return _FailingSegments(error), _FakeInfo()

    _install_model(monkeypatch, lambda *_a, **_k: Model())
    audio = tmp_path / "broken.mp3"
    audio.write_bytes(b"x")
    with (
        WhisperTranscriber(Settings(device="cpu"), None) as t,
        pytest.raises(MeetingNotesError) as info,
    ):
        t.transcribe(audio)
    message = str(info.value)
    assert audio.name in message
    assert str(error) in message
    assert "MEETING_NOTES_COMPUTE_TYPE" not in message
    assert "MEETING_NOTES_DEVICE" not in message
    assert info.value.__cause__ is error


def test_unknown_model_is_reported_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []

    def factory(name: str, **_k: Any) -> Any:
        created.append(name)
        return _Model(_SegmentsThenError())

    _install_model(monkeypatch, factory)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    settings = Settings(device="cpu", model="large-v4")
    with WhisperTranscriber(settings, None) as t, pytest.raises(MeetingNotesError) as info:
        t.transcribe(audio)
    message = str(info.value)
    assert "'large-v4'" in message
    assert all(name in message for name in _KNOWN_MODELS)
    assert "MEETING_NOTES_MODEL" in message
    assert "MEETING_NOTES_COMPUTE_TYPE" not in message
    assert created == []


@pytest.mark.parametrize("name", ["D:/models/typo", "./x", "a/b/c", "org/"])
def test_mistyped_local_path_is_not_treated_as_hf_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    created: list[str] = []

    def factory(model_name: str, **_k: Any) -> Any:
        created.append(model_name)
        return _Model(_SegmentsThenError())

    _install_model(monkeypatch, factory)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with (
        WhisperTranscriber(Settings(device="cpu", model=name), None) as t,
        pytest.raises(MeetingNotesError) as info,
    ):
        t.transcribe(audio)
    message = str(info.value)
    assert repr(name) in message
    assert "существующий каталог" in message
    assert created == []


def test_model_dir_is_created_and_passed_as_download_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "nested" / "whisper"
    received: dict[str, Any] = {}

    def factory(*_a: Any, **kwargs: Any) -> Any:
        received.update(kwargs)
        return _Model(_SegmentsThenError())

    _install_model(monkeypatch, factory)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with WhisperTranscriber(Settings(device="cpu", model_dir=model_dir), None) as t:
        t.transcribe(audio)
    assert received["download_root"] == str(model_dir)
    assert model_dir.is_dir()


def test_unusable_model_dir_names_dir_and_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "file"
    model_dir.write_bytes(b"x")
    settings = Settings(device="cpu", model_dir=model_dir)
    with pytest.raises(MeetingNotesError) as info:
        _run(tmp_path, monkeypatch, _Model(_SegmentsThenError()), settings)
    assert f"{model_dir}: " in str(info.value)
    assert "MEETING_NOTES_MODEL_DIR" in str(info.value)
    assert isinstance(info.value.__cause__, OSError)


@pytest.mark.parametrize(
    "error",
    [OSError("Not enough free disk space"), RuntimeError("Background writer channel closed")],
)
def test_model_download_failure_has_disk_and_network_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def make_model(*_a: Any, **_k: Any) -> Any:
        raise error

    model_dir = tmp_path / "models"
    settings = Settings(device="cpu", model_dir=model_dir)
    _install_model(monkeypatch, make_model)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with WhisperTranscriber(settings, None) as t, pytest.raises(MeetingNotesError) as info:
        t.transcribe(audio)
    message = str(info.value)
    assert repr(settings.model) in message
    assert str(model_dir) in message
    assert str(error) in message
    assert "свободное место" in message
    assert "MEETING_NOTES_MODEL_DIR" in message
    assert info.value.__cause__ is error


def test_cuda_runtime_error_during_model_load_keeps_cuda_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = RuntimeError("CUDA failed with error out of memory")

    def make_model(*_a: Any, **_k: Any) -> Any:
        raise error

    _install_model(monkeypatch, make_model)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with WhisperTranscriber(Settings(device="cpu"), None) as t, pytest.raises(CudaOutOfMemoryError):
        t.transcribe(audio)


class _ConnectError(Exception):
    """Как httpx2.ConnectError: не наследует OSError."""


def _load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> MeetingNotesError:
    def make_model(*_a: Any, **_k: Any) -> Any:
        raise error

    _install_model(monkeypatch, make_model)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with (
        WhisperTranscriber(Settings(device="cpu"), None) as t,
        pytest.raises(MeetingNotesError) as info,
    ):
        t.transcribe(audio)
    assert info.value.__cause__ is error
    return info.value


def test_non_os_error_during_model_load_names_exception_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = _ConnectError("All connection attempts failed")
    message = str(_load_failure(tmp_path, monkeypatch, error))
    assert "_ConnectError: All connection attempts failed" in message
    assert "MEETING_NOTES_MODEL_DIR" in message
    assert "MEETING_NOTES_COMPUTE_TYPE" not in message


def test_invalid_repo_id_is_load_failure_not_compute_type_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValueError("Repo id must use alphanumeric chars or '-', '_', '.'")
    message = str(_load_failure(tmp_path, monkeypatch, error))
    assert "ValueError: Repo id must" in message
    assert "свободное место" in message
    assert "MEETING_NOTES_COMPUTE_TYPE" not in message


def test_compute_type_value_error_keeps_settings_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValueError("Requested float16 compute type, but the target device is unsupported")
    message = str(_load_failure(tmp_path, monkeypatch, error))
    assert "MEETING_NOTES_COMPUTE_TYPE" in message
    assert "свободное место" not in message


def test_cuda_error_of_any_class_during_model_load_is_cuda_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = _ConnectError("cudnn64_9.dll is not found")
    assert isinstance(_load_failure(tmp_path, monkeypatch, error), CudaUnavailableError)


def test_keyboard_interrupt_during_model_load_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make_model(*_a: Any, **_k: Any) -> Any:
        raise KeyboardInterrupt

    _install_model(monkeypatch, make_model)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    with WhisperTranscriber(Settings(device="cpu"), None) as t, pytest.raises(KeyboardInterrupt):
        t.transcribe(audio)


def test_local_model_dir_does_not_create_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "models"
    settings = Settings(device="cpu", model=str(tmp_path), model_dir=model_dir)
    _run(tmp_path, monkeypatch, _Model(_SegmentsThenError()), settings)
    assert not model_dir.exists()


@pytest.mark.parametrize("kind", ["hf_repo", "local_dir"])
def test_hf_repo_id_and_local_dir_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    name = "org/name" if kind == "hf_repo" else str(tmp_path)
    settings = Settings(device="cpu", model=name)
    transcript = _run(tmp_path, monkeypatch, _Model(_SegmentsThenError()), settings)
    assert [s.text for s in transcript.segments] == ["hi"]


def test_unsupported_language_is_reported_before_transcribing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Model(_SegmentsThenError())
    settings = Settings(device="cpu", language="xx")
    with pytest.raises(MeetingNotesError, match="MEETING_NOTES_LANGUAGE") as info:
        _run(tmp_path, monkeypatch, model, settings)
    assert "'xx'" in str(info.value)
    assert settings.model in str(info.value)
    assert model.transcribe_calls == 0


def test_os_error_during_iteration_is_reported_as_decode_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = OSError("Input/output error")
    with pytest.raises(MeetingNotesError) as info:
        _run(tmp_path, monkeypatch, _Model(_SegmentsThenError(error)))
    assert "a.mp3" in str(info.value)
    assert str(error) in str(info.value)
    assert info.value.__cause__ is error


@pytest.mark.parametrize("error", [ValueError("cb boom"), OSError("cb boom")])
def test_progress_callback_error_propagates_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def on_progress(_done: float, _total: float) -> None:
        raise error

    with pytest.raises(type(error)) as info:
        _run(tmp_path, monkeypatch, _Model(_SegmentsThenError()), on_progress=on_progress)
    assert info.value is error


def test_progress_callback_receives_segment_end_and_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[float, float]] = []
    _run(
        tmp_path,
        monkeypatch,
        _Model(_SegmentsThenError()),
        on_progress=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(_Segment.end, _FakeInfo.duration)]
