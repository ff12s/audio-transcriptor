import sys
import threading
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from meeting_notes.capture import (
    AppSource,
    CallRecorder,
    MicSource,
    SourceWarning,
    _pcm_to_block,
)
from meeting_notes.errors import MeetingNotesError
from meeting_notes.track import SOURCE_RATE, TRACK_RATE, TrackWriter

Block = npt.NDArray[np.float32]
BLOCK_FRAMES = SOURCE_RATE // 20


class FakeSource:
    """Шлёт тихие блоки из своего потока, как делают реальные источники."""

    def __init__(self, fail_after_blocks: int | None = None) -> None:
        self.fail_after_blocks = fail_after_blocks
        self.stopped = False
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self, on_block: Callable[[Block], None], on_error: Callable[[Exception], None]
    ) -> None:
        def work() -> None:
            sent = 0
            while not self._halt.wait(0.05):
                if self.fail_after_blocks is not None and sent >= self.fail_after_blocks:
                    on_error(RuntimeError("приложение закрылось"))
                    return
                on_block(np.zeros((BLOCK_FRAMES, 2), dtype=np.float32))
                sent += 1

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stopped = True
        self._halt.set()
        if self._thread is not None:
            self._thread.join()


class RaisingStop(FakeSource):
    def stop(self) -> None:
        super().stop()
        raise RuntimeError("не остановился")


def _frames(path: Path) -> int:
    with wave.open(str(path), "rb") as r:
        assert (r.getnchannels(), r.getframerate(), r.getsampwidth()) == (1, TRACK_RATE, 2)
        return r.getnframes()


def _interrupt_on_tick(n: int) -> Callable[[float], None]:
    ticks = 0

    def on_tick(_elapsed: float) -> None:
        nonlocal ticks
        ticks += 1
        if ticks == n:
            raise KeyboardInterrupt

    return on_tick


def _recorder(
    tmp_path: Path,
    mic: FakeSource,
    apps: list[FakeSource],
    warnings: list[str] | None = None,
) -> tuple[CallRecorder, Path, Path]:
    me_path, them_path = tmp_path / "x.me.wav", tmp_path / "x.them.wav"
    recorder = CallRecorder(
        TrackWriter(me_path),
        TrackWriter(them_path),
        mic,
        list(apps),
        on_warning=warnings.append if warnings is not None else None,
    )
    return recorder, me_path, them_path


def test_interrupt_closes_both_tracks(tmp_path: Path) -> None:
    mic, app = FakeSource(), FakeSource()
    recorder, me_path, them_path = _recorder(tmp_path, mic, [app])
    duration = recorder.record_until_interrupted(on_tick=_interrupt_on_tick(3))
    assert mic.stopped and app.stopped
    assert _frames(me_path) == pytest.approx(duration * TRACK_RATE, abs=0.05 * TRACK_RATE)
    assert _frames(them_path) == pytest.approx(duration * TRACK_RATE, abs=0.05 * TRACK_RATE)


def test_failing_stop_still_closes_tracks(tmp_path: Path) -> None:
    mic, app = RaisingStop(), FakeSource()
    recorder, me_path, them_path = _recorder(tmp_path, mic, [app])
    recorder.record_until_interrupted(on_tick=_interrupt_on_tick(2))
    assert app.stopped
    assert _frames(me_path) > 0
    assert _frames(them_path) > 0


def test_app_failure_keeps_recording(tmp_path: Path) -> None:
    warnings: list[str] = []
    mic, app = FakeSource(), FakeSource(fail_after_blocks=2)
    recorder, me_path, them_path = _recorder(tmp_path, mic, [app], warnings)
    recorder.record_until_interrupted(on_tick=_interrupt_on_tick(8))
    assert len(warnings) == 1
    assert "приложение закрылось" in warnings[0]
    assert abs(_frames(them_path) - _frames(me_path)) < 0.05 * TRACK_RATE


def test_start_failure_is_a_warning(tmp_path: Path) -> None:
    class BrokenStart(FakeSource):
        def start(self, *_args: object) -> None:
            raise MeetingNotesError("нет доступа")

    warnings: list[str] = []
    recorder, me_path, _ = _recorder(tmp_path, FakeSource(), [BrokenStart()], warnings)
    recorder.record_until_interrupted(on_tick=_interrupt_on_tick(2))
    assert len(warnings) == 1
    assert "нет доступа" in warnings[0]
    assert _frames(me_path) > 0


def test_bytes_to_frames() -> None:
    block = _pcm_to_block(np.ones(8, dtype=np.float32).tobytes())
    assert block.shape == (4, 2)
    assert block.dtype == np.float32


def test_app_start_error_is_user_facing(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def __init__(self, pid: int, on_data: object = None) -> None:
            raise OSError("access denied")

    monkeypatch.setattr("proctap.ProcessAudioCapture", Broken)
    with pytest.raises(MeetingNotesError, match="1234.*access denied"):
        AppSource(1234).start(lambda _b: None, lambda _e: None)


def test_missing_microphone_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_device() -> object:
        raise RuntimeError("no device")

    monkeypatch.setattr("soundcard.default_microphone", no_device)
    with pytest.raises(MeetingNotesError, match="Микрофон не найден.*no device"):
        MicSource()


def test_microphone_init_failure_is_not_called_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # None в sys.modules заставляет import soundcard упасть, как при сбое инициализации COM.
    monkeypatch.setitem(sys.modules, "soundcard", None)
    with pytest.raises(MeetingNotesError, match="Не удалось открыть микрофон") as info:
        MicSource()
    assert "Подключите" not in str(info.value)


def test_disk_error_on_finish_is_user_facing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, me_path, them_path = _recorder(tmp_path, FakeSource(), [FakeSource()])

    def full_disk(_block: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(recorder._me, "write", full_disk)
    with pytest.raises(MeetingNotesError, match="No space left"):
        recorder.record_until_interrupted(on_tick=_interrupt_on_tick(8))
    assert _frames(them_path) >= 0 and _frames(me_path) == 0


def test_stop_failure_is_reported(tmp_path: Path) -> None:
    warnings: list[str] = []
    recorder, _, _ = _recorder(tmp_path, RaisingStop(), [FakeSource()], warnings)
    recorder.record_until_interrupted(on_tick=_interrupt_on_tick(2))
    assert any("микрофон" in w and "не остановился" in w for w in warnings)


def test_repeated_errors_are_counted(tmp_path: Path) -> None:
    class Chatty(FakeSource):
        def start(self, on_block: object, on_error: Callable[[Exception], None]) -> None:
            for _ in range(3):
                on_error(SourceWarning("битый пакет"))

    warnings: list[str] = []
    recorder, _, _ = _recorder(tmp_path, FakeSource(), [Chatty()], warnings)
    recorder.record_until_interrupted(on_tick=_interrupt_on_tick(2))
    assert sum("битый пакет" in w for w in warnings) == 1
    assert any("ещё 2 ошибок" in w for w in warnings)


def test_distinct_warnings_are_each_shown(tmp_path: Path) -> None:
    class Chatty(FakeSource):
        def start(self, on_block: object, on_error: Callable[[Exception], None]) -> None:
            for text in ("битый пакет", "пропуск данных", "битый пакет"):
                on_error(SourceWarning(text))

    warnings: list[str] = []
    recorder, _, _ = _recorder(tmp_path, FakeSource(), [Chatty()], warnings)
    recorder.record_until_interrupted(on_tick=_interrupt_on_tick(2))
    assert sum("битый пакет" in w for w in warnings) == 1
    assert sum("пропуск данных" in w for w in warnings) == 1
    assert any("ещё 1 ошибок" in w for w in warnings)


class WarnThenDie(FakeSource):
    """Сначала безобидное предупреждение, затем источник умирает."""

    def start(self, on_block: object, on_error: Callable[[Exception], None]) -> None:
        on_error(SourceWarning("data discontinuity in recording"))
        on_error(RuntimeError("устройство отключено"))


@pytest.mark.parametrize(("role", "track"), [("mic", "Дорожка «Я»"), ("app", "Собеседники")])
def test_fatal_error_is_shown_after_earlier_warning(tmp_path: Path, role: str, track: str) -> None:
    warnings: list[str] = []
    mic, app = (WarnThenDie(), FakeSource()) if role == "mic" else (FakeSource(), WarnThenDie())
    recorder, _, _ = _recorder(tmp_path, mic, [app], warnings)
    during: list[str] = []
    ticks = 0

    def on_tick(_elapsed: float) -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 3:
            # Итоговые сообщения пишутся при остановке; здесь только то, что показано по ходу.
            during.extend(warnings)
            raise KeyboardInterrupt

    recorder.record_until_interrupted(on_tick=on_tick)
    fatal = [w for w in during if "устройство отключено" in w]
    assert len(fatal) == 1
    assert "перестал записывать" in fatal[0]
    assert track in fatal[0]
    assert "по часам" not in fatal[0]


def test_app_exit_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class Capture:
        def __init__(self, pid: int, on_data: object = None) -> None: ...

        def start(self) -> None: ...

        def close(self) -> None: ...

    monkeypatch.setattr("proctap.ProcessAudioCapture", Capture)
    monkeypatch.setattr("psutil.pid_exists", lambda _pid: False)
    monkeypatch.setattr("meeting_notes.capture._ALIVE_CHECK_SECONDS", 0.01)
    errors: list[Exception] = []
    got = threading.Event()
    source = AppSource(1234)
    source.start(lambda _b: None, lambda e: (errors.append(e), got.set()))
    assert got.wait(2)
    source.stop()
    assert "1234" in str(errors[0])


def test_mic_discontinuity_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import warnings

    class SoundcardRuntimeWarning(RuntimeWarning): ...

    class Rec:
        def __enter__(self) -> "Rec":
            return self

        def __exit__(self, *_args: object) -> None: ...

        def record(self, numframes: int) -> Block:
            warnings.warn("data discontinuity in recording", SoundcardRuntimeWarning, stacklevel=2)
            return np.zeros((numframes, 2), dtype=np.float32)

    class Mic:
        def recorder(self, samplerate: int, blocksize: int) -> Rec:
            assert blocksize > SOURCE_RATE // 10
            return Rec()

    monkeypatch.setattr("soundcard.default_microphone", lambda: Mic())
    errors: list[Exception] = []
    got = threading.Event()
    mic = MicSource()
    mic.start(lambda _b: None, lambda e: (errors.append(e), got.set()))
    assert got.wait(2)
    mic.stop()
    assert "discontinuity" in str(errors[0])
