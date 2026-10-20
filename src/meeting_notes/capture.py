import sys
import threading
import time
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from typing import Any, Protocol, TextIO

import numpy as np
import numpy.typing as npt

from meeting_notes.errors import MeetingNotesError
from meeting_notes.track import SOURCE_RATE, ClockedTrack, TrackWriter

Block = npt.NDArray[np.float32]

_TICK_SECONDS = 0.1
_MIC_BLOCK_FRAMES = SOURCE_RATE // 10
# WASAPI может недобирать буфер при blocksize == numframes; запас снижает число пропусков.
_MIC_BUFFER_FRAMES = _MIC_BLOCK_FRAMES * 2
# Потеря меньше этого порога - округление и дрожание доставки, а не пропавшая речь.
_MIN_REPORTED_LOSS = 0.5
# Как часто проверять, что процесс приложения ещё жив.
_ALIVE_CHECK_SECONDS = 1.0
# Поток микрофона блокируется в record(); на остановке ждём его недолго и бросаем.
_JOIN_TIMEOUT = 2.0


class SourceWarning(Exception):
    """Нефатальная неполадка источника: он продолжает работать, но звук мог пропасть.

    Любая другая ошибка, переданная в on_error, означает, что источник перестал писать.
    """


class AudioSource(Protocol):
    def start(
        self, on_block: Callable[[Block], None], on_error: Callable[[Exception], None]
    ) -> None: ...

    def stop(self) -> None: ...


def _pcm_to_block(pcm: bytes) -> Block:
    # proc-tap отдаёт 48 kHz, 2 канала, float32 подряд.
    return np.frombuffer(pcm, dtype=np.float32).reshape(-1, 2)


class AppSource:
    """Звук процесса и его потомков через Process Loopback (proc-tap)."""

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._capture: Any = None
        self._halt = threading.Event()
        self._watchdog: threading.Thread | None = None

    def start(
        self, on_block: Callable[[Block], None], on_error: Callable[[Exception], None]
    ) -> None:
        from proctap import ProcessAudioCapture

        def on_data(data: bytes, _frames: int) -> None:
            try:
                on_block(_pcm_to_block(data))
            except Exception as exc:
                on_error(SourceWarning(str(exc)))

        try:
            self._capture = ProcessAudioCapture(self._pid, on_data=on_data)
            self._capture.start()
        except Exception as exc:
            raise MeetingNotesError(
                f"Не удалось начать захват звука процесса {self._pid}: {exc}"
            ) from exc

        # proc-tap глотает ошибки чтения и при закрытии приложения просто перестаёт слать пакеты,
        # так что о закрытии узнаём только по самому процессу.
        def watch() -> None:
            import psutil

            while not self._halt.wait(_ALIVE_CHECK_SECONDS):
                if not psutil.pid_exists(self._pid):
                    on_error(RuntimeError(f"процесс {self._pid} завершился"))
                    return

        self._watchdog = threading.Thread(target=watch, name="app-watchdog", daemon=True)
        self._watchdog.start()

    def stop(self) -> None:
        self._halt.set()
        if self._capture is not None:
            self._capture.close()
            self._capture = None


class MicSource:
    def __init__(self) -> None:
        # Устройство ищем сразу: без микрофона запись не должна начаться.
        try:
            import soundcard
        except Exception as exc:
            # Сбой импорта или инициализации COM не лечится подключением микрофона.
            raise MeetingNotesError(f"Не удалось открыть микрофон: {exc}") from exc
        try:
            self._mic = soundcard.default_microphone()
        except Exception as exc:
            raise MeetingNotesError(
                f"Микрофон не найден: {exc}. Подключите микрофон и выберите его "
                "устройством ввода по умолчанию в настройках звука Windows."
            ) from exc
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._show: Callable[..., None] | None = None
        self._original_show: Callable[..., None] = warnings.showwarning

    def start(
        self, on_block: Callable[[Block], None], on_error: Callable[[Exception], None]
    ) -> None:
        # Пропуски данных soundcard сообщает через warnings; перехватываем их, чтобы они дошли до
        # пользователя, а не потерялись в stderr под счётчиком времени. catch_warnings здесь нельзя:
        # он подменяет фильтры всего процесса, а не потока, и проглотил бы чужие предупреждения.
        # Фильтр "always" для этой категории soundcard выставляет сам при импорте.
        original = warnings.showwarning

        def show(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: TextIO | None = None,
            line: str | None = None,
        ) -> None:
            if category.__name__ == "SoundcardRuntimeWarning":
                on_error(SourceWarning(str(message)))
            else:
                original(message, category, filename, lineno, file, line)

        self._show = show
        warnings.showwarning = show

        def work() -> None:
            try:
                # Все каналы: на WASAPI запись одного канала даёт мусор, в моно сводит дорожка.
                with self._mic.recorder(
                    samplerate=SOURCE_RATE, blocksize=_MIC_BUFFER_FRAMES
                ) as recorder:
                    while not self._stop.is_set():
                        on_block(recorder.record(numframes=_MIC_BLOCK_FRAMES))
            except Exception as exc:
                on_error(exc)

        self._original_show = original
        self._thread = threading.Thread(target=work, name="mic-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Возвращаем хук, только если его не успел заменить кто-то ещё.
        if self._show is not None and warnings.showwarning is self._show:
            warnings.showwarning = self._original_show
        self._show = None
        if self._thread is not None:
            self._thread.join(timeout=_JOIN_TIMEOUT)
            if self._thread.is_alive():
                raise MeetingNotesError("поток записи микрофона не остановился")


def _warn_stderr(message: str) -> None:
    print(message, file=sys.stderr)


class CallRecorder:
    def __init__(
        self,
        me: TrackWriter,
        them: TrackWriter,
        mic: AudioSource,
        apps: list[AudioSource],
        clock: Callable[[], float] = time.monotonic,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self._me = me
        self._them = them
        self._mic = mic
        self._apps = apps
        self._clock = clock
        # Без получателя предупреждения не должны пропадать: это единственный сигнал о тишине.
        self._on_warning = on_warning or _warn_stderr
        self._repeats: dict[str, int] = {}
        self._seen: set[tuple[str, str]] = set()
        # on_error зовут потоки источников; без замка одну ошибку можно напечатать дважды.
        self._lock = threading.Lock()

    def record_until_interrupted(self, on_tick: Callable[[float], None] | None = None) -> float:
        me_track = ClockedTrack(self._me, 1, self._clock)
        them_track = ClockedTrack(self._them, len(self._apps), self._clock)
        sources: list[tuple[str, str, AudioSource, Callable[[Block], None]]] = [
            (
                "микрофон",
                "Дорожка «Я» дальше пишется тишиной.",
                self._mic,
                partial(me_track.push, 0),
            )
        ]
        for i, app in enumerate(self._apps):
            sources.append(
                (
                    f"приложение {i + 1}",
                    "Его звук в дорожке «Собеседники» дальше заменяется тишиной.",
                    app,
                    partial(them_track.push, i),
                )
            )
        self._repeats = {}
        self._seen = set()

        started = self._clock()
        try:
            with ExitStack() as stack:
                # Обратный порядок: сначала стоп источников, затем добивка дорожек, затем закрытие
                # файлов. Каждый шаг выполняется, даже если предыдущий бросил: заголовок WAV
                # должен быть валиден в любом исходе.
                stack.callback(self._report_repeats)
                stack.callback(self._me.close)
                stack.callback(self._them.close)
                stack.callback(self._finish, "Я", me_track)
                stack.callback(self._finish, "Собеседники", them_track)
                stack.callback(self._stop_all, [(label, source) for label, _, source, _ in sources])

                me_track.begin()
                them_track.begin()
                try:
                    for label, consequence, source, on_block in sources:
                        self._start(label, consequence, source, on_block)
                    while True:
                        time.sleep(_TICK_SECONDS)
                        me_track.tick()
                        them_track.tick()
                        if on_tick is not None:
                            on_tick(self._clock() - started)
                except KeyboardInterrupt:
                    pass
        except OSError as exc:
            # Диск мог переполниться и в цикле, и при добивке или закрытии: ловим всё вместе.
            raise MeetingNotesError(f"Не удалось записать дорожку: {exc}") from exc
        # Длительность считается после остановки и добивки: она совпадает с длиной файлов
        # с точностью до одного блока.
        return self._clock() - started

    def _start(
        self, label: str, consequence: str, source: AudioSource, on_block: Callable[[Block], None]
    ) -> None:
        def on_error(exc: Exception) -> None:
            if isinstance(exc, SourceWarning):
                self._warn_once(label, exc)
            else:
                # Конец источника всегда показываем сразу: за прежними мелкими предупреждениями
                # его легко пропустить, а дорожка дальше молча пишет тишину.
                self._on_warning(f"Источник «{label}» перестал записывать: {exc}. {consequence}")

        try:
            source.start(on_block, on_error)
        except Exception as exc:
            self._on_warning(f"Источник «{label}» не запустился: {exc}. {consequence}")

    def _warn_once(self, label: str, exc: Exception) -> None:
        # Источник может сыпать одним и тем же предупреждением на каждый пакет.
        with self._lock:
            key = (label, str(exc))
            repeated = key in self._seen
            self._seen.add(key)
            if repeated:
                self._repeats[label] = self._repeats.get(label, 0) + 1
        if not repeated:
            self._on_warning(f"Источник «{label}»: {exc}. Звук мог пропасть.")

    def _report_repeats(self) -> None:
        for label, count in self._repeats.items():
            self._on_warning(f"Источник «{label}»: ещё {count} ошибок после первой.")

    def _finish(self, name: str, track: ClockedTrack) -> None:
        try:
            lost = sum(track.finish()) / SOURCE_RATE
        except OSError as exc:
            raise MeetingNotesError(f"Не удалось дописать дорожку «{name}»: {exc}") from exc
        if lost >= _MIN_REPORTED_LOSS:
            self._on_warning(f"Дорожка «{name}»: потеряно {lost:.1f} с звука.")

    def _stop_all(self, sources: list[tuple[str, AudioSource]]) -> None:
        for label, source in sources:
            # Сбой остановки одного источника не должен оставить остальные работать.
            try:
                source.stop()
            except Exception as exc:
                self._on_warning(f"Источник «{label}» не остановился: {exc}.")
