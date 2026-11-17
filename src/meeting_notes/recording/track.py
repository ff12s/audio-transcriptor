import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt
from scipy.signal import firwin, lfilter

from meeting_notes.errors import OutputExistsError

SOURCE_RATE = 48000
TRACK_RATE = 16000

_DECIMATION = SOURCE_RATE // TRACK_RATE
_FIR_TAPS = 63
# Нули добавляются кусками, чтобы после сна системы не выделять сотни мегабайт за раз.
_MAX_FILL_FRAMES = SOURCE_RATE * 10

Samples = npt.NDArray[np.float32]

_ZERO: Samples = np.zeros(1, dtype=np.float32)


def _to_mono(block: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return block.mean(axis=1) if block.ndim == 2 else block


class TrackWriter:
    """Пишет моно WAV 16 kHz int16 по мере поступления блоков 48 kHz."""

    def __init__(self, path: Path) -> None:
        try:
            self._file = path.open("xb")
        except FileExistsError:
            raise OutputExistsError(f"Файл уже существует: {path}.") from None
        # Владелец файла - этот объект: wave не закрывает переданный ему файл.
        self._wav = wave.open(self._file, "wb")  # noqa: SIM115
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(TRACK_RATE)
        self.frames_written = 0
        self._closed = False
        self._fir = firwin(_FIR_TAPS, 1 / _DECIMATION)
        # Состояние фильтра и фаза прореживания переходят между блоками: иначе на стыках щелчки,
        # а сумма длин блоков расходится с len // 3.
        self._zi = np.zeros(_FIR_TAPS - 1)
        self._pos = 0

    def write(self, block: npt.NDArray[np.floating]) -> None:
        mono = _to_mono(block)
        if mono.size == 0:
            return
        filtered, self._zi = lfilter(self._fir, 1.0, mono, zi=self._zi)
        decimated = filtered[(-self._pos) % _DECIMATION :: _DECIMATION]
        self._pos += mono.size
        pcm = np.rint(np.clip(decimated, -1.0, 1.0) * 32767).astype("<i2")
        self._wav.writeframes(pcm.tobytes())
        self.frames_written += pcm.size

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._wav.close()  # дописывает размер данных в заголовок
        finally:
            self._file.close()

    def __enter__(self) -> "TrackWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class ClockedTrack:
    """Сводит входы в одну дорожку по настенным часам.

    Источник может молчать или стартовать с задержкой, поэтому длину дорожки задаёт время,
    а не число пришедших кадров, а у каждого входа своё место на шкале времени: блок
    ставится туда, где он был записан, а нехватка добивается нулями.
    """

    def __init__(
        self,
        writer: TrackWriter,
        n_inputs: int,
        clock: Callable[[], float] = time.monotonic,
        latency: float = 0.5,
    ) -> None:
        self._writer = writer
        self._clock = clock
        self._latency = latency
        # Повторно входимый: push зовёт begin, уже держа блокировку.
        self._lock = threading.RLock()
        # Порядок записи в файл: извлечение куска и write идут под этой блокировкой, а не под
        # _lock, чтобы push не ждал запись на диск.
        self._flush_lock = threading.Lock()
        self._chunks: list[deque[Samples]] = [deque() for _ in range(n_inputs)]
        # Конец поставленного на шкалу звука входа (кадры от начала записи, после обрезки и
        # добивки нулями); None, пока ничего не поставлено.
        self._end: list[int | None] = [None] * n_inputs
        self._dropped = [0] * n_inputs
        self._start: float | None = None
        self._emitted = 0

    def begin(self) -> None:
        with self._lock:
            if self._start is None:
                self._start = self._clock()

    def push(self, idx: int, block: npt.NDArray[np.floating]) -> None:
        mono = _to_mono(block).astype(np.float32)
        with self._lock:
            self.begin()
            assert self._start is not None
            arrival = round((self._clock() - self._start) * SOURCE_RATE)
            # Блок целиком получен к моменту прихода, значит записан до него и ни кадром позже.
            # Блок длиннее прошедшего времени ставится от начала записи: иначе он весь выпал бы.
            start = max(arrival - mono.size, 0)
            end = self._end[idx]
            queued = end is not None and end >= self._emitted
            if end is not None and queued:
                # Дрожание доставки: опоздание начала в пределах задержки не рвёт звук нулями.
                pos, tolerance = end, round(self._latency * SOURCE_RATE)
            else:
                # Своё место входа уже заполнено нулями (или вход ещё молчал): положение точное.
                pos, tolerance = self._emitted, 0
            delta = start - pos
            chunks = self._chunks[idx]
            if delta > tolerance:
                # Вид с нулевым шагом: пауза после сна системы не выделяет память.
                chunks.append(np.broadcast_to(_ZERO, (delta,)))
                pos = start
            elif delta < 0 and not (queued and arrival > pos):
                # Начало блока лежит там, где звук уже записан, либо поставлен впрок с концом не
                # раньше прихода (пачка блоков с одной отметкой после зависания): голову режем,
                # иначе блок кончался бы позже прихода и вход навсегда отстал бы. Перекрытие
                # только с поставленным звуком при приходе позже его конца - дрожание: не режем.
                cut = min(-delta, mono.size)
                self._dropped[idx] += cut
                mono = mono[cut:]
            if mono.size:
                chunks.append(mono)
                self._end[idx] = pos + mono.size

    def tick(self) -> None:
        self._flush(self._latency)

    def finish(self) -> list[int]:
        """Дописывает дорожку до текущего времени; возвращает по входам число кадров 48 kHz,
        которые в неё не попали: перекрытые начала опоздавших блоков и звук, оставшийся после
        текущего времени (блок длиннее прошедшего с begin() времени, сбой часов)."""
        self._flush(0.0)
        with self._lock:
            dropped = [
                lost + sum(chunk.size for chunk in chunks)
                for lost, chunks in zip(self._dropped, self._chunks, strict=True)
            ]
            for chunks in self._chunks:
                chunks.clear()
            self._dropped = [0] * len(self._dropped)
        return dropped

    def _flush(self, latency: float) -> None:
        self.begin()
        assert self._start is not None
        # Задержка даёт источникам время доставить блок, прежде чем его место заполнится нулями.
        # Округление то же, что в push: иначе кадр блока, пришедшего прямо перед finish, не влезет.
        target = round((self._clock() - self._start - latency) * SOURCE_RATE)
        with self._flush_lock:
            while True:
                with self._lock:
                    need = min(target - self._emitted, _MAX_FILL_FRAMES)
                    if need <= 0:
                        return
                    mixed = np.zeros(need, dtype=np.float32)
                    for chunks in self._chunks:
                        mixed += self._take(chunks, need)
                    # Вместе с извлечением: push сверяет положение блока с этим счётчиком.
                    self._emitted += need
                self._writer.write(mixed)

    @staticmethod
    def _take(chunks: deque[Samples], n: int) -> Samples:
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        while chunks and filled < n:
            head = chunks[0]
            part = head[: n - filled]
            out[filled : filled + part.size] = part
            filled += part.size
            if part.size == head.size:
                chunks.popleft()
            else:
                chunks[0] = head[part.size :]
        return out
