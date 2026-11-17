import wave
from collections import deque
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from meeting_notes.errors import OutputExistsError
from meeting_notes.recording.track import SOURCE_RATE, TRACK_RATE, ClockedTrack, TrackWriter


def sine(freq: float, seconds: float, amp: float = 0.5) -> npt.NDArray[np.float32]:
    t = np.arange(int(seconds * SOURCE_RATE)) / SOURCE_RATE
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def read_pcm(path: Path) -> npt.NDArray[np.int16]:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def rms(pcm: npt.NDArray[np.int16]) -> float:
    return float(np.sqrt(np.mean((pcm / 32767.0) ** 2)))


LATENCY = 0.5
# Начало метки определяется с точностью до миллисекунд (задержка фильтра ~0.7 мс), а сбой
# выравнивания даёт сдвиг не меньше задержки в 0.5 с: десятой доли задержки хватает с запасом.
TOL = LATENCY / 10
STEP = SOURCE_RATE // 10

Push = tuple[float, int, npt.NDArray[np.float32]]


def burst(total: float, start: float, length: float, amp: float) -> npt.NDArray[np.float32]:
    sig = np.zeros(round(total * SOURCE_RATE), dtype=np.float32)
    first = round(start * SOURCE_RATE)
    sig[first : first + round(length * SOURCE_RATE)] = amp
    return sig


def onset(pcm: npt.NDArray[np.int16], level: float) -> float:
    """Секунда, на которой дорожка впервые поднимается выше level."""
    above = np.abs(pcm) > level * 32767
    assert above.any(), "метка не попала в дорожку"
    return float(np.argmax(above)) / TRACK_RATE


def deliver(
    idx: int, sig: npt.NDArray[np.float32], first: int, last: int, at: float | None = None
) -> list[Push]:
    """Блоки по 0.1 с с номерами first..last; без `at` блок приходит в конце своего слота."""
    return [
        (k / 10 if at is None else at, idx, sig[(k - 1) * STEP : k * STEP])
        for k in range(first, last + 1)
    ]


def play(track: ClockedTrack, clock: "FakeClock", pushes: list[Push], until: float) -> list[int]:
    pending = deque(sorted(pushes, key=lambda p: p[0]))
    track.begin()
    for i in range(1, round(until * 10) + 1):
        t = i / 10
        while pending and pending[0][0] <= t + 1e-9:
            clock.now, idx, block = pending.popleft()
            track.push(idx, block)
        clock.now = t
        track.tick()
    return track.finish()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_one_second_becomes_16000_frames(tmp_path: Path) -> None:
    path = tmp_path / "t.wav"
    with TrackWriter(path) as w:
        w.write(np.zeros((SOURCE_RATE, 2), dtype=np.float32))
        assert w.frames_written == TRACK_RATE
    with wave.open(str(path), "rb") as r:
        assert (r.getnchannels(), r.getframerate(), r.getsampwidth()) == (1, TRACK_RATE, 2)
        assert r.getnframes() == TRACK_RATE


def test_split_blocks_equal_single_block(tmp_path: Path) -> None:
    data = sine(1000, 1.0)
    with TrackWriter(tmp_path / "one.wav") as w:
        w.write(data)
    # 1000 не кратно 3: проверяет перенос фазы прореживания между блоками.
    with TrackWriter(tmp_path / "split.wav") as w:
        for i in range(0, len(data), 1000):
            w.write(data[i : i + 1000])
        assert w.frames_written == TRACK_RATE
    one = read_pcm(tmp_path / "one.wav").astype(np.int32)
    split = read_pcm(tmp_path / "split.wav").astype(np.int32)
    assert np.allclose(one, split, atol=1)


def test_tone_kept_high_freq_cut(tmp_path: Path) -> None:
    with TrackWriter(tmp_path / "low.wav") as w:
        w.write(sine(1000, 1.0))
    with TrackWriter(tmp_path / "high.wav") as w:
        w.write(sine(10000, 1.0))
    # Пропускаем начало: там переходный процесс фильтра.
    low = rms(read_pcm(tmp_path / "low.wav")[200:])
    high = rms(read_pcm(tmp_path / "high.wav")[200:])
    assert low == pytest.approx(0.5 / np.sqrt(2), rel=0.1)
    assert high < 0.05


def test_existing_file_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.wav"
    path.write_bytes(b"keep")
    with pytest.raises(OutputExistsError):
        TrackWriter(path)
    assert path.read_bytes() == b"keep"


def test_gap_filled_by_wall_clock(tmp_path: Path) -> None:
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=1, clock=clock)
        track.begin()
        track.push(0, np.zeros((SOURCE_RATE, 2), dtype=np.float32))
        clock.now = 5.5
        track.tick()
        track.finish()
        assert abs(w.frames_written - 5.5 * TRACK_RATE) <= 160


def test_audio_then_silence_until_wall_clock(tmp_path: Path) -> None:
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=1, clock=clock)
        track.begin()
        track.push(0, np.full((SOURCE_RATE, 1), 0.5, dtype=np.float32))
        clock.now = 3.0
        track.tick()
        track.finish()
    pcm = read_pcm(tmp_path / "t.wav")
    assert rms(pcm[2000 : TRACK_RATE - 2000]) > 0.4
    assert rms(pcm[TRACK_RATE + 2000 :]) < 0.01


def test_two_inputs_summed(tmp_path: Path) -> None:
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=2, clock=clock)
        track.begin()
        for idx in (0, 1):
            track.push(idx, np.full((2 * SOURCE_RATE, 2), 0.25, dtype=np.float32))
        clock.now = 2.5
        track.tick()
        track.finish()
    pcm = read_pcm(tmp_path / "t.wav")
    assert np.mean(pcm[2000 : 2 * TRACK_RATE - 2000] / 32767.0) == pytest.approx(0.5, abs=0.02)


def test_clipped_not_wrapped(tmp_path: Path) -> None:
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=2, clock=clock)
        track.begin()
        track.push(0, np.full((SOURCE_RATE, 1), 1.0, dtype=np.float32))
        track.push(1, np.full((SOURCE_RATE, 1), 0.5, dtype=np.float32))
        clock.now = 1.5
        track.tick()
        track.finish()
    pcm = read_pcm(tmp_path / "t.wav")
    # Края исключены: ступеньку фильтр обводит выбросом в обе стороны.
    assert (pcm[2000 : TRACK_RATE - 2000] == 32767).all()


@pytest.mark.parametrize("lead", [0.0, 0.4])
def test_late_block_lands_at_wall_clock_position(tmp_path: Path, lead: float) -> None:
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=1, clock=clock, latency=LATENCY)
        track.begin()
        clock.now = 1.0
        track.tick()  # первые 0.5 с уже заполнены нулями
        # Блок покрывает [1.0 - длина, 1.0]: метка стоит в его последних 0.2 с, то есть на 0.8 с.
        # При lead=0.4 блок начинается раньше уже записанного места: голова отбрасывается.
        track.push(0, np.concatenate([burst(lead, 0, 0, 0), burst(0.2, 0, 0.2, 0.5)]))
        clock.now = 3.0
        track.finish()
    assert onset(read_pcm(tmp_path / "t.wav"), 0.25) == pytest.approx(0.8, abs=TOL)


def test_input_starting_late_keeps_wall_clock_position(tmp_path: Path) -> None:
    sig0 = burst(4.0, 2.0, 0.2, 0.25)
    sig1 = burst(4.0, 3.0, 0.2, 0.5)
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=2, clock=clock, latency=LATENCY)
        # Второй вход начинает отдавать звук только на 1.3 с (как proc-tap).
        pushes = deliver(0, sig0, 1, 40) + deliver(1, sig1, 13, 40)
        play(track, clock, pushes, until=4.0)
        frames = w.frames_written
    pcm = read_pcm(tmp_path / "t.wav")
    assert onset(pcm, 0.1) == pytest.approx(2.0, abs=TOL)
    assert onset(pcm, 0.4) == pytest.approx(3.0, abs=TOL)
    assert abs(frames - 4.0 * TRACK_RATE) <= 160


def test_stall_then_backlog_adds_no_permanent_delay(tmp_path: Path) -> None:
    sig = burst(8.0, 7.0, 0.2, 0.5)
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=1, clock=clock, latency=LATENCY)
        pushes = (
            deliver(0, sig, 1, 10)
            + deliver(0, sig, 11, 50, at=5.0)  # застряло на 4 с, выдано одной пачкой
            + deliver(0, sig, 51, 80)
        )
        play(track, clock, pushes, until=8.0)
        frames = w.frames_written
    assert abs(frames - 8.0 * TRACK_RATE) <= 160
    # Без выравнивания смещение равнялось бы длине пачки, с допуском только на опоздание - LATENCY.
    t = onset(read_pcm(tmp_path / "t.wav"), 0.25)
    assert 7.0 - TOL <= t <= 7.0 + LATENCY + TOL
    assert t == pytest.approx(7.0, abs=TOL)


def test_late_first_block_is_not_cut_by_later_ones(tmp_path: Path) -> None:
    sig = sine(1000, 3.0)[:, 0]
    # Первый блок опоздал сильнее следующих: перекрытие с поставленным звуком - не повод резать.
    jitter = [0.1, 0.05, 0.15, 0.3, 0.2, 0.1]
    pushes = [
        (k / 10 + jitter[(k - 1) % len(jitter)], 0, block)
        for k, (_, _, block) in enumerate(deliver(0, sig, 1, 30), start=1)
    ]
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=1, clock=clock, latency=LATENCY)
        assert play(track, clock, pushes, until=3.5) == [0]


def test_jittery_delivery_inserts_nothing(tmp_path: Path) -> None:
    sig = sine(1000, 3.0)[:, 0]
    with TrackWriter(tmp_path / "ref.wav") as w:
        w.write(sig)
    # Опоздание каждого блока меньше LATENCY, поэтому он приходит раньше, чем его место сброшено.
    jitter = [0.0, 0.05, 0.15, 0.3, 0.2, 0.1]
    pushes = [
        (k / 10 + jitter[(k - 1) % len(jitter)], 0, block)
        for k, (_, _, block) in enumerate(deliver(0, sig, 1, 30), start=1)
    ]
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=1, clock=clock, latency=LATENCY)
        dropped = play(track, clock, pushes, until=3.5)
    assert dropped == [0]
    ref = read_pcm(tmp_path / "ref.wav").astype(np.int32)
    out = read_pcm(tmp_path / "t.wav").astype(np.int32)
    assert np.allclose(out[: ref.size], ref, atol=1)
    assert rms(out[ref.size + 200 :].astype(np.int16)) < 0.01


def test_finish_reports_frames_past_wall_clock(tmp_path: Path) -> None:
    clock = FakeClock()
    block = np.full((3 * SOURCE_RATE, 1), 0.5, dtype=np.float32)
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=2, clock=clock, latency=LATENCY)
        track.begin()
        clock.now = 1.0
        track.push(0, block)
        track.push(1, block[:SOURCE_RATE])
        # Звук длиннее прошедшего времени: сверх 1.0 с в дорожку не попадает.
        assert track.finish() == [len(block) - round(clock.now * SOURCE_RATE), 0]


def test_finish_counts_backlog_trimmed_on_push(tmp_path: Path) -> None:
    sig = burst(8.0, 7.0, 0.2, 0.5)
    clock = FakeClock()
    with TrackWriter(tmp_path / "t.wav") as w:
        track = ClockedTrack(w, n_inputs=1, clock=clock, latency=LATENCY)
        pushes = deliver(0, sig, 1, 10) + deliver(0, sig, 11, 50, at=5.0) + deliver(0, sig, 51, 80)
        dropped = play(track, clock, pushes, until=8.0)
    # Из пачки сохранён только первый блок, остальные 39 блоков по 0.1 с отброшены.
    assert dropped == [39 * STEP]
