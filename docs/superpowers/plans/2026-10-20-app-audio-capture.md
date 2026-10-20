# App Audio Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uv run record --app chrome D:\records\standup` пишет звук одного приложения и микрофон в две WAV-дорожки до Ctrl+C, затем транскрибирует их в `standup.md` с метками `Я` / `Собеседники`.

**Architecture:** Источники звука (proc-tap на каждый PID, soundcard для микрофона) кладут float32-блоки в `ClockedTrack`. Он раз в тик сводит входы по настенным часам (нули на месте пропусков) и пишет через `TrackWriter` (моно, 48k→16k, int16, инкрементально). После остановки `CallTranscriptionPipeline` прогоняет обе дорожки через существующий `Transcriber`, помечает сегменты говорящим, перемежает по времени и рендерит существующим `TranscriptRenderer`.

**Tech Stack:** Python 3.12, uv, typer, faster-whisper 1.2.1, numpy 2.5.4, новые: proc-tap 1.1.1, soundcard 0.4.6, psutil, scipy.

**Spec:** `docs/superpowers/specs/2026-10-20-app-audio-capture-design.md`. Читать целиком, раздел «Проверенные факты» — обязательные API-факты.

## Global Constraints

- Windows build ≥ 20348 (на практике Windows 11); иначе ошибка до записи.
- Новые зависимости только: `proc-tap`, `soundcard`, `psutil`, `scipy` (все `; sys_platform == 'win32'`) и `types-psutil` в dev. Ничего больше.
- mypy strict: `soundcard.*`, `proctap.*`, `scipy.*` в `ignore_missing_imports`.
- Файлы: `<база>.me.wav`, `<база>.them.wav`, `<база>.md`. Дорожки: WAV, моно, 16000 Hz, int16.
- Метки: `Я` (микрофон), `Собеседники` (приложение). Строка абзаца: `[mm:ss] **Я:** текст`.
- Команда `transcribe` и формат её `.md` не меняются (существующие тесты не правятся).
- Сообщения пользователю — по-русски, через `MeetingNotesError`, без стектрейса.
- Комментарии в коде — по-русски, только «почему», в стиле репо.
- proc-tap: API — см. «Результат Task 0» (не README PyPI); 48 kHz, 2 канала, float32 в `bytes` → `np.frombuffer(pcm, np.float32).reshape(-1, 2)`. Пример WAV из README proc-tap не копировать.
- soundcard: не записывать один канал (мусор на WASAPI); писать все каналы, сводить в моно самим.

## Review Focus

1. Ctrl+C посреди записи → обе WAV закрыты с валидным заголовком, транскрипция стартует; при падении транскрипции дорожки на месте, в сообщении команда перезапуска. → тест в Task 6.
2. Приложение молчит / пакеты не приходят N секунд → `.them.wav` по длине равна настенному времени. → тест в Task 3.
3. Нет процесса с именем `--app` или есть `.me.wav`/`.them.wav`/`.md` без `--overwrite` → ошибка до старта записи, никакие файлы не созданы. → тесты в Task 1 и Task 6.
4. Микрофона нет (soundcard бросает при `default_microphone()`) → понятная ошибка до записи, а не стектрейс. → тест в Task 6.
5. Одна из дорожек пустая (весь созвон молчал) → `.md` пишется только с репликами другой; ноль сегментов в обеих → `.md` с frontmatter и пустым телом, без падения. → тест в Task 5.

---

### Task 0: Проверка proc-tap на реальном Chrome (вне продукта, выполняет оркестратор + пользователь)

Не входит в Review Loop. Скрипт в scratchpad, проект не трогает.

- [ ] **Step 1:** Скрипт `spike_proctap.py` в scratchpad: через psutil находит корни `chrome.exe` (родитель не `chrome.exe`), 20 с пишет `ProcTap(root_pid, StreamConfig(), on_data=...)` в WAV 48k float→int16; логирует по секундам, сколько кадров пришло. Запуск: `uv run --no-project --with proc-tap --with psutil --with numpy python spike_proctap.py`.
- [ ] **Step 2:** Пользователь: видео в Chrome, музыка в другом приложении, на 8–13 с пауза видео.
- [ ] **Step 3:** Критерии: звук Chrome есть, музыки нет; лог показывает, приходят ли пакеты в паузе. Если на корне тишина — повторить с ProcTap на каждый PID дерева (fallback из спеки). Не прошло и это — СТОП, к пользователю.
- [ ] **Step 4:** Записать вывод в этот план (раздел «Результат Task 0»): include-tree работает? пакеты в паузе идут? Это решает, сколько PID берёт `find_capture_pids` в Task 1.

**Результат Task 0 (2026-10-20, Windows build 26200, живой созвон в Chrome + музыка в другом приложении):**
- **API proc-tap 1.1.1 отличается от README на PyPI.** Экспортируется `from proctap import ProcessAudioCapture`
  (класса `ProcTap` и `StreamConfig` нет). Конструктор: `ProcessAudioCapture(pid: int, on_data: Callable[[bytes, int], None] | None = None, resample_quality='best')`.
  Методы: `start()`, `stop()`, `close()`, `read(timeout=1.0) -> bytes | None`, а также контекстный менеджер. Колбэк вызывается как `on_data(data, -1)`, второй аргумент **не** число кадров.
  Данные: 48 kHz, 2 канала, float32.
- Нативный код использует `PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE`. Захват **корня** `chrome.exe` (1 корень из 26 процессов) дал голоса собеседников, а музыки из другого приложения не было (проверено на слух).
  **`find_capture_pids` возвращает корни** (`find_app_roots`); fallback «PID на каждый процесс дерева» не нужен, и `tree_pids` из Task 1 убрать.
- Пакеты шли непрерывно, 48000 кадров/с во всех 20 секундах, в том числе в паузах между репликами. Задержка старта около 1.3 с.
  Добивка нулей по настенным часам в `ClockedTrack` остаётся: она страхует от выхода приложения и задержки старта.

---

### Task 1: Зависимости, проверка ОС и выбор процессов

**Files:**
- Modify: `pyproject.toml` (зависимости, mypy overrides, `[project.scripts] record = "meeting_notes.cli:record_app"` — сам `record_app` появится в Task 6; до него скрипт не вызывать)
- Create: `src/meeting_notes/processes.py`
- Test: `tests/test_processes.py`

**Interfaces:**
- Produces:
  - `ProcInfo = NamedTuple("ProcInfo", [("pid", int), ("ppid", int), ("name", str)])`
  - `find_app_roots(app: str, procs: Iterable[ProcInfo]) -> list[int]` — PID корней деревьев: имя совпадает (без регистра, `.exe` необязателен), а родитель не процесс с тем же именем. Сортировка по pid.
  - `find_capture_pids(app: str) -> list[int]` — psutil-снимок → корни (include-tree подтверждён в Task 0). Пусто → `MeetingNotesError("Процесс {app!r} не найден. Запустите приложение и повторите.")`.
  - `check_windows_build(build: int) -> None` — `< 20348` → `MeetingNotesError` с текстом про Windows 11.
  - `current_windows_build() -> int` — `sys.getwindowsversion().build`; вне Windows бросает `MeetingNotesError("Запись поддерживается только на Windows 11.")`.

- [ ] **Step 1: тесты**

```python
P = ProcInfo
PROCS = [P(1, 0, "explorer.exe"), P(10, 1, "chrome.exe"), P(11, 10, "chrome.exe"),
         P(12, 11, "chrome.exe"), P(20, 1, "Chrome.exe"), P(30, 1, "Zoom.exe")]

def test_roots_by_name_case_insensitive(): assert find_app_roots("chrome", PROCS) == [10, 20]
def test_exe_suffix_optional(): assert find_app_roots("zoom.exe", PROCS) == [30]
def test_no_match(): assert find_app_roots("teams", PROCS) == []
def test_root_whose_parent_is_other_app(): assert find_app_roots("chrome", [P(5, 4, "chrome.exe"), P(4, 1, "launcher.exe")]) == [5]
def test_old_build_rejected(): with pytest.raises(MeetingNotesError, match="Windows 11"): check_windows_build(19045)
def test_new_build_ok(): check_windows_build(22631)
def test_find_capture_pids_missing(monkeypatch): # подменить снимок процессов пустым → MeetingNotesError с "не найден"
```

- [ ] **Step 2:** `uv run pytest tests/test_processes.py -v` → FAIL (нет модуля).
- [ ] **Step 3:** Реализация. Снимок через `psutil.process_iter(["pid", "ppid", "name"])`; исчезнувшие процессы (`psutil.NoSuchProcess`, `AccessDenied`) пропускать. Модульная функция `_snapshot() -> list[ProcInfo]` — точка подмены в тестах.
- [ ] **Step 4:** `uv sync` и `uv run pytest -q` → всё зелёное; `uv run mypy src` и `uv run ruff check .` чистые.
- [ ] **Step 5:** commit `feat: add process lookup and Windows build check for app capture`.

---

### Task 2: Метки говорящего в модели, склейке и рендере

**Files:**
- Modify: `src/meeting_notes/models.py`, `src/meeting_notes/paragraphs.py`, `src/meeting_notes/renderer.py`
- Test: `tests/test_paragraphs.py`, `tests/test_renderer.py` (только новые тесты)

**Interfaces:**
- Produces:
  - `Segment.speaker: str | None = None`
  - `SPEAKER_ME = "Я"`, `SPEAKER_THEM = "Собеседники"` в `models.py`
  - `interleave(*tracks: list[Segment]) -> list[Segment]` в `paragraphs.py` — все сегменты, стабильная сортировка по `start`.
  - `merge_segments` не склеивает сегменты с разным `speaker`.
  - Рендер: при `speaker` не None строка `[{tc}] **{speaker}:** {text}`; при None — как сейчас.

- [ ] **Step 1: тесты**

```python
def test_merge_does_not_cross_speakers():
    segs = [Segment(start=0, end=3, text="а", speaker="Я"), Segment(start=3, end=6, text="б", speaker="Собеседники")]
    assert [s.text for s in merge_segments(segs)] == ["а", "б"]

def test_merge_same_speaker_still_merges():
    segs = [Segment(start=0, end=3, text="а", speaker="Я"), Segment(start=3, end=6, text="б", speaker="Я")]
    assert [(s.text, s.speaker) for s in merge_segments(segs)] == [("а б", "Я")]

def test_interleave_orders_by_start():
    me = [Segment(start=5, end=6, text="m", speaker="Я")]
    them = [Segment(start=1, end=2, text="t1", speaker="Собеседники"), Segment(start=7, end=8, text="t2", speaker="Собеседники")]
    assert [s.text for s in interleave(me, them)] == ["t1", "m", "t2"]

def test_render_with_speaker():  # сегмент start=72, speaker="Я", text="Начнём" → "[01:12] **Я:** Начнём" в теле
```

- [ ] **Step 2:** `uv run pytest tests/test_paragraphs.py tests/test_renderer.py -v` → новые FAIL.
- [ ] **Step 3:** Реализация (merged-сегмент наследует `speaker`).
- [ ] **Step 4:** `uv run pytest -q` → всё зелёное, старые тесты не тронуты.
- [ ] **Step 5:** commit `feat: label transcript paragraphs by speaker`.

---

### Task 3: Запись дорожки — TrackWriter и ClockedTrack

**Files:**
- Create: `src/meeting_notes/track.py`
- Test: `tests/test_track.py`

**Interfaces:**
- Produces:
  - `SOURCE_RATE = 48000`, `TRACK_RATE = 16000`
  - `class TrackWriter` — контекстный менеджер. `__init__(self, path: Path)` открывает файл в режиме `"xb"` (существующий → `OutputExistsError`). `write(self, block: np.ndarray) -> None` принимает float32 `(frames, channels)` на 48k: усредняет каналы, ресемплит 3:1, клипует, пишет int16. `frames_written: int` (кадры 16k). `close()` закрывает WAV и фиксирует заголовок.
  - Ресемплер со состоянием: FIR `scipy.signal.firwin(63, 1/3)`, `scipy.signal.lfilter(..., zi=...)`, затем каждый 3-й отсчёт с учётом фазы между блоками. Иначе на стыках блоков щелчки, а сумма длин не равна `len // 3`.
  - `class ClockedTrack` — `__init__(self, writer: TrackWriter, n_inputs: int, clock: Callable[[], float] = time.monotonic, latency: float = 0.5)`. `push(self, idx: int, block: np.ndarray) -> None` потокобезопасен (Lock); блоки входа копятся в его буфере. `tick(self) -> None`: целевая длина = `(clock() - start - latency) * SOURCE_RATE` кадров. Для каждого входа берётся до `need` кадров из буфера, недостаток добивается нулями, входы суммируются, результат идёт в `writer.write`. `finish(self) -> None` вызывает `tick` без задержки (`latency=0`) и сбрасывает остатки буферов.
  - `start` = момент первого `tick` либо явный `begin()`. `begin()` вызывается у обеих дорожек одним моментом.

- [ ] **Step 1: тесты**

```python
def test_one_second_becomes_16000_frames(tmp_path):  # write 48000 нулей stereo → frames_written == 16000; wave.open: nchannels 1, framerate 16000, sampwidth 2
def test_split_blocks_equal_single_block(tmp_path):   # синус 1 kHz, 48000 кадров одним блоком vs блоками по 1000 → байты данных совпадают (np.allclose ±1 LSB)
def test_tone_kept_high_freq_cut(tmp_path):           # 1 kHz амплитуда 0.5 → RMS на выходе ≈ 0.35 ±10%; 10 kHz → RMS < 0.05
def test_existing_file_refused(tmp_path):             # путь существует → OutputExistsError, файл не изменён
def test_gap_filled_by_wall_clock(tmp_path):          # fake clock: begin() t=0; push 1 с; clock=5.5; tick(); finish() → frames_written == 5.5*16000 ±160
def test_two_inputs_summed(tmp_path):                 # n_inputs=2, вход 0 = 0.25, вход 1 = 0.25 постоянные → на выходе ≈ 0.5
def test_clipped_not_wrapped(tmp_path):               # сумма 1.5 → int16 32767, не отрицательные
```

- [ ] **Step 2:** `uv run pytest tests/test_track.py -v` → FAIL.
- [ ] **Step 3:** Реализация (stdlib `wave` с `open(path, "xb")` → `wave.open(f, "wb")`; nframes в заголовке wave обновит при `close`).
- [ ] **Step 4:** `uv run pytest -q`, mypy, ruff → чисто.
- [ ] **Step 5:** commit `feat: add wall-clock track writer with 48k to 16k resampling`.

---

### Task 4: Источники звука и рекордер

**Files:**
- Create: `src/meeting_notes/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Consumes: `ClockedTrack`, `TrackWriter` (Task 3), `find_capture_pids` (Task 1).
- Produces:
  - `class AudioSource(Protocol)`: `start(self, on_block: Callable[[np.ndarray], None]) -> None`; `stop(self) -> None`.
  - `class AppSource` — `__init__(self, pid: int)`; один `proctap.ProcessAudioCapture(pid, on_data=...)` (колбэк `(data: bytes, _: int)`); `bytes` → `float32 (-1, 2)`. Ошибки proc-tap на старте → `MeetingNotesError`.
  - `class MicSource` — `soundcard.default_microphone()` (нет устройства или ошибка → `MeetingNotesError("Микрофон не найден...")`). Свой поток: `recorder(samplerate=48000, blocksize=4800)` и цикл `record(numframes=4800)` до `stop()`. Поток инициализирует COM сам (soundcard делает это при импорте в потоке — проверить; при `RPC_E_CHANGED_MODE` не падать).
  - `class CallRecorder` — `__init__(self, me: TrackWriter, them: TrackWriter, mic: AudioSource, apps: list[AudioSource], clock=time.monotonic)`. `record_until_interrupted(self, on_tick: Callable[[float], None] | None = None) -> float`: создаёт `ClockedTrack` (me: 1 вход, them: `len(apps)` входов), `begin()` обоим, стартует источники, тикает раз в 0.1 с до `KeyboardInterrupt`. Затем в `finally` останавливает все источники, `finish()` обоих, закрывает writers. Возвращает длительность в секундах. Ошибка любого источника, пойманная в его потоке, не роняет запись: предупреждение через колбэк `on_warning(str)`, дорожка добивается нулями (так покрывается «приложение закрылось»).

- [ ] **Step 1: тесты (фейковые источники, без реального звука)**

```python
def test_interrupt_closes_both_tracks(tmp_path):     # FakeSource шлёт блоки; on_tick на 3-м тике бросает KeyboardInterrupt → оба wav валидны (wave.open читает), stop() вызван у всех
def test_app_failure_keeps_recording(tmp_path):      # AppSource-фейк бросает в своём потоке → on_warning вызван, .them.wav по длине ≈ .me.wav
def test_bytes_to_frames():                          # helper _pcm_to_block(np.ones(8, float32).tobytes()) → shape (4, 2)
```

- [ ] **Step 2:** FAIL → **Step 3:** реализация → **Step 4:** `uv run pytest -q`, mypy, ruff чисто.
- [ ] **Step 5:** commit `feat: add app and mic capture sources with call recorder`.

---

### Task 5: Пайплайн транскрипции пары дорожек

**Files:**
- Modify: `src/meeting_notes/pipeline.py`
- Test: `tests/test_pipeline.py` (только новые тесты)

**Interfaces:**
- Consumes: `interleave`, `SPEAKER_ME`, `SPEAKER_THEM` (Task 2); `Transcriber` (как есть).
- Produces:
  - `call_paths(base: Path) -> tuple[Path, Path, Path]` → `(me_wav, them_wav, md)`.
  - `check_call_outputs(base: Path, overwrite: bool, recording: bool) -> None`: `.md` существует и нет overwrite → `OutputExistsError`. При `recording=True` существующий `.me.wav` или `.them.wav` → `OutputExistsError` с текстом «дорожки уже есть, используйте --transcribe-only или другое имя» (`--overwrite` дорожки не трогает).
  - `class CallTranscriptionPipeline` — те же аргументы конструктора, что у `TranscriptionPipeline`. `run(self, base: Path, overrides: MetaOverrides, overwrite: bool) -> Path`: транскрибирует обе дорожки одним transcriber, проставляет speaker, вызывает `interleave`. Meta: `title`/`source` = `base.name`, `duration` = max, `date` берётся из `me`. Применяет overrides, рендерит, пишет; при сбое записи используется тот же fallback, что у `TranscriptionPipeline` (вынести `_save_fallback` в модульную функцию, поведение не менять).

- [ ] **Step 1: тесты** (FakeTranscriber из conftest, сегменты на путь)

```python
def test_call_md_labels_and_order(tmp_path):     # me: [5..6 "м"], them: [1..2 "т"] → тело "[00:01] **Собеседники:** т" перед "[00:05] **Я:** м"; source: "standup"; оба пути переданы transcriber
def test_call_one_track_empty(tmp_path):         # me пусто → только реплики собеседников, без ошибки
def test_call_both_empty(tmp_path):              # обе пусты → md = frontmatter + пустое тело, exit без исключения
def test_check_outputs_refuses_existing_wav(tmp_path):  # .me.wav есть, recording=True → OutputExistsError
def test_check_outputs_md_overwrite(tmp_path):   # .md есть, overwrite=True → ok
```

- [ ] **Step 2:** FAIL → **Step 3:** реализация → **Step 4:** `uv run pytest -q` (старые тесты пайплайна зелёные), mypy, ruff.
- [ ] **Step 5:** commit `feat: transcribe mic and app tracks into a speaker-labeled transcript`.

---

### Task 6: CLI `record`, README

**Files:**
- Modify: `src/meeting_notes/cli.py`, `README.md`
- Test: `tests/test_cli_record.py`

**Interfaces:**
- Consumes: всё выше.
- Produces:
  - `record_app = typer.Typer(add_completion=False)`, команда `record(base: Path, app: str | None = --app, title, date, participants, model, overwrite, transcribe_only: bool = --transcribe-only)`.
  - Точки подмены для тестов (по образцу `build_transcriber`): `build_recorder(me_path, them_path, pids, on_warning) -> CallRecorder` и `windows_build() -> int`.
  - Порядок: проверка сборки → `check_call_outputs` → (если не `--transcribe-only`) `--app` обязателен, `find_capture_pids` → печать «Запись… Ctrl+C — стоп» и тикающего времени → `record_until_interrupted` → транскрипция (общий код прогресса и глоссария вынести из `main` в helper, не дублировать) → «Готово: <md>».
  - Ошибка транскрипции после записи: к тексту `MeetingNotesError` добавить строку `Дорожки сохранены: <me>, <them>. Повторить: uv run record --transcribe-only "<base>"`.
  - `--transcribe-only` без существующих дорожек → `MeetingNotesError`.

- [ ] **Step 1: тесты** (CliRunner, фейковые recorder и transcriber, `windows_build` → 22631)

```python
def test_record_then_transcribe(tmp_path):        # фейк recorder создаёт обе wav → exit 0, .md с "**Я:**"
def test_missing_app_flag(tmp_path):              # без --app и без --transcribe-only → exit 1, "--app"
def test_app_not_found_creates_nothing(tmp_path): # find_capture_pids бросает → exit 1, в tmp_path нет файлов, recorder не вызывался
def test_existing_md_before_recording(tmp_path):  # .md есть → exit 1, recorder не вызывался
def test_old_windows(tmp_path):                   # windows_build → 19045 → exit 1, "Windows 11"
def test_mic_missing_message(tmp_path):           # build_recorder бросает MeetingNotesError("Микрофон не найден") → exit 1, без Traceback
def test_transcription_failure_keeps_tracks(tmp_path):  # transcriber бросает MeetingNotesError → exit 1, вывод содержит "--transcribe-only" и оба пути, wav на месте
def test_transcribe_only(tmp_path):               # wav уже есть, --transcribe-only → recorder не вызывался, .md создан
```

- [ ] **Step 2:** FAIL → **Step 3:** реализация. README: раздел «Запись созвона». В нём: команда, Windows 11, наушники, `--transcribe-only`, что в запись попадает весь звук приложения (для Chrome — все вкладки).
- [ ] **Step 4:** `uv run pytest -q`, `uv run mypy src`, `uv run ruff check . ; uv run ruff format --check .` → чисто.
- [ ] **Step 5:** commit `feat: add record command for app + mic call capture`.

---

### Task 7: Ручная приёмка (оркестратор + пользователь)

- [ ] Пользователь: 2–3 минуты реального созвона в Chrome в наушниках, параллельно музыка в другом приложении → `uv run record --app chrome D:\records\test-call`, Ctrl+C.
- [ ] Ожидаем: в `.them.wav` нет музыки; в `.md` есть реплики `Я` и `Собеседники` в правильном порядке; длительности обеих WAV ≈ времени записи.
