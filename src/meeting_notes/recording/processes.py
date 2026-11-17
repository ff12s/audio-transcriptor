import sys
from collections.abc import Iterable
from typing import NamedTuple

from meeting_notes.errors import MeetingNotesError

# Process Loopback API появился в сборке 20348; ни один выпуск Windows 10 её не достиг.
MIN_WINDOWS_BUILD = 20348


class ProcInfo(NamedTuple):
    pid: int
    ppid: int
    name: str


def _norm(name: str) -> str:
    low = name.lower()
    return low.removesuffix(".exe")


def find_app_roots(app: str, procs: Iterable[ProcInfo]) -> list[int]:
    target = _norm(app)
    matched = [p for p in procs if _norm(p.name) == target]
    matched_pids = {p.pid for p in matched}
    # Захват включает дочерние процессы, поэтому берём только корни деревьев:
    # иначе звук одного дерева записался бы несколько раз.
    return sorted(p.pid for p in matched if p.ppid not in matched_pids)


def _snapshot() -> list[ProcInfo]:
    import psutil

    result: list[ProcInfo] = []
    for proc in psutil.process_iter(["pid", "ppid", "name"]):
        # При AccessDenied/ZombieProcess поле в info равно None, а исчезнувший процесс
        # process_iter пропускает сам.
        info = proc.info
        if info["name"] is None:
            continue
        result.append(ProcInfo(info["pid"], info["ppid"] or 0, info["name"]))
    return result


def find_capture_pids(app: str) -> list[int]:
    pids = find_app_roots(app, _snapshot())
    if not pids:
        raise MeetingNotesError(f"Процесс {app!r} не найден. Запустите приложение и повторите.")
    return pids


def check_windows_build(build: int) -> None:
    if build < MIN_WINDOWS_BUILD:
        raise MeetingNotesError(
            f"Запись звука приложения: нужна Windows 11 (сборка {MIN_WINDOWS_BUILD} или новее), "
            f"у вас сборка {build}."
        )


def current_windows_build() -> int:
    if sys.platform != "win32":
        raise MeetingNotesError("Запись поддерживается только на Windows 11.")
    return sys.getwindowsversion().build
