import pytest

from meeting_notes.errors import MeetingNotesError
from meeting_notes.recording import processes
from meeting_notes.recording.processes import (
    MIN_WINDOWS_BUILD,
    ProcInfo,
    check_windows_build,
    find_app_roots,
    find_capture_pids,
)

P = ProcInfo
PROCS = [
    P(1, 0, "explorer.exe"),
    P(10, 1, "chrome.exe"),
    P(11, 10, "chrome.exe"),
    P(12, 11, "chrome.exe"),
    P(20, 1, "Chrome.exe"),
    P(30, 1, "Zoom.exe"),
]


def test_roots_by_name_case_insensitive() -> None:
    assert find_app_roots("chrome", PROCS) == [10, 20]


def test_exe_suffix_optional() -> None:
    assert find_app_roots("zoom.exe", PROCS) == [30]


def test_no_match() -> None:
    assert find_app_roots("teams", PROCS) == []


def test_root_whose_parent_is_other_app() -> None:
    procs = [P(5, 4, "chrome.exe"), P(4, 1, "launcher.exe")]
    assert find_app_roots("chrome", procs) == [5]


def test_roots_sorted_by_pid() -> None:
    procs = [P(9, 1, "chrome.exe"), P(3, 1, "chrome.exe")]
    assert find_app_roots("chrome", procs) == [3, 9]


def test_old_build_rejected() -> None:
    with pytest.raises(MeetingNotesError, match="Windows 11"):
        check_windows_build(19045)


def test_new_build_ok() -> None:
    check_windows_build(22631)


def test_old_build_message_names_required_build() -> None:
    with pytest.raises(MeetingNotesError, match=rf"{MIN_WINDOWS_BUILD} или новее.*19045"):
        check_windows_build(19045)


def test_build_at_threshold_ok() -> None:
    check_windows_build(MIN_WINDOWS_BUILD)


class FakeProc:
    def __init__(self, **info: object) -> None:
        self.info = info


def test_snapshot_handles_none_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    psutil = pytest.importorskip("psutil")
    # process_iter при отказе доступа кладёт в info None, а не бросает исключение.
    procs = [
        FakeProc(pid=1, ppid=None, name="System"),
        FakeProc(pid=2, ppid=1, name=None),
        FakeProc(pid=3, ppid=1, name="chrome.exe"),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: iter(procs))
    assert processes._snapshot() == [P(1, 0, "System"), P(3, 1, "chrome.exe")]


def test_find_capture_pids_returns_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(processes, "_snapshot", lambda: PROCS)
    assert find_capture_pids("chrome") == [10, 20]


def test_find_capture_pids_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(processes, "_snapshot", lambda: [])
    with pytest.raises(MeetingNotesError, match="не найден"):
        find_capture_pids("chrome")
