import os
import sys
import types
from pathlib import Path

import pytest

from meeting_notes.asr import cuda_setup
from meeting_notes.errors import CudaUnavailableError


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    # pytest ветвится по sys.platform при создании basetemp; создаём его до подмены платформы.
    tmp_path_factory.getbasetemp()
    monkeypatch.setattr(cuda_setup, "_prepared", False)
    monkeypatch.setattr(cuda_setup, "_added_dirs", set())
    monkeypatch.setattr(cuda_setup, "_dll_handles", [])
    # Не-Windows ветка: DLL-каталоги не трогаем, если тест сам не включит win32.
    monkeypatch.setattr(sys, "platform", "linux")


def _fake_ctranslate2(monkeypatch: pytest.MonkeyPatch, count: object) -> None:
    def get_cuda_device_count() -> int:
        if isinstance(count, Exception):
            raise count
        assert isinstance(count, int)
        return count

    module = types.ModuleType("ctranslate2")
    module.get_cuda_device_count = get_cuda_device_count  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", module)


def test_runtime_error_becomes_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ctranslate2(monkeypatch, RuntimeError("driver too old"))
    with pytest.raises(CudaUnavailableError, match="driver too old"):
        cuda_setup.prepare_cuda()


def test_import_error_becomes_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ctranslate2", None)  # import -> ImportError
    with pytest.raises(CudaUnavailableError, match="nvidia-smi"):
        cuda_setup.prepare_cuda()


def test_zero_devices_is_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ctranslate2(monkeypatch, 0)
    with pytest.raises(CudaUnavailableError, match="CUDA-устройств не найдено"):
        cuda_setup.prepare_cuda()


def test_success_marks_prepared(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ctranslate2(monkeypatch, 1)
    cuda_setup.prepare_cuda()
    _fake_ctranslate2(monkeypatch, RuntimeError("must not be called"))
    cuda_setup.prepare_cuda()


def test_dll_dirs_added_once_across_failed_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bins = {package: tmp_path / package / "bin" for package, _ in cuda_setup._REQUIRED}
    for package, dll in cuda_setup._REQUIRED:
        bins[package].mkdir(parents=True)
        (bins[package] / dll).touch()
    (bins["nvidia.cudnn"] / "cudnn_ops64_9.dll").touch()

    added: list[str] = []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: added.append(p), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cuda_setup, "_package_bin", lambda package: bins[package])
    monkeypatch.setenv("PATH", "base")
    _fake_ctranslate2(monkeypatch, RuntimeError("no driver"))

    for _ in range(2):
        with pytest.raises(CudaUnavailableError):
            cuda_setup.prepare_cuda()

    expected = sorted(map(str, bins.values()))
    assert sorted(added) == expected
    path_entries = os.environ["PATH"].split(os.pathsep)
    assert [path_entries.count(entry) for entry in expected] == [1] * len(expected)
