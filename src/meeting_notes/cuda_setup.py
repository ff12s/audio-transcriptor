"""Подключение CUDA DLL из pip-пакетов nvidia-* до импорта faster_whisper."""

import importlib.util
import os
import sys
from pathlib import Path

from meeting_notes.errors import CudaUnavailableError

_INSTALL_HINT = (
    "Выполните `uv sync` или установите: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*"
)

# (пакет, обязательная DLL)
_REQUIRED = (("nvidia.cublas", "cublas64_12.dll"), ("nvidia.cudnn", "cudnn64_9.dll"))

# Хэндлы AddDllDirectory нужно держать живыми: после закрытия каталог убирается из поиска.
_dll_handles: list[object] = []
# Повторный prepare_cuda после сбоя не должен дублировать каталоги в поиске DLL и PATH.
_added_dirs: set[Path] = set()
_prepared = False


def _package_bin(package: str) -> Path:
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        spec = None
    if spec is None or not spec.submodule_search_locations:
        raise CudaUnavailableError(f"Не найден пакет {package}. {_INSTALL_HINT}")
    return Path(next(iter(spec.submodule_search_locations))) / "bin"


def _add_windows_dll_dirs() -> None:
    bins: list[Path] = []
    for package, dll in _REQUIRED:
        bin_dir = _package_bin(package)
        if not (bin_dir / dll).is_file():
            raise CudaUnavailableError(f"Не найден {dll} в {bin_dir}. {_INSTALL_HINT}")
        bins.append(bin_dir)
    cudnn_bin = bins[1]
    if not any(cudnn_bin.glob("cudnn_ops64_9.dll")):
        raise CudaUnavailableError(f"Не найден cudnn_ops64_9.dll в {cudnn_bin}. {_INSTALL_HINT}")

    new_bins = [bin_dir for bin_dir in bins if bin_dir not in _added_dirs]
    for bin_dir in new_bins:
        _dll_handles.append(os.add_dll_directory(str(bin_dir)))
        _added_dirs.add(bin_dir)
    if new_bins:
        # cuDNN 9 подгружает свои под-DLL сам, и этот поиск может игнорировать AddDllDirectory;
        # PATH правим только у текущего процесса.
        os.environ["PATH"] = os.pathsep.join([*map(str, new_bins), os.environ.get("PATH", "")])


def prepare_cuda() -> None:
    global _prepared
    if _prepared:
        return
    if sys.platform == "win32":
        _add_windows_dll_dirs()

    try:
        import ctranslate2

        device_count = ctranslate2.get_cuda_device_count()
    except (ImportError, OSError, RuntimeError) as exc:
        raise CudaUnavailableError(
            f"Не удалось инициализировать CUDA: {exc}. Проверьте драйвер NVIDIA "
            f"(`nvidia-smi`). {_INSTALL_HINT}"
        ) from exc

    if device_count == 0:
        raise CudaUnavailableError(
            "CUDA-устройств не найдено. Проверьте драйвер NVIDIA (`nvidia-smi`) и наличие GPU."
        )
    _prepared = True
