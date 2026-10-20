import os
import shutil
import tempfile
from pathlib import Path

from meeting_notes.errors import OutputExistsError


def output_path_for(audio: Path) -> Path:
    return audio.with_suffix(".md")


def write_output(path: Path, text: str, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise OutputExistsError(f"Файл уже существует: {path}. Укажите --overwrite для перезаписи.")
    if overwrite:
        _replace_atomically(path, text)
        return
    try:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except FileExistsError:
        raise OutputExistsError(
            f"Файл уже существует: {path}. Укажите --overwrite для перезаписи."
        ) from None


def _replace_atomically(path: Path, text: str) -> None:
    # Временный файл в том же каталоге: os.replace атомарен только в пределах одной ФС.
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        # mkstemp создаёт файл с 0600. На Windows copymode переносит лишь read-only, и такой
        # временный файл потом не удалить при сбое замены.
        if os.name == "posix" and path.exists():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
