import os
import stat
import sys
from pathlib import Path

import pytest

from meeting_notes.errors import OutputExistsError
from meeting_notes.render.output import write_output


def test_overwrite_replaces_content(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("old", encoding="utf-8")
    write_output(path, "новый\n", overwrite=True)
    assert path.read_text(encoding="utf-8") == "новый\n"
    assert [p.name for p in tmp_path.iterdir()] == ["a.md"]


def test_overwrite_failure_keeps_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a.md"
    path.write_text("old", encoding="utf-8")

    def boom(src: object, dst: object) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(PermissionError):
        write_output(path, "new", overwrite=True)
    assert path.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.md"]


def test_exclusive_create_refuses_existing(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("old", encoding="utf-8")
    with pytest.raises(OutputExistsError):
        write_output(path, "new", overwrite=False)
    assert path.read_text(encoding="utf-8") == "old"


@pytest.mark.skipif(sys.platform == "win32", reason="права POSIX не применимы на Windows")
def test_overwrite_keeps_permissions(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o640)
    write_output(path, "new", overwrite=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
