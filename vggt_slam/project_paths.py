"""Make this checkout's bundled third-party packages win over stale editable installs."""

from pathlib import Path
import sys


def prefer_bundled_third_party():
    """Prepend third_party sources from the current checkout to ``sys.path``.

    DSW environments are often reused across checkouts. An editable VGGT
    install can therefore keep pointing at an older repository even after the
    user changes directory. The package bootstrap and main entry points call
    this before importing any module that imports VGGT or SALAD.
    """
    project_root = Path(__file__).resolve().parents[1]
    bundled = (
        project_root / "third_party" / "vggt",
        project_root / "third_party" / "salad",
    )
    for path in reversed(bundled):
        if path.is_dir():
            path_string = str(path)
            if path_string not in sys.path:
                sys.path.insert(0, path_string)

    return tuple(str(path) for path in bundled if path.is_dir())
