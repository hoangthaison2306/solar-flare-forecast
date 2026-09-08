"""
paths.py
--------
One place to normalize image paths.

prediction_history.csv was written on Windows, so every image_path uses
backslashes ("data\\hmi_jpg\\...\\x.jpg"). On Linux or any cloud host those
strings never resolve, so the dashboard shows "HMI MAGNETOGRAM · NOT FOUND"
even though the file is present on disk. Always write paths with as_posix()
and always read them through resolve_image_path().
"""

from pathlib import Path, PurePosixPath, PureWindowsPath


def to_storage_path(path: Path) -> str:
    """Canonical form written to CSV: forward slashes, on every platform."""
    return Path(path).as_posix()


def resolve_image_path(raw, base: Path | None = None) -> Path | None:
    """
    Turn a stored image_path into a Path that works on this machine,
    accepting either separator. Returns None if nothing exists there.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # PureWindowsPath splits on both separators; PurePosixPath only on "/".
    # Using the Windows parser first means "a\b/c.jpg" is handled either way.
    parts = PureWindowsPath(text).parts if "\\" in text else PurePosixPath(text).parts
    if not parts:
        return None

    candidate = Path(*parts)
    for root in (Path.cwd() if base is None else base, Path(__file__).parent):
        full = candidate if candidate.is_absolute() else root / candidate
        if full.exists():
            return full
    return None
