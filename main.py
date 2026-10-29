"""Launch the DriveMind 2D autonomous-driving demonstration."""
import os
from pathlib import Path
import sys
import ctypes
import atexit
import subprocess


def _windows_short_path(path: Path) -> str:
    """Return an 8.3 path so Tcl can open its library under space-containing user paths."""
    try:
        get_short_path = ctypes.windll.kernel32.GetShortPathNameW
        get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        get_short_path.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_short_path(str(path), buffer, len(buffer))
        if length and length < len(buffer):
            return buffer.value
    except (AttributeError, OSError):
        pass
    return str(path)


def _tcl_paths(prefix: Path) -> tuple[str, str]:
    tcl = prefix / "tcl" / "tcl8.6"
    tk = prefix / "tcl" / "tk8.6"

    return (
        _windows_short_path(tcl),
        _windows_short_path(tk),
    )

# The installed Python 3.12 environment has Tk bindings but does not always
# discover its bundled Tcl/Tk scripts when invoked from a virtual environment.
for prefix in {Path(sys.prefix), Path(sys.base_prefix)}:
    tcl = prefix / "tcl" / "tcl8.6"
    tk = prefix / "tcl" / "tk8.6"
    if (tcl / "init.tcl").is_file():
        tcl_path, tk_path = _tcl_paths(prefix)
        os.environ["TCL_LIBRARY"] = tcl_path
        if tk.is_dir():
            os.environ["TK_LIBRARY"] = tk_path
        break

from etl.simulation.dashboard import main

if __name__ == "__main__":
    main()
