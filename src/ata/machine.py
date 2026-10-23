"""Small helpers for recording reproducible local machine details."""

from __future__ import annotations

import os
import platform


def cpu_model() -> str:
    """Return a CPU model name when the operating system exposes one."""
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except (ImportError, OSError):
            pass
    return platform.processor() or "unreported"


def core_count() -> int | None:
    """Return the number of logical CPU cores available to this process."""
    return os.cpu_count()
