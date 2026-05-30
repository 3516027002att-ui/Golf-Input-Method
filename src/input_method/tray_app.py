"""golf 输入法 Windows 后台启动器。

功能：
- 双击即可启动 GUI（无需命令行）
- 单实例保护（同一用户只能运行一个 golf 实例）
- 后台守护：GUI 关闭后进程自动退出

注意：当前 golf 是本地输入法框架 + GUI 编辑器，尚未接入系统级 IME/TSF。
本启动器是桌面便捷入口，不是系统输入法安装器。
"""

import ctypes
import os
import sys
import tkinter as tk
from pathlib import Path
from typing import Optional

# ── 单实例锁 ──
_LOCK_DIR = os.path.join(os.path.expanduser("~"), ".golf")
_LOCK_FILE = os.path.join(_LOCK_DIR, "golf_instance.lock")


class _SingleInstance:
    """基于文件锁的单实例保护。"""

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._fd: Optional[int] = None

    def try_acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        try:
            import msvcrt
            self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
            msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            return True
        except (IOError, OSError):
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                import msvcrt
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None


def _hide_console() -> None:
    """隐藏控制台窗口（仅 Windows）。"""
    if os.name == "nt":
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd != 0:
                ctypes.windll.user32.ShowWindow(hwnd, 0)
        except Exception:
            pass


def main() -> None:
    """golf 后台启动入口。

    1. 检查单实例，避免重复启动。
    2. 隐藏控制台窗口。
    3. 启动 GUI 编辑器。
    """
    if os.name != "nt":
        print("当前仅为 Windows 提供后台启动。其他平台请直接使用 python -m src.input_method.app。")
        sys.exit(1)

    lock = _SingleInstance(_LOCK_FILE)
    if not lock.try_acquire():
        # 已有实例在运行 — 尝试激活已有窗口
        try:
            ctypes.windll.user32.MessageBoxW(0, "golf 输入法已在后台运行。\n请检查系统托盘区域。", "golf 输入法", 0x40)
        except Exception:
            pass
        sys.exit(0)

    _hide_console()

    # 导入并启动 GUI
    from .app import main as gui_main
    try:
        gui_main()
    finally:
        lock.release()


if __name__ == "__main__":
    main()
