"""golf 系统输入法应用。

启动后在系统托盘中运行，全局拦截键盘输入，在任意应用中提供中文输入能力。
候选窗为独立置顶窗口，自动定位到当前光标位置。
"""

import ctypes
import logging
import os
import sys
import threading
import tkinter as tk
from typing import Optional

from .config import InputMethodConfig
from .engine import InputMethodEngine
from .gui_candidate_window import GuiCandidateWindow
from .ime_host import CaretTracker, SystemImeHost, TextInjector

logger = logging.getLogger(__name__)

# ── 单实例锁 ──
_LOCK_DIR = os.path.join(os.path.expanduser("~"), ".golf")
_LOCK_FILE = os.path.join(_LOCK_DIR, "golf_ime.lock")


class SingleInstance:
    def __init__(self, path: str):
        self.path = path
        self._fd = None

    def try_acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            import msvcrt
            self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
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


# ── 系统 IME 应用 ──

class GolfImeApp:
    """golf 系统输入法主应用。

    架构：
    - 隐藏 Tk root → 承载候选窗和消息轮询
    - 后台线程 → 全局键盘钩子 (WH_KEYBOARD_LL)
    - SystemImeHost → 桥接钩子事件到 Engine
    - CaretTracker → 获取外部应用光标位置
    - TextInjector → 向外部应用注入文本
    """

    def __init__(self, config: Optional[InputMethodConfig] = None):
        self.config = config or InputMethodConfig(mode="pinyin")
        self.engine = InputMethodEngine(self.config)

        # 隐藏的 Tk root（只用于候选窗和消息处理）
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.title("golf IME Host")

        # 候选窗（parent=None → 独立顶层窗口）
        self._cand_win: Optional[GuiCandidateWindow] = None
        self._ime_host: Optional[SystemImeHost] = None
        self._enabled = True
        self._running = False

    # ── 候选窗管理 ──

    def _ensure_cand_win(self) -> GuiCandidateWindow:
        if self._cand_win is None:
            self._cand_win = GuiCandidateWindow(parent=None)
        return self._cand_win

    def _update_candidate_ui(self) -> None:
        """根据引擎状态更新候选窗位置和内容。"""
        cands = self.engine.get_current_page_candidates()
        composing = self.engine.composing

        if not composing and not cands:
            if self._cand_win:
                self._cand_win.hide()
            return

        win = self._ensure_cand_win()
        win.update_cands(
            composing=composing,
            candidates=cands,
            page_index=self.engine.page_index,
            total_pages=self.engine.total_pages(),
            mode=self.engine.config.mode,
        )

        # 获取外部应用光标位置
        caret_pos = CaretTracker.get_caret_screen_pos()
        if caret_pos:
            x, y = caret_pos
            win.show(x, y)
        else:
            # 回退：显示在屏幕右下角
            win.show(800, 500)

    def _commit_text(self, text: str) -> None:
        """将文本注入到当前前台应用。"""
        if text:
            TextInjector.inject(text)
            logger.info("已提交文本到前台应用: %r", text)

    # ── 生命周期 ──

    def start(self) -> None:
        """启动系统 IME。"""
        if self._running:
            return

        self._ime_host = SystemImeHost(
            engine=self.engine,
            root=self._root,
            on_candidate_update=self._update_candidate_ui,
            on_text_commit=self._commit_text,
        )
        self._ime_host.start()
        self._running = True
        logger.info("golf 系统 IME 已启动。当前模式: %s", self.engine.config.mode)

    def stop(self) -> None:
        """停止系统 IME。"""
        if self._ime_host:
            self._ime_host.stop()
            self._ime_host = None
        if self._cand_win:
            self._cand_win.destroy()
            self._cand_win = None
        self._running = False
        self._root.quit()

    def toggle_enabled(self) -> None:
        """启用/禁用 IME。"""
        self._enabled = not self._enabled
        if self._ime_host:
            self._ime_host.enabled = self._enabled
        if not self._enabled and self._cand_win:
            self._cand_win.hide()
        status = "启用" if self._enabled else "禁用"
        logger.info("IME 已%s", status)

    def switch_mode(self) -> None:
        """切换中/英文模式。"""
        new = "english" if self.engine.config.mode == "pinyin" else "pinyin"
        self.engine.switch_mode(new)
        self._update_candidate_ui()
        logger.info("模式切换: %s", new.upper())

    def run(self) -> None:
        """启动并进入 Tkinter 事件循环。"""
        self.start()
        self._root.mainloop()
        self.stop()


# ── 命令行入口 ──

def _hide_console() -> None:
    if os.name == "nt":
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd != 0:
                ctypes.windll.user32.ShowWindow(hwnd, 0)
        except Exception:
            pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="golf 系统输入法")
    parser.add_argument("--mode", "-m", choices=["pinyin", "english"], default="pinyin")
    parser.add_argument("--dict-path", type=str, default=None)
    parser.add_argument("--show-console", action="store_true")
    args = parser.parse_args()

    if not args.show_console:
        _hide_console()

    # 单实例检查
    lock = SingleInstance(_LOCK_FILE)
    if not lock.try_acquire():
        try:
            ctypes.windll.user32.MessageBoxW(
                0,
                "golf 输入法已在运行中。\n"
                "如需重启，请先退出已有实例。",
                "golf 输入法",
                0x40,
            )
        except Exception:
            pass
        sys.exit(0)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    config = InputMethodConfig(mode=args.mode, dict_path=args.dict_path)
    app = GolfImeApp(config)

    try:
        app.run()
    finally:
        lock.release()


if __name__ == "__main__":
    main()
