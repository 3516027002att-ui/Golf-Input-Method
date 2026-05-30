"""golf 系统输入法应用 — 系统托盘 + 全局键盘钩子 + 候选窗。

启动后在通知区域显示图标，全局拦截键盘输入，可在任意应用中使用中文输入。
"""

import ctypes
import ctypes.wintypes
import logging
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
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


# ── 系统托盘图标 (Shell_NotifyIcon) ──

NIM_ADD = 0
NIM_DELETE = 2
NIM_MODIFY = 1
NIF_MESSAGE = 0x01
NIF_ICON = 0x02
NIF_TIP = 0x04
NIF_INFO = 0x10
WM_TRAY = 0x8001
WM_COMMAND = 0x0111

IDM_TOGGLE = 1001
IDM_SWITCH = 1002
IDM_EXIT = 1003


class _NOTIFYICONDATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.wintypes.DWORD),
        ("hWnd", ctypes.wintypes.HWND),
        ("uID", ctypes.wintypes.UINT),
        ("uFlags", ctypes.wintypes.UINT),
        ("uCallbackMessage", ctypes.wintypes.UINT),
        ("hIcon", ctypes.wintypes.HICON),
        ("szTip", ctypes.c_char * 128),
        ("dwState", ctypes.wintypes.DWORD),
        ("dwStateMask", ctypes.wintypes.DWORD),
        ("szInfo", ctypes.c_char * 256),
        ("uVersion", ctypes.wintypes.UINT),
        ("szInfoTitle", ctypes.c_char * 64),
        ("dwInfoFlags", ctypes.wintypes.DWORD),
    ]


class SystemTrayIcon:
    """Windows 通知区域图标，使用 Shell_NotifyIcon + 独立线程消息循环。"""

    def __init__(self, on_toggle=None, on_switch=None, on_exit=None):
        self._on_toggle = on_toggle
        self._on_switch = on_switch
        self._on_exit = on_exit
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hwnd = None
        self._tip = b"golf \xe8\xbe\x93\xe5\x85\xa5\xe6\xb3\x95"  # "golf 输入法"

    def set_tip(self, text: str) -> None:
        self._tip = text.encode("gbk", errors="replace")[:127]
        if self._hwnd:
            self._modify()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._tray_thread, daemon=True, name="golf-tray")
        self._thread.start()

    def _tray_thread(self) -> None:
        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        kernel32 = ctypes.windll.kernel32

        # 注册窗口类
        wndclass = ctypes.wintypes.WNDCLASSW()
        wndclass.lpfnWndProc = _tray_wnd_proc
        wndclass.hInstance = kernel32.GetModuleHandleW(None)
        wndclass.lpszClassName = "GolfIMETrayWindow"
        atom = user32.RegisterClassW(ctypes.byref(wndclass))

        # 创建隐藏消息窗口
        hwnd = user32.CreateWindowExW(
            0, ctypes.c_char_p(atom), b"GolfTray", 0,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self._hwnd = hwnd

        # 存储 self 引用到全局变量，供窗口过程回调使用
        _tray_instances[id(self)] = self

        # 添加托盘图标
        nid = _NOTIFYICONDATA()
        nid.cbSize = ctypes.sizeof(_NOTIFYICONDATA)
        nid.hWnd = hwnd
        nid.uID = 1
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        nid.uCallbackMessage = WM_TRAY
        nid.hIcon = user32.LoadIconW(0, 32512)  # IDI_APPLICATION
        nid.szTip = self._tip
        shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid))
        _tray_instances[id(self)] = self

        logger.info("系统托盘图标已创建")

        # 消息循环
        msg = ctypes.wintypes.MSG()
        while self._running:
            result = user32.GetMessageW(ctypes.byref(msg), 0, 0, 0)
            if result <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        # 清理
        nid2 = _NOTIFYICONDATA()
        nid2.cbSize = ctypes.sizeof(_NOTIFYICONDATA)
        nid2.hWnd = hwnd
        nid2.uID = 1
        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid2))
        user32.DestroyWindow(hwnd)
        self._hwnd = None
        _tray_instances.pop(id(self), None)
        logger.info("系统托盘图标已移除")

    def _modify(self) -> None:
        if not self._hwnd:
            return
        shell32 = ctypes.windll.shell32
        nid = _NOTIFYICONDATA()
        nid.cbSize = ctypes.sizeof(_NOTIFYICONDATA)
        nid.hWnd = self._hwnd
        nid.uID = 1
        nid.uFlags = NIF_TIP
        nid.szTip = self._tip
        shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))

    def _handle_message(self, msg: int, lParam: int) -> None:
        if msg == WM_TRAY:
            if lParam == 0x0205:  # WM_RBUTTONUP — 右击弹出菜单
                self._show_menu()
            elif lParam == 0x0203:  # WM_LBUTTONDBLCLK — 双击切换
                if self._on_switch:
                    self._on_switch()
        elif msg == WM_COMMAND:
            cmd = lParam & 0xFFFF
            if cmd == IDM_TOGGLE and self._on_toggle:
                self._on_toggle()
            elif cmd == IDM_SWITCH and self._on_switch:
                self._on_switch()
            elif cmd == IDM_EXIT and self._on_exit:
                self._on_exit()

    def _show_menu(self) -> None:
        user32 = ctypes.windll.user32
        # 创建弹出菜单
        menu = user32.CreatePopupMenu()
        user32.AppendMenuW(menu, 0, IDM_TOGGLE, "启用/禁用 IME")
        user32.AppendMenuW(menu, 0, IDM_SWITCH, "切换中/英文")
        user32.AppendMenuW(menu, 0x800, 0, "")  # 分隔线
        user32.AppendMenuW(menu, 0, IDM_EXIT, "退出 golf")
        # 获取光标位置
        pt = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        user32.SetForegroundWindow(self._hwnd)
        user32.TrackPopupMenu(menu, 0, pt.x, pt.y, 0, self._hwnd, None)
        user32.DestroyMenu(menu)

    def stop(self) -> None:
        self._running = False
        if self._hwnd:
            user32 = ctypes.windll.user32
            user32.PostMessageW(self._hwnd, 0x0012, 0, 0)  # WM_QUIT
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


# 全局引用 — 窗口过程需要访问 TrayIcon 实例
_tray_instances: dict = {}


@ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.wintypes.HWND, ctypes.wintypes.UINT, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM)
def _tray_wnd_proc(hwnd, msg, wparam, lparam):
    if msg in (WM_TRAY, WM_COMMAND):
        # 遍历查找处理者（简单策略：取最后一个注册的实例）
        for obj in list(_tray_instances.values()):
            try:
                obj._handle_message(msg, lparam)
            except Exception:
                pass
            break
    return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)


# ── 系统 IME 应用 ──

class GolfImeApp:
    """golf 系统输入法主应用。"""

    def __init__(self, config: Optional[InputMethodConfig] = None):
        self.config = config or InputMethodConfig(mode="pinyin")
        self.engine = InputMethodEngine(self.config)

        self._root = tk.Tk()
        self._root.withdraw()
        self._root.title("golf IME Host")

        self._cand_win: Optional[GuiCandidateWindow] = None
        self._ime_host: Optional[SystemImeHost] = None
        self._enabled = True
        self._running = False

        # 托盘
        self._tray = SystemTrayIcon(
            on_toggle=self.toggle_enabled,
            on_switch=self.switch_mode,
            on_exit=self.stop,
        )

    def _ensure_cand_win(self) -> GuiCandidateWindow:
        if self._cand_win is None:
            self._cand_win = GuiCandidateWindow(parent=None)
        return self._cand_win

    def _update_candidate_ui(self) -> None:
        cands = self.engine.get_current_page_candidates()
        composing = self.engine.composing

        if not composing and not cands:
            if self._cand_win:
                self._cand_win.hide()
            return

        win = self._ensure_cand_win()
        win.update_cands(
            composing=composing, candidates=cands,
            page_index=self.engine.page_index,
            total_pages=self.engine.total_pages(),
            mode=self.engine.config.mode,
        )

        caret_pos = CaretTracker.get_caret_screen_pos()
        if caret_pos:
            x, y = caret_pos
            win.show(x, y)
        else:
            win.show(800, 500)

    def _commit_text(self, text: str) -> None:
        if text:
            TextInjector.inject(text)
            logger.info("已提交: %r", text)

    def start(self) -> None:
        if self._running:
            return
        self._tray.start()
        self._ime_host = SystemImeHost(
            engine=self.engine, root=self._root,
            on_candidate_update=self._update_candidate_ui,
            on_text_commit=self._commit_text,
        )
        self._ime_host.start()
        self._running = True
        self._update_tray_tip()
        logger.info("golf 系统 IME 已启动")

    def stop(self) -> None:
        if self._ime_host:
            self._ime_host.stop()
        if self._cand_win:
            self._cand_win.destroy()
        self._tray.stop()
        self._running = False
        self._root.quit()

    def toggle_enabled(self) -> None:
        self._enabled = not self._enabled
        if self._ime_host:
            self._ime_host.enabled = self._enabled
        if not self._enabled:
            if self._cand_win:
                self._cand_win.hide()
            self.engine.clear()
        self._update_tray_tip()

    def switch_mode(self) -> None:
        new = "english" if self.engine.config.mode == "pinyin" else "pinyin"
        self.engine.switch_mode(new)
        self._update_candidate_ui()
        self._update_tray_tip()

    def _update_tray_tip(self) -> None:
        mode = "中" if self.engine.config.mode == "pinyin" else "EN"
        status = "启用" if self._enabled else "禁用"
        self._tray.set_tip(f"golf 输入法 [{mode}] {status}")

    def run(self) -> None:
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

    lock = SingleInstance(_LOCK_FILE)
    if not lock.try_acquire():
        try:
            ctypes.windll.user32.MessageBoxW(
                0, "golf 输入法已在运行中。\n请查看通知区域图标。", "golf 输入法", 0x40
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
