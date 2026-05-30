"""golf 系统输入法宿主 —— 全局键盘钩子 + 文本注入 + 光标追踪。

通过低层 Windows 键盘钩子 (WH_KEYBOARD_LL) 拦截全局按键，
送入 golf 引擎处理，并在任意应用中显示候选窗和提交文本。

此模块是 golf 从"自带编辑器 Demo"升级到"系统级可用输入法"的核心桥梁。
"""

import ctypes
import ctypes.wintypes
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ── Windows API 常量 ──
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
VK_BACK = 0x08
VK_RETURN = 0x0D
VK_SPACE = 0x20
VK_ESCAPE = 0x1B
VK_OEM_MINUS = 0xBD  # - 键
VK_OEM_PLUS = 0xBB   # = 键
VK_PRIOR = 0x21       # PageUp
VK_NEXT = 0x22        # PageDown
VK_DELETE = 0x2E
VK_0 = 0x30
VK_9 = 0x39
VK_A = 0x41
VK_Z = 0x5A

INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002

# ── ctypes 结构定义 ──

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.wintypes.DWORD),
        ("scanCode", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.wintypes.ULONG),
    ]


class INPUT_U(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("ki", ctypes.c_ubyte * 32),  # placeholder for KEYBDINPUT union
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("hwndActive", ctypes.wintypes.HWND),
        ("hwndFocus", ctypes.wintypes.HWND),
        ("hwndCapture", ctypes.wintypes.HWND),
        ("hwndMenuOwner", ctypes.wintypes.HWND),
        ("hwndMoveSize", ctypes.wintypes.HWND),
        ("hwndCaret", ctypes.wintypes.HWND),
        ("rcCaret", RECT),
    ]


# ── 键盘事件 ──

class KeyEvent:
    """标准化的按键事件。"""
    __slots__ = ("vk_code", "char", "is_keydown")
    def __init__(self, vk_code: int, char: str, is_keydown: bool):
        self.vk_code = vk_code
        self.char = char
        self.is_keydown = is_keydown


# ── 文本注入器 ──

class TextInjector:
    """通过 SendInput 向当前焦点窗口注入文本。"""

    @staticmethod
    def inject(text: str) -> None:
        user32 = ctypes.windll.user32
        for ch in text:
            _send_unicode_char(user32, ch)

    @staticmethod
    def inject_key(vk_code: int, keydown: bool = True) -> None:
        """注入虚拟键事件。"""
        user32 = ctypes.windll.user32

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.wintypes.WORD),
                ("wScan", ctypes.wintypes.WORD),
                ("dwFlags", ctypes.wintypes.DWORD),
                ("time", ctypes.wintypes.DWORD),
                ("dwExtraInfo", ctypes.wintypes.ULONG),
            ]

        class INPUT_STRUCT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.wintypes.DWORD),
                ("ki", KEYBDINPUT),
            ]

        inp = INPUT_STRUCT()
        inp.type = INPUT_KEYBOARD
        inp.ki.wVk = vk_code
        inp.ki.wScan = 0
        inp.ki.dwFlags = 0 if keydown else KEYEVENTF_KEYUP
        inp.ki.time = 0
        inp.ki.dwExtraInfo = 0
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _send_unicode_char(user32, ch: str) -> None:
    """发送单个 Unicode 字符。"""
    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.wintypes.WORD),
            ("wScan", ctypes.wintypes.WORD),
            ("dwFlags", ctypes.wintypes.DWORD),
            ("time", ctypes.wintypes.DWORD),
            ("dwExtraInfo", ctypes.wintypes.ULONG),
        ]

    class INPUT_STRUCT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.wintypes.DWORD),
            ("ki", KEYBDINPUT),
        ]

    val = ord(ch)
    inp = INPUT_STRUCT()
    inp.type = INPUT_KEYBOARD
    inp.ki.wVk = 0
    inp.ki.wScan = val
    inp.ki.dwFlags = KEYEVENTF_UNICODE
    inp.ki.time = 0
    inp.ki.dwExtraInfo = 0
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    inp.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# ── 光标追踪器 ──

class CaretTracker:
    """获取当前前台窗口的光标屏幕坐标。"""

    @staticmethod
    def get_caret_screen_pos() -> Optional[tuple[int, int]]:
        """返回光标屏幕坐标 (x, y)，失败返回 None。"""
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        # 前台窗口线程
        foreground_hwnd = user32.GetForegroundWindow()
        if not foreground_hwnd:
            return None

        thread_id = user32.GetWindowThreadProcessId(foreground_hwnd, None)
        current_thread_id = kernel32.GetCurrentThreadId()

        # AttachThreadInput 以获取 GUI 线程信息
        user32.AttachThreadInput(current_thread_id, thread_id, True)
        try:
            gti = GUITHREADINFO()
            gti.cbSize = ctypes.sizeof(GUITHREADINFO)
            if user32.GetGUIThreadInfo(thread_id, ctypes.byref(gti)):
                # rcCaret 是相对于窗口的坐标，需要转换为屏幕坐标
                caret_rect = gti.rcCaret
                if caret_rect.left != 0 or caret_rect.top != 0:
                    pt = POINT(caret_rect.left, caret_rect.bottom)
                    user32.ClientToScreen(gti.hwndFocus or foreground_hwnd, ctypes.byref(pt))
                    return (pt.x, pt.y + 2)
        finally:
            user32.AttachThreadInput(current_thread_id, thread_id, False)

        return None

    @staticmethod
    def get_foreground_app_name() -> str:
        """返回前台应用窗口标题。"""
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 255)
        return buf.value or ""


# ── 全局键盘钩子 ──

class GlobalKeyboardHook:
    """低层全局键盘钩子 (WH_KEYBOARD_LL)。

    在独立线程中运行 Windows 消息循环。
    """

    def __init__(self, event_callback: Callable[[KeyEvent], bool]):
        """
        event_callback: 接收 KeyEvent，返回 True 表示已处理（吞键），False 表示放行。
        """
        self._callback = event_callback
        self._hook_id: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32

        # 防止回调被 GC 回收
        HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM)
        self._hook_proc = HOOKPROC(self._low_level_keyboard_proc)

    def _low_level_keyboard_proc(self, nCode: int, wParam: int, lParam: int) -> int:
        """钩子回调：在钩子线程上下文中调用。"""
        if nCode >= 0:
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk = kb.vkCode
            is_keydown = (wParam == WM_KEYDOWN or wParam == WM_SYSKEYDOWN)

            if is_keydown:
                char = _vk_to_char(vk, kb.flags)
                event = KeyEvent(vk_code=vk, char=char, is_keydown=True)
                try:
                    handled = self._callback(event)
                    if handled:
                        return 1  # 吞键（阻止传递给目标应用）
                except Exception:
                    logger.exception("IME hook callback error")

        return self._user32.CallNextHookEx(
            self._hook_id if self._hook_id else 0, nCode, wParam, lParam
        )

    def start(self) -> None:
        """在后台线程中安装钩子并启动消息循环。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._hook_thread, daemon=True, name="golf-ime-hook")
        self._thread.start()

    def _hook_thread(self) -> None:
        """钩子线程入口：安装钩子 + 消息循环。"""
        hinstance = self._kernel32.GetModuleHandleW(None)
        self._hook_id = self._user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._hook_proc, hinstance, 0
        )
        if not self._hook_id:
            logger.error("SetWindowsHookExW 失败")
            self._running = False
            return

        logger.info("全局键盘钩子已安装 (hook_id=%s)", self._hook_id)

        # Windows 消息循环（钩子需要）
        msg = ctypes.create_string_buffer(28)
        while self._running:
            result = self._user32.GetMessageW(ctypes.byref(msg), 0, 0, 0)
            if result in (0, -1):
                break
            self._user32.TranslateMessage(ctypes.byref(msg))
            self._user32.DispatchMessageW(ctypes.byref(msg))

        self._unhook()

    def _unhook(self) -> None:
        if self._hook_id:
            self._user32.UnhookWindowsHookEx(self._hook_id)
            self._hook_id = None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._user32.PostThreadMessageW(self._thread.ident, 0x0012, 0, 0)  # WM_QUIT
            self._thread.join(timeout=2)


def _vk_to_char(vk: int, flags: int) -> str:
    """虚拟键码 → 字符（简化映射）。"""
    # 字母 A-Z
    if VK_A <= vk <= VK_Z:
        return chr(vk - VK_A + ord("a"))
    # 数字 0-9（主键区）
    if VK_0 <= vk <= VK_9:
        return chr(vk - VK_0 + ord("0"))
    # 空格
    if vk == VK_SPACE:
        return " "
    # 其他键返回空字符串（由调用方按 vk_code 处理）
    return ""


# ── IME 引擎桥接 ──

class ImeBridge:
    """连接 GlobalKeyboardHook 与 InputMethodEngine 的桥梁。

    处理按键 → 引擎 → 候选/提交 的完整流程。
    所有引擎操作在 Tkinter 主线程中执行（通过 after 调度）。
    """

    def __init__(
        self,
        engine: Any,  # InputMethodEngine
        root: Any,    # tk.Tk
        on_update_ui: Callable[[], None],
        on_commit_text: Callable[[str], None],
    ):
        self.engine = engine
        self._root = root
        self._on_update_ui = on_update_ui
        self._on_commit_text = on_commit_text
        self._enabled = True
        self._pending_queue: queue.Queue[KeyEvent] = queue.Queue()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val
        if not val:
            self.engine.clear()

    # 会被 IME 拦截的键码集合
    _IME_KEYS = {
        VK_BACK, VK_RETURN, VK_SPACE, VK_ESCAPE, VK_DELETE,
        VK_OEM_MINUS, VK_OEM_PLUS, VK_PRIOR, VK_NEXT,
    }

    # 中英切换相关的虚拟键码
    VK_CONTROL = 0x11
    VK_SHIFT = 0x10

    def handle_key_event(self, event: KeyEvent) -> bool:
        """全局键盘事件入口（钩子线程调用）。返回 True 表示吞键。

        只拦截 IME 可能处理的键；其余放行让目标应用正常接收。
        """
        if not self._enabled:
            return False

        vk = event.vk_code

        # Ctrl+Shift 切换中英文（任一方向）
        if vk == self.VK_SHIFT or vk == self.VK_CONTROL:
            ctrl = ctypes.windll.user32.GetAsyncKeyState(self.VK_CONTROL) & 0x8000
            shift = ctypes.windll.user32.GetAsyncKeyState(self.VK_SHIFT) & 0x8000
            if ctrl and shift:
                self._pending_queue.put(KeyEvent(vk, "", True))  # 特殊标记
                return True

        # 英文模式完全不拦截
        if self.engine.config.mode == "english":
            return False

        # 字母键：始终拦截（中文模式核心输入）
        if event.char and event.char.isalpha():
            self._pending_queue.put(event)
            return True

        # 数字键 1-5：候选选词
        if VK_0 < vk <= VK_0 + min(5, self.engine.config.page_size):
            self._pending_queue.put(event)
            return True

        # 特殊功能键
        if vk in self._IME_KEYS:
            self._pending_queue.put(event)
            return True

        # 放行
        return False

    def process_pending(self) -> None:
        """处理队列中的按键事件（必须在 Tkinter 主线程调用）。"""
        try:
            while True:
                event = self._pending_queue.get_nowait()
                self._dispatch(event)
        except queue.Empty:
            pass

    def _dispatch(self, event: KeyEvent) -> None:
        """在 Tkinter 线程中分发按键事件到引擎。"""
        eng = self.engine
        vk = event.vk_code
        char = event.char

        # Ctrl+Shift 切换中英文
        if vk in (self.VK_SHIFT, self.VK_CONTROL) and not char:
            ctrl = ctypes.windll.user32.GetAsyncKeyState(self.VK_CONTROL) & 0x8000
            shift = ctypes.windll.user32.GetAsyncKeyState(self.VK_SHIFT) & 0x8000
            if ctrl and shift:
                new = "english" if eng.config.mode == "pinyin" else "pinyin"
                eng.switch_mode(new)
                self._on_update_ui()
                logger.info("热键切换模式: %s", new.upper())
                return

        # 退格
        if vk == VK_BACK:
            if eng.composing:
                eng.handle_backspace()
                self._on_update_ui()
                return

        # 回车
        if vk == VK_RETURN:
            if eng.composing:
                text = eng.composing
                eng.handle_enter()
                self._on_commit_text(text)
                self._on_update_ui()
                return

        # 空格
        if vk == VK_SPACE:
            if eng.composing or (
                eng.get_current_page_candidates()
                and any(c.source == "association" for c in eng.candidates[:1])
            ):
                page = eng.get_current_page_candidates()
                if page:
                    text = page[0].text
                    eng.handle_space()
                    self._on_commit_text(text)
                    self._on_update_ui()
                return

        # Esc
        if vk == VK_ESCAPE:
            if eng.composing:
                eng.composing = ""
                eng.refresh_candidates()
                self._on_update_ui()
                return

        # 数字选词 1-5
        if VK_0 < vk <= VK_0 + min(5, eng.config.page_size):
            num = vk - VK_0
            page = eng.get_current_page_candidates()
            if page and len(page) >= num:
                text = page[num - 1].text
                eng.handle_candidate_select(num)
                self._on_commit_text(text)
                self._on_update_ui()
                return

        # 翻页 - / = / PageUp / PageDown
        if vk in (VK_OEM_MINUS, VK_PRIOR):
            if eng.handle_page_prev():
                self._on_update_ui()
            return
        if vk in (VK_OEM_PLUS, VK_NEXT):
            if eng.handle_page_next():
                self._on_update_ui()
            return

        # 字母输入（仅中文/日语模式）
        if char and char.isalpha() and eng.config.mode != "english":
            if eng.handle_char(char):
                self._on_update_ui()
                return

        # 未处理 → 放行
        return


# ── 系统 IME 宿主 ──

class SystemImeHost:
    """系统输入法宿主：管理键盘钩子 + 引擎 + 候选窗的完整生命周期。

    用法:
        host = SystemImeHost(engine, root, on_candidate_update, on_text_commit)
        host.start()   # 启动全局键盘拦截
        host.stop()    # 停止
    """

    def __init__(
        self,
        engine: Any,
        root: Any,
        on_candidate_update: Callable[[], None],
        on_text_commit: Callable[[str], None],
    ):
        self._bridge = ImeBridge(engine, root, on_candidate_update, on_text_commit)
        self._hook = GlobalKeyboardHook(event_callback=self._bridge.handle_key_event)
        self._root = root
        self._polling = False

    @property
    def enabled(self) -> bool:
        return self._bridge.enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._bridge.enabled = val

    def start(self) -> None:
        """启动全局键盘钩子并开始轮询。"""
        self._hook.start()
        self._start_polling()
        logger.info("SystemImeHost 已启动")

    def stop(self) -> None:
        """停止钩子和轮询。"""
        self._polling = False
        self._hook.stop()
        logger.info("SystemImeHost 已停止")

    def _start_polling(self) -> None:
        """在 Tkinter 主线程中周期性处理按键队列。"""
        self._polling = True

        def poll() -> None:
            if not self._polling:
                return
            self._bridge.process_pending()
            self._root.after(10, poll)  # 每 10ms 处理一次

        self._root.after(10, poll)
