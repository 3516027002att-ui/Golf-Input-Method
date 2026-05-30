import tkinter as tk
from typing import Callable, List, Optional
from .generator.base import Candidate


def _truncate_text(text: str, max_len: int = 20) -> str:
    """截断过长文本"""
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


class GuiCandidateWindow(tk.Toplevel):
    """输入法悬浮候选词窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.withdraw()

        self.bg_color = "#1E1E24"
        self.border_color = "#00E5FF"
        self.text_color = "#E2E8F0"
        self.composing_color = "#FFD60A"
        self.highlight_bg = "#007ACC"
        self.highlight_fg = "#FFFFFF"
        self.dim_color = "#718096"

        # 候选点击回调：signature (index_on_page: int) -> None
        self.on_click_callback: Optional[Callable[[int], None]] = None

        self.main_frame = tk.Frame(self, bg=self.border_color, bd=0)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.inner_frame = tk.Frame(self.main_frame, bg=self.bg_color, padx=12, pady=6)
        self.inner_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        self.composing_label = tk.Label(
            self.inner_frame,
            text="",
            font=("Consolas", 11, "underline"),
            fg=self.composing_color,
            bg=self.bg_color,
            anchor="w",
        )
        self.composing_label.pack(fill=tk.X, anchor="w", pady=(0, 4))

        self.cand_container = tk.Frame(self.inner_frame, bg=self.bg_color)
        self.cand_container.pack(fill=tk.X, anchor="w")

    def update_cands(
        self,
        composing: str,
        candidates: List[Candidate],
        page_index: int,
        total_pages: int,
        mode: str,
    ) -> None:
        for widget in self.cand_container.winfo_children():
            widget.destroy()

        if not composing and not candidates:
            self.hide()
            return

        mode_tag = {"pinyin": "[PY]", "english": "[EN]", "japanese": "[JA 原型]"}.get(mode, "[??]")
        if composing:
            self.composing_label.config(text=f"{mode_tag} {composing}|")
            self.composing_label.pack(fill=tk.X, anchor="w", pady=(0, 4))
        else:
            self.composing_label.pack_forget()

        if candidates:
            for idx, cand in enumerate(candidates):
                cand_frame = tk.Frame(self.cand_container, bg=self.bg_color)
                cand_frame.pack(side=tk.LEFT, padx=6)

                is_primary = idx == 0 and bool(composing)
                num_lbl = tk.Label(
                    cand_frame,
                    text=f"{idx + 1}.",
                    font=("Segoe UI", 10, "bold"),
                    fg=self.highlight_fg if is_primary else self.border_color,
                    bg=self.highlight_bg if is_primary else self.bg_color,
                    padx=2,
                    cursor="hand2",
                )
                num_lbl.pack(side=tk.LEFT)

                display_text = _truncate_text(cand.text)
                text_lbl = tk.Label(
                    cand_frame,
                    text=f" {display_text} ",
                    font=("Segoe UI", 10),
                    fg=self.highlight_fg if is_primary else self.text_color,
                    bg=self.highlight_bg if is_primary else self.bg_color,
                    padx=4,
                    pady=2,
                    cursor="hand2",
                )
                text_lbl.pack(side=tk.LEFT)

                # 鼠标点击候选上屏
                click_idx = idx
                num_lbl.bind("<Button-1>", lambda e, i=click_idx: self._on_click(i))
                text_lbl.bind("<Button-1>", lambda e, i=click_idx: self._on_click(i))

            if total_pages > 1:
                page_lbl = tk.Label(
                    self.cand_container,
                    text=f" [{page_index + 1}/{total_pages}]",
                    font=("Segoe UI", 9),
                    fg=self.dim_color,
                    bg=self.bg_color,
                )
                page_lbl.pack(side=tk.LEFT, padx=(12, 0))
        else:
            none_text = "无匹配字词"
            if mode == "japanese":
                none_text = "日语模式原型：暂无完整候选"
            none_lbl = tk.Label(
                self.cand_container,
                text=none_text,
                font=("Segoe UI", 10, "italic"),
                fg=self.dim_color,
                bg=self.bg_color,
            )
            none_lbl.pack(side=tk.LEFT)

        self.update_idletasks()
        req_w = self.inner_frame.winfo_reqwidth() + 4
        req_h = self.inner_frame.winfo_reqheight() + 4
        self.geometry(f"{req_w}x{req_h}")

    def _on_click(self, index: int) -> None:
        """内部点击处理"""
        if self.on_click_callback:
            self.on_click_callback(index)

    def show(self, x: int, y: int) -> None:
        self.geometry(f"+{x}+{y}")
        self.deiconify()

    def hide(self) -> None:
        self.withdraw()

    def destroy(self) -> None:
        """安全销毁窗口"""
        try:
            super().destroy()
        except tk.TclError:
            pass
