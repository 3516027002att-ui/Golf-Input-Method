import tkinter as tk
from typing import List
from .generator.base import Candidate

class GuiCandidateWindow(tk.Toplevel):
    """
    输入法悬浮候选词窗口。
    
    采用无边框设计，永远置顶，采用精美的暗色质感视觉设计，展示输入缓冲区和候选列表。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 1. 基础无边框与置顶设置
        self.overrideredirect(True)       # 去除 Windows 边框和标题栏
        self.attributes("-topmost", True)  # 永远置顶在最前
        self.withdraw()                    # 初始隐藏
        
        # 2. 视觉配色定义 (现代深色主题)
        self.bg_color = "#1E1E24"          # 深色主背
        self.border_color = "#00E5FF"      # 青色霓虹窄边框
        self.text_color = "#E2E8F0"        # 浅白文字
        self.composing_color = "#FFD60A"   # 黄色拼音缓冲
        
        self.highlight_bg = "#007ACC"      # 首位选词高亮蓝
        self.highlight_fg = "#FFFFFF"      # 高亮文字白
        self.dim_color = "#718096"         # 灰色分页信息
        
        # 3. 窗口最外层容器（用于画 1px 细边框）
        self.main_frame = tk.Frame(self, bg=self.border_color, bd=0)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.inner_frame = tk.Frame(self.main_frame, bg=self.bg_color, padx=12, pady=6)
        self.inner_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # 4. 子标签元素初始化
        # 拼音输入显示行 (e.g. nihao|)
        self.composing_label = tk.Label(
            self.inner_frame,
            text="",
            font=("Consolas", 11, "underline"),
            fg=self.composing_color,
            bg=self.bg_color,
            anchor="w"
        )
        self.composing_label.pack(fill=tk.X, anchor="w", pady=(0, 4))
        
        # 候选词列表展示横行容器
        self.cand_container = tk.Frame(self.inner_frame, bg=self.bg_color)
        self.cand_container.pack(fill=tk.X, anchor="w")

    def update_cands(self, composing: str, candidates: List[Candidate], page_index: int, total_pages: int, mode: str) -> None:
        """更新候选窗口渲染内容"""
        # 清理之前的候选词控件
        for widget in self.cand_container.winfo_children():
            widget.destroy()

        # 如果没有 composing 且没有候选，则不展示
        if not composing and not candidates:
            self.hide()
            return

        # 更新拼音展示
        mode_tag = "[PY]" if mode == "pinyin" else "[EN]"
        if composing:
            self.composing_label.config(text=f"{mode_tag} {composing}|")
            self.composing_label.pack(fill=tk.X, anchor="w", pady=(0, 4))
        else:
            self.composing_label.pack_forget()  # 无 composing 时隐藏此 Label，用于纯联想展示

        # 渲染候选词
        if candidates:
            for idx, cand in enumerate(candidates):
                # 候选词小框容器
                cand_frame = tk.Frame(self.cand_container, bg=self.bg_color)
                cand_frame.pack(side=tk.LEFT, padx=6)
                
                num_lbl = tk.Label(
                    cand_frame,
                    text=f"{idx+1}.",
                    font=("Segoe UI", 10, "bold"),
                    fg=self.border_color if idx != 0 else self.highlight_fg,
                    bg=self.bg_color if idx != 0 else self.highlight_bg,
                    padx=2
                )
                num_lbl.pack(side=tk.LEFT)
                
                text_lbl = tk.Label(
                    cand_frame,
                    text=f" {cand.text} ",
                    font=("Segoe UI", 10),
                    fg=self.text_color if idx != 0 else self.highlight_fg,
                    bg=self.bg_color if idx != 0 else self.highlight_bg,
                    padx=4,
                    pady=2
                )
                text_lbl.pack(side=tk.LEFT)
                
                # 为首词做圆角或高亮底色修饰
                if idx == 0 and composing:
                    num_lbl.config(bg=self.highlight_bg)
                    text_lbl.config(bg=self.highlight_bg)

            # 分页指示 label
            if total_pages > 1:
                page_lbl = tk.Label(
                    self.cand_container,
                    text=f" [{page_index+1}/{total_pages}]",
                    font=("Segoe UI", 9),
                    fg=self.dim_color,
                    bg=self.bg_color
                )
                page_lbl.pack(side=tk.LEFT, padx=(12, 0))
        else:
            none_lbl = tk.Label(
                self.cand_container,
                text="无匹配字词",
                font=("Segoe UI", 10, "italic"),
                fg=self.dim_color,
                bg=self.bg_color
            )
            none_lbl.pack(side=tk.LEFT)

        # 动态重算悬浮窗宽高
        self.update_idletasks()
        req_w = self.inner_frame.winfo_reqwidth() + 4
        req_h = self.inner_frame.winfo_reqheight() + 4
        self.geometry(f"{req_w}x{req_h}")

    def show(self, x: int, y: int) -> None:
        """在绝对坐标 (x, y) 展现窗口"""
        self.geometry(f"+{x}+{y}")
        self.deiconify()

    def hide(self) -> None:
        """隐藏窗口"""
        self.withdraw()
