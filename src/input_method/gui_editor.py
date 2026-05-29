import tkinter as tk
from tkinter import ttk
from typing import Optional
from .engine import InputMethodEngine
from .config import InputMethodConfig
from .gui_candidate_window import GuiCandidateWindow

class GuiEditor:
    """
    极简墨黑风格输入法演示记事本。
    
    拦截 Text 编辑器的键盘事件并转发给输入法引擎，获取光标像素物理位置并弹出无边框候选栏。
    """

    def __init__(self, engine: InputMethodEngine):
        self.engine = engine
        
        # 1. 初始化 Tkinter 主窗口
        self.root = tk.Tk()
        self.root.title("golf AI 输入法记事本原型")
        self.root.geometry("800x600")
        
        # 2. 配色系统 (墨黑极简风格)
        self.bg_dark = "#121214"         # 窗口背景
        self.bg_editor = "#18181C"       # 输入框背景
        self.text_color = "#E2E8F0"      # 文字浅白
        self.border_color = "#2D3748"    # 边框深灰
        self.cyan_accent = "#00E5FF"     # 青色亮眼点缀
        self.gray_text = "#A0AEC0"       # 辅助性文字置灰

        self.root.configure(bg=self.bg_dark)

        # 3. 实例化悬浮候选窗口
        self.cand_win = GuiCandidateWindow(self.root)

        # 4. 构建主界面布局
        self._build_ui()

        # 5. 绑定键盘事件
        self.text_area.bind("<Key>", self.on_key_press)
        # 绑定窗口关闭事件，顺便把子窗口销毁
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        """渲染奢华现代墨黑质感的编辑器界面"""
        # A. 顶部工具栏
        self.toolbar = tk.Frame(self.root, bg=self.bg_dark, height=45)
        self.toolbar.pack(fill=tk.X, side=tk.TOP, padx=16, pady=8)
        
        # 标题 Label
        title_lbl = tk.Label(
            self.toolbar,
            text="GOLF EDITOR",
            font=("Segoe UI", 12, "bold"),
            fg=self.cyan_accent,
            bg=self.bg_dark
        )
        title_lbl.pack(side=tk.LEFT)
        
        # 状态指示器
        self.status_lbl = tk.Label(
            self.toolbar,
            text=f"当前模式: {self.engine.config.mode.upper()}",
            font=("Segoe UI", 9),
            fg=self.text_color,
            bg=self.bg_dark,
            padx=12
        )
        self.status_lbl.pack(side=tk.LEFT, padx=12)

        # 控制选项：切换输入模式按钮
        self.mode_btn = tk.Button(
            self.toolbar,
            text="切换中/英文模式",
            font=("Segoe UI", 9),
            fg=self.bg_dark,
            bg=self.cyan_accent,
            activebackground="#00B0FF",
            activeforeground=self.bg_dark,
            bd=0,
            padx=10,
            pady=4,
            command=self.toggle_mode
        )
        self.mode_btn.pack(side=tk.RIGHT, padx=6)
        
        # 控制选项：切换 AI 重排开关
        use_ml_str = "AI重排: 开启" if self.engine.config.use_model_rerank else "AI重排: 关闭"
        self.ml_btn = tk.Button(
            self.toolbar,
            text=use_ml_str,
            font=("Segoe UI", 9),
            fg=self.text_color,
            bg=self.bg_editor,
            activebackground=self.border_color,
            activeforeground=self.text_color,
            bd=0,
            padx=10,
            pady=4,
            command=self.toggle_ml
        )
        self.ml_btn.pack(side=tk.RIGHT, padx=6)

        # B. 编辑器主体文本区域 (外包一层带圆角/边框感的 Frame)
        self.editor_frame = tk.Frame(self.root, bg=self.border_color, bd=1)
        self.editor_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        
        # 滚动条
        self.scrollbar = tk.Scrollbar(self.editor_frame, bg=self.bg_editor, bd=0)
        self.scrollbar.pack(fill=tk.Y, side=tk.RIGHT)

        # 文本编辑器控件
        self.text_area = tk.Text(
            self.editor_frame,
            bg=self.bg_editor,
            fg=self.text_color,
            insertbackground=self.cyan_accent,   # 光标颜色
            selectbackground=self.border_color, # 选中区域颜色
            selectforeground=self.text_color,
            font=("Consolas", 13),
            padx=12,
            pady=12,
            bd=0,
            yscrollcommand=self.scrollbar.set
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.scrollbar.config(command=self.text_area.yview)

        # 默认为编辑器获取焦点
        self.text_area.focus_set()

    # --- 输入法按键拦截处理 ---

    def on_key_press(self, event) -> Optional[str]:
        """按键事件侦听器。拦截字母及输入法关联键并阻止默认字符在 Text 控件的上屏行为"""
        char = event.char
        keysym = event.keysym

        # 忽略 Ctrl、Alt、Win 等系统组合热键
        if event.state & (0x0004 | 0x0020):  # Control 或 Alt 键处于按下状态
            return None

        # 1. 处理字母输入
        # 中文模式下拦截所有英文字母 a-z；英文模式下拦截字母、数字、一些标点以进行前缀召回
        is_letter = char.isalpha() and char.isascii()
        
        if is_letter:
            # 统一转为小写传给引擎
            char_to_handle = char.lower()
            # 如果是拼音模式，或者英文模式下我们只要在拼写单词前缀就进行拦截
            if self.engine.config.mode == "pinyin" or (self.engine.config.mode == "english"):
                if self.engine.handle_char(char_to_handle):
                    self.update_ime_ui()
                    return "break" # 拦截，阻止直接写入 Text 框

        # 2. 处理退格键 (Backspace)
        if keysym == "BackSpace":
            if self.engine.composing:
                if self.engine.handle_backspace():
                    self.update_ime_ui()
                else:
                    self.cand_win.hide()
                return "break"

        # 3. 处理空格键 (Space)
        if keysym == "space":
            # 如果输入缓冲区不为空，空格选择首选词上屏
            # 如果缓冲区为空，但由于上一次输入，有联想候选词显示，则空格也是确认该联想词
            current_cands = self.engine.get_current_page_candidates()
            if self.engine.composing or (not self.engine.composing and current_cands and self.engine.candidates[0].source == "association"):
                first_cand = current_cands[0].text if current_cands else ""
                if first_cand:
                    self.commit_to_editor(first_cand)
                return "break"

        # 4. 处理回车键 (Return / Enter)
        if keysym == "Return":
            if self.engine.composing:
                # 提交 composing 原始拼音/字母上屏
                raw_comp = self.engine.composing
                self.commit_to_editor(raw_comp)
                return "break"

        # 5. 处理数字键 1-5 (选择候选词)
        if char in ("1", "2", "3", "4", "5"):
            current_cands = self.engine.get_current_page_candidates()
            # 只有在有候选列表时，数字键才进行拦截选词
            if current_cands and (self.engine.composing or current_cands[0].source == "association"):
                num = int(char)
                idx = num - 1
                if idx < len(current_cands):
                    selected_cand = current_cands[idx].text
                    self.commit_to_editor(selected_cand)
                return "break"

        # 6. 处理翻页键 - / = 或是 PageUp / PageDown
        if keysym in ("minus", "equal", "Prior", "Next"):
            current_cands = self.engine.get_current_page_candidates()
            if current_cands and (self.engine.composing or current_cands[0].source == "association"):
                if keysym == "minus" or keysym == "Prior": # "-" 键或 PageUp
                    if self.engine.handle_page_prev():
                        self.update_ime_ui()
                elif keysym == "equal" or keysym == "Next": # "=" 键或 PageDown
                    if self.engine.handle_page_next():
                        self.update_ime_ui()
                return "break"

        return None

    # --- 输入法与界面渲染机制 ---

    def update_ime_ui(self) -> None:
        """获取光标物理坐标位置，更新悬浮候选窗内容和位置"""
        current_page_cands = self.engine.get_current_page_candidates()
        
        if not self.engine.composing and not current_page_cands:
            self.cand_win.hide()
            return

        # 1. 刷新内容
        self.cand_win.update_cands(
            composing=self.engine.composing,
            candidates=current_page_cands,
            page_index=self.engine.page_index,
            total_pages=self.engine.total_pages(),
            mode=self.engine.config.mode
        )

        # 2. 定位物理像素位置 (在 Text 光标 insert 正下方弹出)
        # 获取光标在 Text 里的相对坐标 (x, y, w, h)
        bbox = self.text_area.bbox("insert")
        if bbox:
            rel_x, rel_y, _, char_h = bbox
            
            # 计算绝对物理像素坐标
            abs_x = self.text_area.winfo_rootx() + rel_x
            # 候选框放在光标文字下一行，给 6px 外边距
            abs_y = self.text_area.winfo_rooty() + rel_y + char_h + 6
            
            # 展示窗口
            self.cand_win.show(abs_x, abs_y)
        else:
            self.cand_win.hide()

    def commit_to_editor(self, text: str) -> None:
        """将选定文本插入到 Text 控件的光标处，清空输入引擎并触发联想"""
        # 1. 在当前光标处插入文字
        self.text_area.insert(tk.INSERT, text)
        
        # 2. 更新引擎端的状态，同步已上屏文本（引擎用于联想词生成）
        # 我们可以获取最后 50 个字符提供给引擎做上下文联想
        context = self.text_area.get("insert -50 chars", "insert")
        self.engine.committed_history = context
        self.engine.composing = ""
        self.engine._update_candidates() # 引擎重新召回联想词

        # 3. 刷新 GUI 输入法窗口（显示联想词）
        self.update_ime_ui()

    # --- 控制逻辑 ---

    def toggle_mode(self) -> None:
        """中英文输入模式切换"""
        new_mode = "english" if self.engine.config.mode == "pinyin" else "pinyin"
        self.engine.config.mode = new_mode
        # 重建引擎相关的 generators
        self.engine.__init__(self.engine.config)
        
        self.status_lbl.config(text=f"当前模式: {new_mode.upper()}")
        self.text_area.focus_set()
        
        # 重新触发一次更新
        self.update_ime_ui()

    def toggle_ml(self) -> None:
        """开启或关闭 AI 排序"""
        new_val = not self.engine.config.use_model_rerank
        self.engine.config.use_model_rerank = new_val
        # 重新初始化引擎以载入/卸载 model_reranker
        self.engine.__init__(self.engine.config)

        btn_str = "AI重排: 开启" if new_val else "AI重排: 关闭"
        self.ml_btn.config(text=btn_str, bg=self.cyan_accent if new_val else self.bg_editor, fg=self.bg_dark if new_val else self.text_color)
        self.text_area.focus_set()

    def on_close(self) -> None:
        """主窗口关闭时安全销毁子窗口"""
        self.cand_win.destroy()
        self.root.destroy()

    def run(self) -> None:
        """启动主交互循环"""
        self.root.mainloop()
