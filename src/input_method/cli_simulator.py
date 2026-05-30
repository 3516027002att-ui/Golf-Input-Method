import locale
import logging
import os
import sys
import time
from typing import List
from .engine import InputMethodEngine

logger = logging.getLogger(__name__)

# 尝试载入 Windows 专用的 msvcrt 模块
try:
    import msvcrt
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False

# ANSI 颜色转义字符定义
class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    
    # 组合特效
    BG_PRIMARY_CAND = '\033[44;97m'  # 蓝底白字 (首位候选高亮)
    BG_TITLE = '\033[45;97m'         # 洋红底白字
    BORDER_COLOR = '\033[36m'        # 青色边框


class CliSimulator:
    """基于终端的无外部依赖输入法交互模拟客户端"""

    def __init__(self, engine: InputMethodEngine):
        self.engine = engine
        self.running = False
        self._command_buffer = ""  # 独立命令缓冲区（处理 /exit 等）
        self._in_command_mode = False  # 是否处于命令输入模式
        self._system_encoding = locale.getpreferredencoding(False) or "utf-8"
        
        # 激活 Windows 终端对 ANSI 转义序列的支持
        if IS_WINDOWS:
            os.system('')

    def draw_ui(self) -> None:
        """重绘输入法浮动框界面"""
        # 清屏并将光标移至左上角
        # 使用 \033[H\033[J 实现防闪烁重绘，较 os.system('cls') 更加平滑低延迟
        print("\033[H\033[J", end="")

        title_bar = f" {TermColors.BOLD}golf AI 输入法框架模拟器{TermColors.ENDC} "
        border = f"{TermColors.BORDER_COLOR}=" * 60 + TermColors.ENDC
        sub_border = f"{TermColors.BORDER_COLOR}-" * 60 + TermColors.ENDC

        # 1. 绘制头部
        print(border)
        print(f"{TermColors.BORDER_COLOR}||{TermColors.ENDC} {TermColors.BG_TITLE}{title_bar.center(56)}{TermColors.ENDC} {TermColors.BORDER_COLOR}||{TermColors.ENDC}")
        print(border)

        # 2. 绘制已提交的历史文本
        history_display = self.engine.committed_history
        if not history_display:
            history_display = f"{TermColors.DIM}（尚未有输入文字上屏，请开始键入...）{TermColors.ENDC}"
        else:
            history_display = f"{TermColors.OKGREEN}{TermColors.BOLD}{history_display}{TermColors.ENDC}"
        print(f" {TermColors.BOLD}已提交文字{TermColors.ENDC}: {history_display}")
        print()

        # 3. 绘制输入缓冲区 (Composing Text)
        if self.engine.config.mode == "pinyin":
            mode_indicator = "拼音"
        elif self.engine.config.mode == "japanese":
            mode_indicator = "日语"
        else:
            mode_indicator = "英文"
        comp_text = self.engine.composing
        if comp_text:
            # 模拟输入法正在输入时，字符下方有下划线，且光标闪烁
            comp_display = f"{TermColors.UNDERLINE}{TermColors.WARNING}{comp_text}{TermColors.ENDC}{TermColors.BOLD}|{TermColors.ENDC}"
        else:
            comp_display = f"{TermColors.DIM}[当前空闲]{TermColors.ENDC}"

        print(f" {TermColors.BOLD}输入缓冲区 ({mode_indicator}){TermColors.ENDC}: {comp_display}")
        print(sub_border)

        # 4. 绘制候选词栏
        current_page_cands = self.engine.get_current_page_candidates()
        cand_line_parts = []
        
        if current_page_cands:
            for idx, cand in enumerate(current_page_cands):
                num_key = idx + 1
                if idx == 0 and self.engine.composing:
                    # 首个候选词在 composing 不为空时予以高亮，表明按空格会直接选择它
                    cand_str = f"{TermColors.BOLD}{num_key}.{TermColors.BG_PRIMARY_CAND} {cand.text} {TermColors.ENDC}"
                else:
                    cand_str = f"{TermColors.OKCYAN}{num_key}.{TermColors.ENDC}{TermColors.BOLD}{cand.text}{TermColors.ENDC}"
                cand_line_parts.append(cand_str)
            
            # 分页标记
            tot_pages = self.engine.total_pages()
            page_info = f" {TermColors.DIM}(页: {self.engine.page_index + 1}/{tot_pages}){TermColors.ENDC}"
            cands_str = "   ".join(cand_line_parts) + page_info
        else:
            if self.engine.composing:
                cands_str = f" {TermColors.FAIL}（无匹配候选）{TermColors.ENDC}"
            else:
                # 联想候选词
                cands_str = f" {TermColors.DIM}（无候选，请输入字母）{TermColors.ENDC}"
                
        print(f" {TermColors.BOLD}候选词{TermColors.ENDC}:  {cands_str}")
        print(border)

        # 5. 绘制快捷键操作说明
        shortcuts = " [1-5] 选词 | [空格] 首选词上屏 | [回车] 原始字母上屏 | [-/=] 翻页 | [ESC] 退出"
        print(f"{TermColors.DIM}{shortcuts}{TermColors.ENDC}")

        # 6. 绘制引擎状态及调试面板
        use_ml = "开启" if self.engine.config.use_model_rerank else "关闭"
        # 显示模型加载状态
        if self.engine.config.use_model_rerank and hasattr(self.engine.reranker, "model_info"):
            model_info = getattr(self.engine.reranker, "model_info")
        else:
            model_info = "无模型 (基于词频静态排序)"
            
        status_line = (
            f" [模式: {TermColors.OKCYAN}{self.engine.config.mode.upper()}{TermColors.ENDC}]"
            f" [AI重排: {TermColors.OKGREEN if use_ml == '开启' else TermColors.DIM}{use_ml}{TermColors.ENDC}]"
            f" [延迟: {TermColors.WARNING}{self.engine.last_query_latency_ms:.2f}ms{TermColors.ENDC}]"
            f" [总候选数: {len(self.engine.candidates)}]"
        )
        print(status_line)
        print(f" [模型状态: {TermColors.DIM}{model_info}{TermColors.ENDC}]")
        print(border)
        print(f"{TermColors.DIM}（支持中英文切换，键入 '/mode' 切换模式，键入 '/clear' 清屏，键入 '/clear_memory' 清空用户记忆，键入 '/exit' 退出）{TermColors.ENDC}")

    def run(self) -> None:
        """启动键盘捕获事件主循环"""
        if not IS_WINDOWS:
            print("警告: 键盘捕获交互模拟器需要 Windows OS (依赖 msvcrt 库)。")
            print("目前您的操作系统不支持实时无回车交互，但仍可通过命令行执行单元测试验证框架功能。")
            return

        self.running = True
        
        # 初始绘制
        self.draw_ui()

        while self.running:
            # 检查是否有输入，若无，小憩 10ms 降级 CPU 消耗
            if not msvcrt.kbhit():
                time.sleep(0.01)
                continue

            # 读取键值
            ch_bytes = msvcrt.getch()
            if not ch_bytes:
                continue

            # 区分特殊按键与普通按键
            # msvcrt 在按方向键、翻页键等特殊功能键时会先返回 b'\xe0' 或 b'\x00'，接着返回扫描码
            if ch_bytes in (b'\xe0', b'\x00'):
                if msvcrt.kbhit():
                    scan_code = msvcrt.getch()
                    # Up Arrow (72) or PageUp (73) -> Page Prev
                    if scan_code in (b'H', b'I'):
                        if self.engine.handle_page_prev():
                            self.draw_ui()
                    # Down Arrow (80) or PageDown (81) -> Page Next
                    elif scan_code in (b'P', b'Q'):
                        if self.engine.handle_page_next():
                            self.draw_ui()
                continue

            # 处理 ESC 退出
            if ch_bytes == b'\x1b':
                self.running = False
                print("\n已退出模拟器。")
                break

            # 处理 Backspace (b'\x08')
            if ch_bytes == b'\x08':
                if self._in_command_mode:
                    # 命令模式下退格
                    if self._command_buffer:
                        self._command_buffer = self._command_buffer[:-1]
                        if not self._command_buffer:
                            self._in_command_mode = False
                    self.draw_ui()
                else:
                    if self.engine.handle_backspace():
                        self.draw_ui()
                continue

            # 处理回车 (b'\r')
            if ch_bytes == b'\r':
                if self._in_command_mode:
                    # 执行命令
                    cmd = self._command_buffer
                    self._command_buffer = ""
                    self._in_command_mode = False
                    if cmd == "/exit":
                        self.running = False
                        print("\n已退出模拟器。")
                        break
                    elif cmd == "/clear":
                        self.engine.clear()
                    elif cmd == "/clear_memory":
                        self.engine.clear_user_memory()
                    elif cmd == "/mode":
                        current_mode = self.engine.config.mode
                        if current_mode == "pinyin":
                            new_mode = "english"
                        elif current_mode == "english":
                            new_mode = "japanese"
                        else:
                            new_mode = "pinyin"
                        self.engine.switch_mode(new_mode)
                    self.draw_ui()
                else:
                    if self.engine.handle_enter():
                        self.draw_ui()
                continue

            # 处理空格 (b' ')
            if ch_bytes == b' ':
                if self._in_command_mode:
                    # 命令模式下空格无效
                    continue
                if self.engine.handle_space():
                    self.draw_ui()
                continue

            # 处理候选词选择快捷键 1-9（仅在非命令模式下）
            if not self._in_command_mode and ch_bytes in (b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9'):
                num = int(ch_bytes.decode('ascii'))
                if self.engine.handle_candidate_select(num):
                    self.draw_ui()
                continue

            # 处理翻页快捷键 - 和 =
            if ch_bytes == b'-':
                if self.engine.handle_page_prev():
                    self.draw_ui()
                continue
            if ch_bytes == b'=':
                if self.engine.handle_page_next():
                    self.draw_ui()
                continue

            # 处理普通的字母和符号按键
            try:
                char = ch_bytes.decode(self._system_encoding)
            except UnicodeDecodeError:
                logger.debug("无法解码按键字节: %r (encoding=%s)", ch_bytes, self._system_encoding)
                continue

            # 检测 '/' 字符进入命令模式
            if char == '/' and not self.engine.composing and not self._in_command_mode:
                self._in_command_mode = True
                self._command_buffer = "/"
                self.draw_ui()
                continue

            if self._in_command_mode:
                self._command_buffer += char
                self.draw_ui()
            else:
                if self.engine.handle_char(char):
                    self.draw_ui()
