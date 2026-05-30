import sys
import os
import argparse
import ctypes
from .config import InputMethodConfig
from .engine import InputMethodEngine
from .gui_editor import GuiEditor

def hide_console_window() -> None:
    """
    如果在 Windows 下运行，并且程序启动了控制台黑框，
    利用 Windows API 将其隐藏，以达到纯图形桌面软件运行的效果（无后台黑框常驻）。
    """
    if os.name == "nt":
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd != 0:
                # SW_HIDE = 0
                ctypes.windll.user32.ShowWindow(hwnd, 0)
        except Exception:
            pass

def main() -> None:
    parser = argparse.ArgumentParser(description="golf AI 输入法图形客户端")
    parser.add_argument(
        "--mode", "-m",
        choices=["pinyin", "english", "japanese"],
        default="pinyin",
        help="指定初始输入法模式: pinyin (中文拼音), english (英文前缀补全) 或 japanese (日语罗马字假名)，默认为 pinyin"
    )
    parser.add_argument(
        "--page-size", "-p",
        type=int,
        default=5,
        help="候选词每页显示大小，默认 5"
    )
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="是否激活排词机器学习模型重排"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="排词模型权重路径"
    )
    parser.add_argument(
        "--dict-path",
        type=str,
        default=None,
        help="外部词表路径"
    )
    parser.add_argument(
        "--user-memory-path",
        type=str,
        default=None,
        help="用户记忆持久化路径"
    )
    parser.add_argument(
        "--show-console",
        action="store_true",
        help="保持命令行控制台窗口可见（用于调试）"
    )

    args = parser.parse_args()

    # 1. 如果用户没有要求保留控制台，且在 Windows 下，自动隐藏黑框
    if not args.show_console:
        hide_console_window()

    # 2. 初始化输入法引擎配置
    config = InputMethodConfig(
        mode=args.mode,
        page_size=args.page_size,
        use_model_rerank=args.use_model,
        model_path=args.model_path
    )
    if args.dict_path:
        config.dict_path = args.dict_path
    if args.user_memory_path:
        config.user_dict_path = args.user_memory_path

    engine = InputMethodEngine(config)

    # 3. 启动图形客户端
    app = GuiEditor(engine)
    app.run()

if __name__ == "__main__":
    main()
