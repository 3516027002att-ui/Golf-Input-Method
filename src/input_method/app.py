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
        default=None,
        help="指定初始输入法模式: pinyin (中文拼音), english (英文前缀补全) 或 japanese (日语罗马字假名)，默认为 pinyin"
    )
    parser.add_argument(
        "--page-size", "-p",
        type=int,
        default=None,
        help="候选词每页显示大小，默认 5"
    )
    parser.add_argument(
        "--use-model",
        action="store_true",
        default=None,
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
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件路径 (例如 config/default.json)"
    )
    parser.add_argument(
        "--learning-enabled",
        action="store_true",
        dest="learning_enabled",
        default=None,
        help="启用用户学习功能"
    )
    parser.add_argument(
        "--no-learning",
        action="store_false",
        dest="learning_enabled",
        help="禁用用户学习功能"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别"
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

    # 2. 初始化输入法引擎配置：先从配置文件加载，再用命令行参数覆盖
    config = InputMethodConfig.from_args_and_file(args, config_path=args.config)

    # 处理 page_size（argparse 中 default=None 以便区分是否显式指定）
    if args.page_size is not None:
        config.page_size = args.page_size

    engine = InputMethodEngine(config)

    # 3. 启动图形客户端
    app = GuiEditor(engine)
    app.run()

if __name__ == "__main__":
    main()
