import argparse
import sys
import os
from .config import InputMethodConfig
from .engine import InputMethodEngine
from .cli_simulator import CliSimulator, IS_WINDOWS

def run_fallback_simulator(engine: InputMethodEngine) -> None:
    """非 Windows 或不支持 msvcrt 时的跨平台行式输入模拟器"""
    print("=" * 60)
    print("      golf AI 输入法框架 — 跨平台简易交互行 (流模式)")
    print("=" * 60)
    print(" [指令说明] ")
    print(" - 键入英文字母并按回车：更新输入缓冲区")
    print(" - 输入数字 (1-5)：选择对应候选词并提交")
    print(" - 输入 '-' 或 '='：翻页")
    print(" - 直接按回车 (空输入)：提交缓冲区原始文本")
    print(" - 输入 '/exit'：退出")
    print(" - 输入 '/mode'：切换中/英文模式")
    print("=" * 60)

    while True:
        if engine.config.mode == "pinyin":
            mode_str = "拼音"
        elif engine.config.mode == "japanese":
            mode_str = "日语"
        else:
            mode_str = "英文"
        print(f"\n[已提交历史]: {engine.committed_history if engine.committed_history else '(空)'}")
        print(f"[当前缓冲区 ({mode_str})]: {engine.composing if engine.composing else '(空)'}")
        
        # 打印候选词
        cands = engine.get_current_page_candidates()
        cand_strs = []
        for idx, cand in enumerate(cands):
            cand_strs.append(f"{idx+1}.{cand.text}")
        
        tot_pages = engine.total_pages()
        page_info = f" (页: {engine.page_index + 1}/{tot_pages})" if tot_pages > 0 else ""
        print(f"[候选列表]: {'  '.join(cand_strs) if cand_strs else '(无候选)'}{page_info}")
        
        try:
            user_input = input("输入命令或字符 >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出模拟器。")
            break

        if user_input == "/exit":
            print("已退出模拟器。")
            break
        elif user_input == "/mode":
            current_mode = engine.config.mode
            if current_mode == "pinyin":
                new_mode = "english"
            elif current_mode == "english":
                new_mode = "japanese"
            else:
                new_mode = "pinyin"
            engine.switch_mode(new_mode)
            print(f"模式已切换为: {new_mode.upper()}")
            continue
        elif user_input == "-":
            engine.handle_page_prev()
            continue
        elif user_input == "=":
            engine.handle_page_next()
            continue
        elif user_input in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            num = int(user_input)
            engine.select_candidate_on_page(num - 1)
            continue
        elif not user_input:
            # 回车直接提交原始字母
            engine.handle_enter()
            continue
        
        # 否则，视为键入字符
        for char in user_input:
            engine.handle_char(char)


def main() -> None:
    parser = argparse.ArgumentParser(description="golf AI 输入法框架模拟端入口")
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
        help="外部词库加载路径"
    )
    parser.add_argument(
        "--user-memory-path",
        type=str,
        default=None,
        help="用户记忆持久化路径"
    )
    
    args = parser.parse_args()

    # 初始化配置
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

    # 启动引擎
    try:
        engine = InputMethodEngine(config)
    except Exception as e:
        print(f"初始化引擎失败: {str(e)}")
        sys.exit(1)

    # 启动交互
    if IS_WINDOWS:
        simulator = CliSimulator(engine)
        try:
            simulator.run()
        except Exception as e:
            print(f"\n模拟器运行中发生异常: {str(e)}")
            print("正在退避到跨平台行交互模式...")
            run_fallback_simulator(engine)
    else:
        run_fallback_simulator(engine)

if __name__ == "__main__":
    main()
