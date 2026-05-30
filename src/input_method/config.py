import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_FUZZY_PINYIN: Dict[str, bool] = {
    "zh_z": False,
    "ch_c": False,
    "sh_s": False,
    "n_l": False,
    "en_eng": False,
    "in_ing": False,
}


@dataclass
class InputMethodConfig:
    # 模式选择: pinyin / english / japanese
    mode: str = "pinyin"

    # 候选词列表分页大小
    page_size: int = 5

    # 召回的最大候选数上限，防止性能下降
    max_recall: int = 100

    # 是否启用机器学习排词模型重排
    use_model_rerank: bool = False

    # 默认 Tokenizer 路径
    tokenizer_path: str = os.path.join("data", "tokenizers", "fineweb_1024_bpe.model")

    # 外部中文词库路径；不存在或加载失败时降级到 RAW_WORDS
    dict_path: Optional[str] = os.path.join("data", "lexicon", "dict.jsonl")

    # AI 排词模型路径 (预留，待后续接入)
    model_path: Optional[str] = None

    # 用户个性化记忆路径；为空时由 UserMemory 使用用户目录默认路径
    user_dict_path: Optional[str] = None

    # 是否启用用户选词学习功能（记忆用户选择以优化排序）
    learning_enabled: bool = True

    # 模糊音配置：键为模糊音对，值为是否开启
    fuzzy_pinyin: Dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_FUZZY_PINYIN))

    # 候选布局: horizontal / vertical
    candidate_layout: str = "horizontal"

    # 日志级别: DEBUG / INFO / WARNING / ERROR
    log_level: str = "WARNING"

    # 主题（预留）
    theme: str = "dark"

    # 字体
    font_family: str = "Consolas"
    font_size: int = 13

    def validate(self) -> None:
        """验证配置项的合法性"""
        if self.mode not in ("pinyin", "english", "japanese"):
            raise ValueError(
                f"Unsupported mode: {self.mode}. Must be 'pinyin', 'english' or 'japanese'."
            )
        if self.page_size <= 0:
            raise ValueError("page_size must be a positive integer.")
        if self.max_recall <= 0:
            raise ValueError("max_recall must be a positive integer.")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError(
                f"Invalid log_level: {self.log_level}. Must be DEBUG/INFO/WARNING/ERROR."
            )
        if not isinstance(self.learning_enabled, bool):
            raise ValueError("learning_enabled must be a boolean.")
        if self.candidate_layout not in ("horizontal", "vertical"):
            raise ValueError(
                f"Invalid candidate_layout: {self.candidate_layout}. Must be horizontal/vertical."
            )
        if self.font_size <= 0:
            raise ValueError("font_size must be a positive integer.")

    @classmethod
    def from_json_file(cls, path: str) -> "InputMethodConfig":
        """从 JSON 配置文件加载配置，失败时返回默认配置"""
        config = cls()
        if not path or not os.path.exists(path):
            logger.warning("配置文件不存在: %s，使用默认配置", path)
            return config
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _apply_dict_to_config(config, data)
        except Exception:
            logger.warning("解析配置文件失败: %s，使用默认配置", path, exc_info=True)
        return config

    @classmethod
    def from_args_and_file(
        cls, args: Any, config_path: Optional[str] = None
    ) -> "InputMethodConfig":
        """先从 JSON 文件加载配置，再用命令行参数覆盖非 None 值"""
        if config_path:
            config = cls.from_json_file(config_path)
        else:
            config = cls()

        # 命令行参数覆盖
        arg_mapping = {
            "mode": "mode",
            "page_size": "page_size",
            "use_model": "use_model_rerank",
            "model_path": "model_path",
            "dict_path": "dict_path",
            "user_memory_path": "user_dict_path",
            "log_level": "log_level",
        }
        for arg_name, config_attr in arg_mapping.items():
            val = getattr(args, arg_name, None)
            if val is not None:
                setattr(config, config_attr, val)

        # 布尔开关
        if getattr(args, "learning_enabled", None) is not None:
            config.learning_enabled = args.learning_enabled

        return config


def _apply_dict_to_config(config: "InputMethodConfig", data: dict) -> None:
    """将字典中的键值对安全地应用到配置对象"""
    simple_fields = {
        "mode", "page_size", "max_recall", "use_model_rerank",
        "tokenizer_path", "dict_path", "model_path", "learning_enabled",
        "candidate_layout", "log_level", "theme", "font_family", "font_size",
    }
    for key in simple_fields:
        if key in data:
            setattr(config, key, data[key])

    if "user_memory_path" in data:
        config.user_dict_path = data["user_memory_path"]

    if "fuzzy_pinyin" in data and isinstance(data["fuzzy_pinyin"], dict):
        config.fuzzy_pinyin.update(data["fuzzy_pinyin"])
