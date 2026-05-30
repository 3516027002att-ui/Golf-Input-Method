import json
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class LexiconEntry:
    word: str
    pinyin: str
    short_pinyin: str
    freq: float
    source: str = "lexicon"


class LexiconLoader:
    """词库加载器，支持解析特定格式的词库文件 (如 JSONL)"""

    @staticmethod
    def load_from_jsonl(file_path: str) -> List[LexiconEntry]:
        """从 JSONL 文件中加载词库条目"""
        entries: List[LexiconEntry] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 校验必填字段
                    word = data["word"]
                    pinyin = data["pinyin"]
                    short_pinyin = data["short_pinyin"]
                    freq = float(data["freq"])
                    source = data.get("source", "lexicon")
                    
                    entries.append(LexiconEntry(
                        word=word,
                        pinyin=pinyin,
                        short_pinyin=short_pinyin,
                        freq=freq,
                        source=source
                    ))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(
                        "解析词库行出错 (file=%s, line=%d): %s",
                        file_path, line_idx, str(e)
                    )
                    # 抛出异常让上层捕获，决定是否降级
                    raise e
        return entries
