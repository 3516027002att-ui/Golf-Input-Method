import json
import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LexiconEntry:
    word: str
    pinyin: str
    short_pinyin: str
    freq: float
    source: str = "lexicon"
    domain: str = ""
    enabled: bool = True


class LexiconLoader:
    """词库加载器，支持解析 JSONL 和 TSV 格式的词库文件，容错模式"""

    @staticmethod
    def load_from_jsonl(file_path: str) -> Tuple[List[LexiconEntry], int]:
        """从 JSONL 文件中加载词库条目，遇到坏行跳过并计数"""
        entries: List[LexiconEntry] = []
        bad_line_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    word = data["word"]
                    pinyin = data["pinyin"]
                    short_pinyin = data.get("short_pinyin", "")
                    freq = float(data.get("freq", 1000.0))
                    source = data.get("source", "lexicon")
                    domain = data.get("domain", "")
                    enabled = data.get("enabled", True)

                    entries.append(LexiconEntry(
                        word=word,
                        pinyin=pinyin,
                        short_pinyin=short_pinyin,
                        freq=freq,
                        source=source,
                        domain=domain,
                        enabled=bool(enabled),
                    ))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    bad_line_count += 1
                    logger.warning(
                        "解析词库行出错 (file=%s, line=%d): %s",
                        file_path, line_idx, str(e),
                    )
        return entries, bad_line_count

    @staticmethod
    def load_from_tsv(file_path: str) -> Tuple[List[LexiconEntry], int]:
        """从 TSV 文件加载词库条目 (word\tpinyin\tshort_pinyin\tfreq)"""
        entries: List[LexiconEntry] = []
        bad_line_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    parts = line.split("\t")
                    if len(parts) < 2:
                        bad_line_count += 1
                        continue
                    word = parts[0]
                    pinyin = parts[1]
                    short_pinyin = parts[2] if len(parts) > 2 else ""
                    freq = float(parts[3]) if len(parts) > 3 else 1000.0
                    source = parts[4] if len(parts) > 4 else "tsv"
                    domain = parts[5] if len(parts) > 5 else ""
                    enabled_str = parts[6] if len(parts) > 6 else "true"
                    enabled = enabled_str.lower() not in ("false", "0", "no")

                    entries.append(LexiconEntry(
                        word=word,
                        pinyin=pinyin,
                        short_pinyin=short_pinyin,
                        freq=freq,
                        source=source,
                        domain=domain,
                        enabled=enabled,
                    ))
                except (ValueError, IndexError) as e:
                    bad_line_count += 1
                    logger.warning(
                        "解析 TSV 词库行出错 (file=%s, line=%d): %s",
                        file_path, line_idx, str(e),
                    )
        return entries, bad_line_count
