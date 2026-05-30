import json
import logging
import os
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)


class UserMemory:
    """
    用户常用词记忆与权重更新持久化模块。
    
    默认以 JSON 格式存储在本地文件系统，支持权重增加、同键衰减、持久化保存与清空。
    """

    def __init__(self, file_path: str = None):
        if not file_path:
            # 默认保存在用户主目录下，避免写入仓库被 Git 跟踪
            self.file_path = os.path.join(os.path.expanduser("~"), ".golf_user_memory.json")
        else:
            self.file_path = file_path

        # 内存数据结构: {input_key: {word: {"weight": float, "count": int, "last_used_at": float}}}
        self.data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.load()

    def load(self) -> None:
        """从磁盘加载用户词库"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.info("成功加载用户记忆文件: %s", self.file_path)
            except Exception:
                logger.warning("加载用户记忆文件失败，初始化为空", exc_info=True)
                self.data = {}
        else:
            self.data = {}

    def save(self) -> None:
        """持久化保存用户词库到磁盘"""
        try:
            # 确保父目录存在
            parent_dir = os.path.dirname(self.file_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("持久化用户记忆失败 (file=%s)", self.file_path, exc_info=True)

    def record_selection(self, word: str, input_key: str) -> None:
        """
        当用户选中某个候选词时，记录选择行为并更新权重。
        
        Args:
            word: 用户选择的上屏词 (如 "你好")
            input_key: 当前输入的拼音/前缀键 (如 "nihao")
        """
        if not input_key or not word:
            return

        input_key = input_key.lower().strip()

        if input_key not in self.data:
            self.data[input_key] = {}

        # 1. 衰减同 input_key 下的其他词的权重，以防权重无限增长并支持新词超越旧词
        for w in list(self.data[input_key].keys()):
            if w != word:
                self.data[input_key][w]["weight"] *= 0.95  # 衰减
                # 权重太小时清除，防止垃圾堆积
                if self.data[input_key][w]["weight"] < 0.01:
                    del self.data[input_key][w]

        # 2. 更新或新建目标词的选择属性
        now = time.time()
        if word not in self.data[input_key]:
            self.data[input_key][word] = {
                "weight": 1.0,
                "count": 1,
                "last_used_at": now,
                "source": "user_selected",
                "enabled": True
            }
        else:
            entry = self.data[input_key][word]
            # 每次选择加权值, 最大限制为 100.0
            entry["weight"] = min(100.0, entry["weight"] + 1.0)
            entry["count"] += 1
            entry["last_used_at"] = now

        # 3. 立即持久化
        self.save()

    def get_user_weight(self, word: str, input_key: str) -> float:
        """获取用户记忆中对应词在特定输入键下的权重，默认为 0.0"""
        if not input_key:
            return 0.0
        input_key = input_key.lower().strip()
        if input_key in self.data and word in self.data[input_key]:
            return float(self.data[input_key][word].get("weight", 0.0))
        return 0.0

    def clear(self) -> None:
        """清空用户记忆内存，并删除磁盘上的持久化文件"""
        self.data = {}
        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
                logger.info("已删除用户记忆文件: %s", self.file_path)
            except Exception:
                logger.warning("删除用户记忆文件失败 (file=%s)", self.file_path, exc_info=True)
        else:
            logger.info("用户记忆文件不存在，无需删除")
