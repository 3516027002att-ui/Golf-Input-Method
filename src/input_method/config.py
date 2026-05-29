import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class InputMethodConfig:
    # 模式选择: 'pinyin' (中文拼音) 或 'english' (英文补全)
    mode: str = "pinyin"
    
    # 候选词列表分页大小 (每页展示候选数，通常为 5)
    page_size: int = 5
    
    # 召回的最大候选数上限，防止性能下降
    max_recall: int = 100
    
    # 是否启用机器学习排词模型重排
    use_model_rerank: bool = False
    
    # 默认 Tokenizer 路径
    tokenizer_path: str = os.path.join("data", "tokenizers", "fineweb_1024_bpe.model")
    
    # AI 排词模型路径 (预留，待后续接入)
    model_path: Optional[str] = None
    
    # 用户个性化词典路径 (预留)
    user_dict_path: Optional[str] = None

    def validate(self) -> None:
        """验证配置项的合法性"""
        if self.mode not in ("pinyin", "english"):
            raise ValueError(f"Unsupported mode: {self.mode}. Must be 'pinyin' or 'english'.")
        if self.page_size <= 0:
            raise ValueError("page_size must be a positive integer.")
        if self.max_recall <= 0:
            raise ValueError("max_recall must be a positive integer.")
