"""重排：把查询和候选文档成对送入交叉编码器（cross-encoder）精细打分。

向量检索是“双塔”：查询和文档各自编码后算相似度，快但粗；
交叉编码器让查询和文档在模型内部逐字交互，更准但慢，所以只对融合后的前 N 条做。
"""
from __future__ import annotations

from typing import Protocol

from .vector import MODEL_DIR


class Reranker(Protocol):
    name: str

    def scores(self, query: str, texts: list[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as e:
            raise RuntimeError("没有安装 fastembed：请运行 pip install fastembed，或设置 EXAM_RERANKER=none") from e
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.name = model
        self._m = TextCrossEncoder(model_name=model, cache_dir=str(MODEL_DIR))

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._m.rerank(query, texts)]


class OverlapReranker:
    """测试用：按查询二元组在文档中的覆盖比例打分。"""
    name = "overlap"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        from .text import tokenize

        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / (len(q) or 1) for t in texts]
