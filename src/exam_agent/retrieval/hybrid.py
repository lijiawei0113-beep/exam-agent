"""混合检索：元数据过滤 → BM25 + 向量 → RRF 融合 → 交叉编码器重排。

为什么用 RRF（Reciprocal Rank Fusion）而不是分数相加
BM25 分数没有上限（0～20 都可能），余弦相似度在 -1～1，两者量纲不同，直接加权相加，权重怎么调都别扭。
RRF 只看名次：score(d) = Σ 1 / (k + rank_i(d))，k 取 60（原论文的经验值）。
在两路检索里都靠前的文档得分最高，只在一路里靠前的也不会被丢掉。

为什么先做元数据过滤
年级、学科这类硬条件绝不交给相似度去“大概判断”。“七年级的解方程”和“九年级的解方程”向量很像，
但对出卷来说一个能用、一个超纲。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .bm25 import BM25
from .kb import Doc
from .rerank import Reranker
from .vector import Embedder, VectorIndex

MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]
MODE_LABEL = {"bm25": "仅 BM25", "vector": "仅向量", "hybrid": "BM25 + 向量（RRF）",
              "hybrid_rerank": "RRF + 重排"}


@dataclass
class Hit:
    doc: Doc
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # 各路检索中的名次（从 1 开始），用于解释和调试

    @property
    def id(self) -> str:
        return self.doc.id


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, i in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + r)
    return sorted(scores.items(), key=lambda x: -x[1])


def _match(meta_value, cond) -> bool:
    if isinstance(cond, (list, tuple, set)):
        return meta_value in cond
    return meta_value == cond


class Retriever:
    def __init__(self, docs: list[Doc], embedder: Embedder | None = None, reranker: Reranker | None = None,
                 candidates: int = 20, rrf_k: int = 60):
        self.docs = docs
        self.embedder, self.reranker = embedder, reranker
        self.candidates, self.rrf_k = candidates, rrf_k
        texts = [d.text for d in docs]
        self.bm25 = BM25(texts)
        self.vec = VectorIndex(texts, embedder) if embedder else None
        self.last_timings: dict[str, float] = {}

    def available_modes(self) -> list[str]:
        modes = ["bm25"]
        if self.vec:
            modes += ["vector", "hybrid"]
            if self.reranker:
                modes.append("hybrid_rerank")
        return modes

    def _allowed(self, where: dict | None) -> set[int] | None:
        if not where:
            return None
        out = set()
        for i, d in enumerate(self.docs):
            ok = True
            for key, cond in where.items():
                value = d.kind if key == "kind" else d.meta.get(key)
                if not _match(value, cond):
                    ok = False
                    break
            if ok:
                out.add(i)
        return out

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", where: dict | None = None) -> list[Hit]:
        if mode not in self.available_modes():
            raise ValueError(f"检索模式 {mode} 不可用（可用：{self.available_modes()}）："
                             "向量模式需要配置 embedder，重排模式还需要 reranker")
        t: dict[str, float] = {}
        allowed = self._allowed(where)
        ranks: dict[int, dict[str, int]] = {}

        def timed(name, fn):
            s = time.perf_counter()
            r = fn()
            t[name] = round((time.perf_counter() - s) * 1000, 1)
            return r

        lists: list[list[int]] = []
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            b = timed("bm25", lambda: self.bm25.scores(query, allowed))[: self.candidates]
            lists.append([i for i, _ in b])
            for r, (i, _) in enumerate(b, 1):
                ranks.setdefault(i, {})["bm25"] = r
            if mode == "bm25":
                fused = b
        if mode in ("vector", "hybrid", "hybrid_rerank"):
            v = timed("vector", lambda: self.vec.scores(query, allowed))[: self.candidates]
            lists.append([i for i, _ in v])
            for r, (i, _) in enumerate(v, 1):
                ranks.setdefault(i, {})["vector"] = r
            if mode == "vector":
                fused = v
        if mode in ("hybrid", "hybrid_rerank"):
            fused = rrf(lists, self.rrf_k)[: self.candidates]
        if mode == "hybrid_rerank" and fused:
            idx = [i for i, _ in fused]
            s = timed("rerank", lambda: self.reranker.scores(query, [self.docs[i].text for i in idx]))
            fused = sorted(zip(idx, s), key=lambda x: -x[1])
            for r, (i, _) in enumerate(fused, 1):
                ranks.setdefault(i, {})["rerank"] = r
        self.last_timings = t
        return [Hit(self.docs[i], sc, ranks.get(i, {})) for i, sc in fused[:k]]
