"""BM25（Okapi）关键词检索，自己实现，便于讲清楚每一项的含义。

score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
- IDF：越少文档包含的词越重要（“去分母”比“方程”更能区分知识点）
- tf 饱和（k1）：一个词出现 10 次不比出现 2 次重要 5 倍
- 长度归一（b）：长文档天然包含更多词，要打折
"""
from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avgdl = sum(self.lens) / len(self.lens) if self.lens else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """返回 (文档下标, 分数)，按分数降序，只包含分数 > 0 的文档。"""
        q = set(tokenize(query))
        out = []
        for i, c in enumerate(self.docs):
            if allowed is not None and i not in allowed:
                continue
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
            for t in q:
                tf = c.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            if s > 0:
                out.append((i, s))
        out.sort(key=lambda x: -x[1])
        return out
