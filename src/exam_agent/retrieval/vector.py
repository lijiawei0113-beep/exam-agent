"""向量检索：Embedder 接口 + 三种实现 + 磁盘缓存。

- FastEmbedEmbedder：本地 ONNX 推理（fastembed），默认 BAAI/bge-small-zh-v1.5。不依赖 torch，
  首次使用自动下载模型（约 100 MB）到 data/models/，之后完全离线。
- ApiEmbedder：任何 OpenAI 兼容的 /embeddings 接口（如硅基流动的 BAAI/bge-m3）。
- HashEmbedder：把字二元组哈希到固定维度。没有语义能力，只用于离线测试和流程演示。

文档向量按 (模型名, 文本哈希) 缓存在 data/emb_cache/，知识库不变就不重复计算。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import PROJECT_ROOT
from .text import tokenize

CACHE_DIR = PROJECT_ROOT / "data" / "emb_cache"
MODEL_DIR = PROJECT_ROOT / "data" / "models"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class HashEmbedder:
    name = "hash-bigram-256"
    semantic = False

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                m[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return _unit(m)


class FastEmbedEmbedder:
    semantic = True

    def __init__(self, model: str = "BAAI/bge-small-zh-v1.5", query_prefix: str = ""):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # 给出可操作的提示，而不是一串堆栈
            raise RuntimeError("没有安装 fastembed：请运行 pip install fastembed，"
                               "或在 .env 里设置 EXAM_EMBEDDER=api 改用接口") from e
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.name = model
        self.query_prefix = query_prefix
        self._m = TextEmbedding(model_name=model, cache_dir=str(MODEL_DIR))

    def embed(self, texts: list[str]) -> np.ndarray:
        return _unit(np.array(list(self._m.embed(texts)), dtype=np.float32))


class ApiEmbedder:
    semantic = True

    def __init__(self, model: str, base_url: str, api_key: str):
        from openai import OpenAI

        if not api_key:
            raise RuntimeError("EXAM_EMBED_API_KEY 为空：请在 .env 中填写向量接口的 Key")
        self.name = model
        self._c = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):  # 分批，避免单次请求过大
            r = self._c.embeddings.create(model=self.name, input=texts[i:i + 32])
            out.extend(d.embedding for d in r.data)
        return _unit(np.array(out, dtype=np.float32))


class CachedEmbedder:
    """包一层磁盘缓存：文档向量只算一次。

    查询不走缓存：线上每次查询都不同，缓存查询会让评测里的延迟偏低（实验 3 第一次运行就踩了这个坑：
    “仅向量”先跑时把查询向量存进了缓存，后面的“RRF 混合”直接命中，延迟显示为 0.8 ms）。
    """

    def __init__(self, inner: Embedder, cache_dir: Path = CACHE_DIR):
        self.inner = inner
        self.name = inner.name
        self.semantic = getattr(inner, "semantic", True)
        self.query_prefix = getattr(inner, "query_prefix", "")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", inner.name)
        self.path = cache_dir / f"{slug}.json"
        self._cache: dict[str, list[float]] = {}
        if self.path.exists():
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _key(t: str) -> str:
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self._cache]
        if missing:
            vecs = self.inner.embed(missing)
            for t, v in zip(missing, vecs):
                self._cache[self._key(t)] = [round(float(x), 6) for x in v]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._cache), encoding="utf-8")
        return np.array([self._cache[self._key(t)] for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed([self.query_prefix + text])[0]


class VectorIndex:
    def __init__(self, texts: list[str], embedder: Embedder):
        self.embedder = embedder
        self.matrix = embedder.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)

    def scores(self, query: str, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if hasattr(self.embedder, "embed_query"):
            q = self.embedder.embed_query(query)
        else:
            q = self.embedder.embed([query])[0]
        sims = self.matrix @ q
        idx = range(len(sims)) if allowed is None else allowed
        out = [(i, float(sims[i])) for i in idx]
        out.sort(key=lambda x: -x[1])
        return out
