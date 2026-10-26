"""阶段四：课程知识库与混合检索。"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from ..config import PROJECT_ROOT
from .hybrid import MODE_LABEL, MODES, Hit, Retriever, rrf
from .kb import Doc, KnowledgeBase, KnowledgePoint, StandardClause, load_kb
from .vector import ApiEmbedder, CachedEmbedder, FastEmbedEmbedder, HashEmbedder

__all__ = ["MODES", "MODE_LABEL", "Hit", "Retriever", "rrf", "Doc", "KnowledgeBase", "KnowledgePoint",
           "StandardClause", "load_kb", "make_embedder", "make_reranker", "build_retriever"]


def make_embedder(kind: str | None = None):
    """kind：fastembed（默认，本地）/ api / hash（无语义，仅测试）/ none。"""
    load_dotenv(PROJECT_ROOT / ".env")
    kind = (kind or os.getenv("EXAM_EMBEDDER", "fastembed")).lower()
    if kind == "none":
        return None
    if kind == "hash":
        return HashEmbedder()
    if kind == "fastembed":
        inner = FastEmbedEmbedder(os.getenv("EXAM_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
                                  os.getenv("EXAM_EMBED_QUERY_PREFIX", ""))
    elif kind == "api":
        inner = ApiEmbedder(os.getenv("EXAM_EMBED_API_MODEL", "BAAI/bge-m3"),
                            os.getenv("EXAM_EMBED_API_BASE", "https://api.siliconflow.cn/v1"),
                            os.getenv("EXAM_EMBED_API_KEY", ""))
    else:
        raise ValueError(f"未知的 EXAM_EMBEDDER={kind}，可选 fastembed / api / hash / none")
    return CachedEmbedder(inner)


def make_reranker(kind: str | None = None):
    load_dotenv(PROJECT_ROOT / ".env")
    kind = (kind or os.getenv("EXAM_RERANKER", "none")).lower()  # 实验 3：重排在短卡片上反而变差，默认关闭
    if kind == "none":
        return None
    if kind == "overlap":
        from .rerank import OverlapReranker
        return OverlapReranker()
    if kind == "fastembed":
        from .rerank import FastEmbedReranker
        return FastEmbedReranker(os.getenv("EXAM_RERANK_MODEL", "BAAI/bge-reranker-base"))
    raise ValueError(f"未知的 EXAM_RERANKER={kind}，可选 fastembed / none")


def build_retriever(kb: KnowledgeBase, embedder="env", reranker="env") -> Retriever:
    emb = make_embedder() if embedder == "env" else embedder
    rer = make_reranker() if reranker == "env" else reranker
    return Retriever(kb.docs(), emb, rer)
