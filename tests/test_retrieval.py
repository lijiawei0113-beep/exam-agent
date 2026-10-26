"""阶段四：知识库与混合检索（离线：用哈希向量和假重排器，不下载模型）。"""
import numpy as np

from exam_agent.experiments.retrieval_eval import load_rcases, run_retrieval_eval
from exam_agent.retrieval import CachedEmbedder, HashEmbedder, Retriever, load_kb, rrf
from exam_agent.retrieval.bm25 import BM25
from exam_agent.retrieval.kb import Doc, Scope
from exam_agent.retrieval.text import tokenize

KB = load_kb()


def test_kb_is_consistent():
    assert KB.check() == []
    assert set(KB.units("七上")) == {1, 2, 3, 4, 5}
    assert len(KB.points_in(Scope(["七上"], 5))) >= 10


def test_out_of_scope_is_relative_to_target_unit():
    # 同一个“去分母”：对第五章的卷子是本单元，对第四章的卷子就是超纲
    u5, u4 = Scope(["七上"], 5), Scope(["七上"], 4)
    assert KB.relation("M7A-05-07", u5) == "current"
    assert KB.relation("M7A-05-07", u4) == "later"
    assert KB.relation("M7A-04-05", u5) == "prior"   # 前置知识可以用
    assert KB.relation("L-SYS", u5) == "later"
    assert KB.relation("L-SYS", Scope(["七上", "七下"])) == "current"   # 七年级全年卷：方程组不算超纲
    assert "方程组" in KB.later_signals(u5) and "方程组" not in KB.later_signals(Scope(["七上", "七下"]))
    assert KB.match_unit("一元一次方程（七年级上册）") == 5
    assert KB.match_unit("几何") is None             # 匹配不到就不猜


def test_tokenize_bigrams_and_fullwidth():
    assert tokenize("去分母") == ["去分", "分母"]
    assert tokenize("ｘ＋１") == tokenize("x+1")      # 全角转半角
    assert "移" in tokenize("移")


def test_bm25_prefers_rare_terms():
    bm = BM25(["解方程 去分母", "解方程 移项", "解方程 去括号"])
    ranked = bm.scores("怎么去分母解方程")
    assert ranked[0][0] == 0
    assert bm.idf["分母"] > bm.idf["方程"]           # 人人都有的词不重要
    assert [i for i, _ in bm.scores("去分母", allowed={1, 2})] == []


def test_rrf_rewards_agreement():
    fused = dict(rrf([[1, 2, 3], [2, 4, 1]], k=60))
    assert fused[2] > fused[1] > fused[4]            # 两路都靠前的最好
    assert abs(fused[2] - (1 / 62 + 1 / 61)) < 1e-12


class CountingEmbedder:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return HashEmbedder().embed(texts)


def test_embedding_cache(tmp_path):
    inner = CountingEmbedder()
    e = CachedEmbedder(inner, cache_dir=tmp_path)
    e.embed(["甲", "乙", "甲"])
    assert inner.calls == 2
    e2 = CachedEmbedder(inner, cache_dir=tmp_path)   # 重新加载：全部来自磁盘
    v = e2.embed(["乙"])
    assert inner.calls == 2 and np.isclose(np.linalg.norm(v[0]), 1, atol=1e-4)
    e2.embed_query("甲")
    e2.embed_query("甲")
    assert inner.calls == 4                           # 查询不缓存，评测延迟才真实


class ReverseReranker:
    name = "reverse"

    def scores(self, query, texts):
        return [-i for i in range(len(texts))][::-1]  # 把原顺序倒过来


def test_retriever_modes_filter_and_rerank():
    r = Retriever(KB.docs(), HashEmbedder(), ReverseReranker())
    assert r.available_modes() == ["bm25", "vector", "hybrid", "hybrid_rerank"]
    hits = r.search("去分母", 3, "hybrid", where={"kind": "kp"})
    assert hits[0].id == "M7A-05-07" and "bm25" in hits[0].ranks
    assert all(h.doc.kind == "kp" for h in hits)
    only_u4 = r.search("去分母", 5, "bm25", where={"unit": 4, "book": "七上"})
    assert all(h.doc.meta["unit"] == 4 for h in only_u4)
    fused = r.search("去分母", 20, "hybrid", where={"kind": "kp"})
    reranked = r.search("去分母", 20, "hybrid_rerank", where={"kind": "kp"})
    assert [h.id for h in reranked] == [h.id for h in fused][::-1]


def test_unavailable_mode_is_explicit():
    r = Retriever([Doc("a", "kp", "a", "去分母")])
    try:
        r.search("去分母", mode="vector")
        assert False, "没有向量模型时不应静默退化"
    except ValueError as e:
        assert "不可用" in str(e)


def test_retrieval_dataset_labels_exist():
    ids = {d.id: d.kind for d in KB.docs()}
    cases = load_rcases()
    assert len(cases) >= 60
    for c in cases:
        for g in c.gold:
            assert ids.get(g) == c.kind, f"{c.id} 的标注 {g} 不存在或类型不对"


def test_retrieval_eval_bm25_baseline(tmp_path):
    summary, (json_path, md_path) = run_retrieval_eval(Retriever(KB.docs()), ["bm25"], out_dir=tmp_path)
    assert json_path.exists() and md_path.exists() and (tmp_path / "retrieval_history.csv").exists()
    # 回归保护：术语原词和超纲查询 BM25 必须全部召回
    assert summary["bm25"]["per_tag"]["term"]["recall@5"] == 1.0
    assert summary["bm25"]["per_tag"]["out_of_scope"]["recall@5"] == 1.0
