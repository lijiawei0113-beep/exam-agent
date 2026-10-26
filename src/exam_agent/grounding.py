"""把课程知识库接入出卷流程（检索增强）。

1. 确定范围：年级 + 册别（老师原话）→ 若干册书；老师点名了单元（“第二单元”）或知识点（“一元一次方程”）时，
   用混合检索在这几册里找到对应单元，进一步缩小范围。
2. 候选知识点：范围内的卡片交给规划 Agent，蓝图的每道题必须选其中一个 kp_id —— 由代码校验，选错就打回。
3. 范围说明：所选卡片的主要内容和“不考什么”写进蓝图 slot，命题和审核都能看到，并随试卷版本保存。
4. 超纲信号：范围之后才出现的专有术语（如三年级上册卷里的“小数点”），题干一出现就由代码直接报告。
知识库没有覆盖的学科或年级，如实返回 None，流程退回到由模型自行判断范围，并在假设中说明。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .domain import ExamRequest
from .retrieval import Retriever
from .retrieval.kb import SUBJECT_DIR, KnowledgeBase, KnowledgePoint, Scope, books_for, load_kb

_CN_NUM = {c: i for i, c in enumerate("一二三四五六七八九十", start=1)}
_UNIT = re.compile(r"第\s*([0-9]+|[一二三四五六七八九十])\s*(?:单元|章)")


@dataclass
class Candidates:
    scope: Scope
    points: list[KnowledgePoint]
    how: str                                   # 范围是怎么确定的（写进假设和运行轨迹）
    signals: dict[str, str] = field(default_factory=dict)   # 超纲信号词 → 来源说明
    vocabulary: list[str] = field(default_factory=list)     # 英语：截至范围末尾已学词汇

    @property
    def ids(self) -> set[str]:
        return {p.id for p in self.points}


def parse_unit(text: str) -> int | None:
    m = _UNIT.search(text)
    if not m:
        return None
    v = m.group(1)
    return int(v) if v.isdigit() else _CN_NUM[v]


class Grounding:
    def __init__(self, kb: KnowledgeBase, retriever: Retriever | None = None, mode: str | None = None):
        self.kb = kb
        self.retriever = retriever or Retriever(kb.docs())
        self.mode = mode or ("hybrid" if "hybrid" in self.retriever.available_modes() else "bm25")

    @classmethod
    def from_env(cls, subject: str = "math") -> "Grounding":  # 单学科，保留给测试和脚本使用
        """读取 .env 中的向量模型配置；模型加载失败时只用 BM25（实验 3：BM25 对术语和超纲内容已足够准）。"""
        from .retrieval import make_embedder

        kb = load_kb(subject)
        try:
            emb = make_embedder()
        except Exception:
            emb = None
        return cls(kb, Retriever(kb.docs(), emb, None))

    def candidates(self, req: ExamRequest, teacher_text: str = "") -> Candidates | None:
        if SUBJECT_DIR.get(req.subject) != self.kb.subject and req.subject != self.kb.subject:
            return None
        books = [b for b in books_for(req.grade, req.term) if b in self.kb.books()]
        if not books:
            return None
        scope = Scope(books)
        how = f"按老师说的年级和册别，范围为 {scope.label()}"

        unit = parse_unit(f"{req.topic} {teacher_text}")
        if unit is not None and len(books) == 1:
            if any(p.unit == unit for p in self.kb.points_in(Scope(books, unit))):
                scope, how = Scope(books, unit), f"老师指定 {Scope(books, unit).label()}"
        elif req.topic:
            hits = self.retriever.search(req.topic, 3, self.mode, where={"kind": "kp", "book": books})
            hits = [h for h in hits if h.doc.meta.get("unit") is not None]
            if hits:
                top = hits[0].doc.obj
                scope = Scope([top.book], top.unit)
                how = (f"检索“{req.topic}”（{self.mode}）定位到 {top.location}"
                       + (f"·{top.name}" if top.level == "point" else ""))
            else:
                how += f"；没有检索到与“{req.topic}”对应的单元，范围内全部单元都可选"

        points = self.kb.points_in(scope)
        if not points:
            return None
        signals = {} if self.kb.subject == "english" else \
            {s: f"{p.id} {p.location}·{p.name}" for s, p in self.kb.later_signals(scope).items()}
        return Candidates(scope, points, how, signals, sorted(self.kb.vocabulary_upto(scope)))

    def explain_missing(self, req: ExamRequest) -> str:
        return "课程知识库没有覆盖这个学科或年级，考查范围由模型自行判断，未经知识库约束"


class MultiGrounding:
    """按学科分派到各自的知识库（数学、英语），共用一个向量模型。"""

    def __init__(self, by_subject: dict[str, Grounding]):
        self.by_subject = by_subject

    @classmethod
    def from_env(cls) -> "MultiGrounding":
        from .retrieval import make_embedder
        from .retrieval.kb import KB_ROOT

        try:
            emb = make_embedder()
        except Exception:
            emb = None   # 向量模型不可用时只用 BM25（实验 3：BM25 对术语和超纲内容已足够准）
        out = {}
        for name, d in SUBJECT_DIR.items():
            if (KB_ROOT / d / "points").exists():
                kb = load_kb(d)
                if kb.points:
                    out[name] = Grounding(kb, Retriever(kb.docs(), emb, None))
        return cls(out)

    @property
    def mode(self) -> str:
        return next(iter(self.by_subject.values())).mode if self.by_subject else "-"

    def size(self) -> str:
        return "，".join(f"{k} {len(g.kb.points)} 张" for k, g in self.by_subject.items())

    def candidates(self, req: ExamRequest, teacher_text: str = "") -> Candidates | None:
        g = self.by_subject.get(req.subject)
        return g.candidates(req, teacher_text) if g else None

    def explain_missing(self, req: ExamRequest) -> str:
        if req.subject == "英语" and req.grade[:1] in "一二":
            return ("英语一般从三年级开设，一二年级以听说活动为主，且小学一二年级不进行纸笔考试；"
                    "知识库没有覆盖，考查范围由模型自行判断")
        return "课程知识库没有覆盖这个学科或年级，考查范围由模型自行判断，未经知识库约束"


def find_signals(text: str, signals: dict[str, str]) -> list[tuple[str, str]]:
    return [(s, src) for s, src in signals.items() if s and s in text]
