"""课程知识库：知识点 / 单元卡片 + 课程标准条目。

组织方式
- 每张卡片属于一册书（book，如“三上”“七下”），有全局顺序 order：
    order = 年级 × 10000 + 册别（上 0 / 下 5000）+ 单元 × 100 + 序号
  所以“谁在前、谁在后”只看 order，跨年级也成立。
- 两种粒度：level="unit"（小学按单元一张卡）、level="point"（七上按知识点细分）。
- “超纲”不存成固定标签，而是相对于出卷范围计算：在范围内 = 本次考查内容；排在范围之前 = 前置知识，可以用；
  排在范围之后 = 超纲。同一个“去分母”，对七上第五章的卷子是本单元内容，对第四章的卷子就是超纲。

数据文件（改内容不用改代码）
  kb/<学科>/points/*.jsonl     卡片（每册一个文件）
  kb/<学科>/standard/*.jsonl   课标条目
  kb/<学科>/sources/toc.json   教材目录（构建小学单元卡片的依据，需人工核对）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..config import PROJECT_ROOT

KB_ROOT = PROJECT_ROOT / "kb"
SUBJECT_DIR = {"数学": "math", "英语": "english"}
GRADES = "一二三四五六七八九"
Relation = Literal["current", "prior", "later"]


def book_of(grade: str, term: str) -> str:
    return f"{grade[:1]}{term}"


def book_base(book: str) -> int:
    """一上 → 10000，三下 → 35000，七上 → 70000。"""
    return (GRADES.index(book[0]) + 1) * 10000 + (5000 if book[1] == "下" else 0)


def books_for(grade: str, term: str) -> list[str]:
    g = grade[:1]
    return [f"{g}上", f"{g}下"] if term == "全年" else [f"{g}{term}"]


class KnowledgePoint(BaseModel):
    id: str
    subject: str
    stage: str
    grade: str
    term: str
    edition: str
    unit: int | None          # 单元 / 章号；只作为“后续内容”边界占位的卡片为 None
    unit_name: str | None
    order: int
    name: str
    level: Literal["unit", "point"] = "point"
    aliases: list[str] = []
    description: str = ""
    key_points: list[str] = []  # 单元卡片的主要内容
    boundary: str = ""         # “不考什么”，交给命题和审核
    requires: list[str] = []   # 前置知识点
    standard_refs: list[str] = []
    signals: list[str] = []    # 本单元首次出现的专有术语；出现在更早范围的卷子里即为超纲信号
    vocabulary: list[str] = [] # 英语：本单元核心词汇（小写）
    patterns: list[str] = []   # 英语：本单元核心句型
    status: str = ""           # skeleton（只有目录）/ draft（模型起草，未抽检）/ checked（人工核对过）
    source: str = ""

    @property
    def book(self) -> str:
        return book_of(self.grade, self.term)

    def index_text(self) -> str:
        parts = [self.name, " ".join(self.aliases), self.description, "；".join(self.key_points),
                 "; ".join(self.patterns), " ".join(self.vocabulary)]
        if self.unit_name and self.unit_name != self.name:
            parts.append(self.unit_name)
        return "\n".join(p for p in parts if p)

    @property
    def location(self) -> str:
        return f"{self.grade}{self.term}册" + (f" {self.unit_name}" if self.unit_name else "")

    def brief(self) -> str:
        """给规划、命题、审核 Agent 看的一行说明。"""
        s = f"[{self.id}] {self.location}·{self.name}" if self.level == "point" else f"[{self.id}] {self.location}"
        if self.key_points:
            s += "：" + "、".join(self.key_points)
        elif self.description:
            s += "：" + self.description
        if self.patterns:
            s += "；句型：" + " / ".join(self.patterns[:4])
        if self.boundary:
            s += f"（不考：{self.boundary}）"
        return s


class StandardClause(BaseModel):
    id: str
    subject: str
    band: str
    domain: str
    topic: str
    text: str
    note: str = ""
    status: str = ""

    def index_text(self) -> str:
        return f"{self.topic}\n{self.text}\n{self.note}".strip()


@dataclass
class Doc:
    """检索的基本单位。meta 用于元数据过滤，obj 指回原始对象。"""
    id: str
    kind: Literal["kp", "standard"]
    title: str
    text: str
    meta: dict = field(default_factory=dict)
    obj: object = None


@dataclass
class Scope:
    """出卷范围：若干册书，可选限定到其中一个单元。"""
    books: list[str]
    unit: int | None = None

    def contains(self, p: KnowledgePoint) -> bool:
        return p.book in self.books and (self.unit is None or p.unit == self.unit)

    def label(self) -> str:
        return "、".join(f"{b[0]}年级{b[1]}册" for b in self.books) + (f"第{self.unit}单元" if self.unit else "")


@dataclass
class KnowledgeBase:
    subject: str
    points: list[KnowledgePoint]
    clauses: list[StandardClause]

    def __post_init__(self):
        self.points.sort(key=lambda p: p.order)
        self.by_id = {p.id: p for p in self.points}
        self.clause_by_id = {c.id: c for c in self.clauses}

    # ---- 范围与超纲关系 ----
    def books(self) -> list[str]:
        return sorted({p.book for p in self.points}, key=book_base)

    def units(self, book: str = "七上") -> dict[int, str]:
        return {p.unit: p.unit_name for p in self.points if p.book == book and p.unit is not None}

    def points_in(self, scope: Scope) -> list[KnowledgePoint]:
        return [p for p in self.points if scope.contains(p) and p.unit is not None]

    def relation(self, kp_id: str, scope: Scope) -> Relation:
        """知识点相对于出卷范围：本次范围内 / 前置（允许使用）/ 后续（超纲）。"""
        p = self.by_id[kp_id]
        if scope.contains(p):
            return "current"
        inside = [q.order for q in self.points if scope.contains(q)]
        if not inside:
            raise KeyError(f"知识库里没有范围 {scope.label()} 的内容")
        return "prior" if p.order < min(inside) else "later"

    def later_signals(self, scope: Scope) -> dict[str, KnowledgePoint]:
        """排在范围之后的内容的信号词 → 所属卡片。题干出现这些词，就有超纲嫌疑。"""
        inside = [q.order for q in self.points if scope.contains(q)]
        if not inside:
            return {}
        top = max(inside)
        out: dict[str, KnowledgePoint] = {}
        for p in self.points:
            if p.order > top:
                for s in p.signals:
                    out.setdefault(s, p)
        return out

    def vocabulary_upto(self, scope: Scope) -> set[str]:
        """英语：截至出卷范围末尾累计学过的词汇（前面各册 + 范围内各单元）。"""
        inside = [q.order for q in self.points if scope.contains(q)]
        if not inside:
            return set()
        top = max(inside)
        return {w.lower() for p in self.points if p.order <= top for w in p.vocabulary}

    def match_unit(self, topic: str, book: str = "七上") -> int | None:
        """从“一元一次方程”“整式的加减”这类主题文字确定单元。匹配不到返回 None，不猜。"""
        hits = [u for u, name in self.units(book).items() if name.split(" ", 1)[-1] in topic]
        return hits[0] if len(hits) == 1 else None

    # ---- 检索文档 ----
    def docs(self) -> list[Doc]:
        out = [Doc(p.id, "kp", p.name, p.index_text(),
                   {"subject": p.subject, "grade": p.grade, "term": p.term, "book": p.book, "unit": p.unit,
                    "order": p.order, "level": p.level}, p) for p in self.points]
        out += [Doc(c.id, "standard", f"课标·{c.topic}", c.index_text(), {"subject": c.subject}, c)
                for c in self.clauses]
        return out

    # ---- 完整性校验 ----
    def check(self) -> list[str]:
        problems: list[str] = []
        seen: set[str] = set()
        for p in self.points:
            if p.id in seen:
                problems.append(f"卡片 id 重复：{p.id}")
            seen.add(p.id)
        orders = [p.order for p in self.points]
        if len(set(orders)) != len(orders):
            problems.append("卡片 order 有重复")
        for p in self.points:
            if p.unit is not None and not (book_base(p.book) <= p.order < book_base(p.book) + 5000):
                problems.append(f"{p.id} 的 order {p.order} 不在 {p.book} 的范围内")
            for r in p.requires:
                if r not in self.by_id:
                    problems.append(f"{p.id} 的前置知识点 {r} 不存在")
                elif self.by_id[r].order >= p.order:
                    problems.append(f"{p.id} 的前置知识点 {r} 排在它后面（order 不对）")
            for s in p.standard_refs:
                if s not in self.clause_by_id:
                    problems.append(f"{p.id} 引用的课标条目 {s} 不存在")
            if p.unit is None and not p.signals:
                problems.append(f"后续内容 {p.id} 没有填写超纲信号词")
        # 信号词必须是“新出现的”：不能已经出现在更早卡片的名称或要点里，否则会把正常内容误判为超纲
        for p in self.points:
            for s in p.signals:
                for q in self.points:
                    earlier = f"{q.name}{''.join(q.key_points)}{''.join(q.aliases)}"
                    hit = (s.lower() in {w.lower() for w in q.vocabulary} | set(re.findall(r"[a-z']+", earlier.lower()))
                           if s.isascii() else s in earlier)
                    if q.order < p.order and q.unit is not None and hit:
                        problems.append(f"{p.id} 的信号词“{s}”在更早的 {q.id}（{q.name}）中已出现")
                        break
        return problems


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_kb(subject: str = "math", root: Path = KB_ROOT) -> KnowledgeBase:
    d = root / SUBJECT_DIR.get(subject, subject)
    if not d.exists():
        raise FileNotFoundError(f"找不到知识库目录 {d}")
    points = [KnowledgePoint(**r) for f in sorted((d / "points").glob("*.jsonl")) for r in _read_jsonl(f)]
    clauses = [StandardClause(**r) for f in sorted((d / "standard").glob("*.jsonl")) for r in _read_jsonl(f)]
    return KnowledgeBase(subject, points, clauses)
