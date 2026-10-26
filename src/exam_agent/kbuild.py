"""课程知识库的半自动构建流水线（数学一至六年级、英语三至七年级）。

  ① 目录（人工核对）  kb/<学科>/sources/toc.json，只有单元标题，是唯一需要人提供的事实
  ② 骨架卡片（代码）   kb init：按目录生成每个单元一张卡片，id、顺序、所属册别都由代码决定
  ③ 起草（模型）       kb build：每册调用一次模型，补写主要内容、考查边界、别名、信号词、课标依据
  ④ 校验（代码）       单元数与目录一致、标题一字不差、课标条目存在且学段匹配、信号词必须是“新出现的”……
                      不合格就带着具体问题让模型重写，最多 3 次
  ⑤ 抽检（人工）       kb sample 随机抽取卡片生成核对表，老师/作者填写对错；kb audit-report 统计准确率

模型只做它擅长的“概括”，事实骨架（有哪些单元、谁先谁后）来自人工核对过的目录，由代码保证。
"""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .agents.base import Tracer, structured_call
from .config import PROJECT_ROOT
from .llm import LLMClient
from .retrieval.kb import GRADES, KB_ROOT, KnowledgeBase, KnowledgePoint, book_base, load_kb

MATH = KB_ROOT / "math"
TOC_PATH = MATH / "sources" / "toc.json"
AUDIT_DIR = MATH / "audit"


@dataclass(frozen=True)
class SubjectSpec:
    key: str                 # 目录名：math / english
    name: str                # 数学 / 英语
    prefix: str              # 卡片 id 前缀
    bands: dict[int, str]    # 年级 → 课标学段（课标条目的 band 以此开头）

    @property
    def root(self) -> Path:
        return KB_ROOT / self.key


SPECS = {
    "math": SubjectSpec("math", "数学", "M", {1: "第一学段", 2: "第一学段", 3: "第二学段", 4: "第二学段",
                                              5: "第三学段", 6: "第三学段"}),
    "english": SubjectSpec("english", "英语", "E", {3: "一级", 4: "一级", 5: "二级", 6: "二级",
                                                    7: "三级", 8: "三级", 9: "三级"}),
}
BAND = SPECS["math"].bands


def load_toc(path: Path | None = None, subject: str = "math") -> dict:
    path = path or SPECS[subject].root / "sources" / "toc.json"
    return json.loads(path.read_text(encoding="utf-8"))


def grade_no(book: str) -> int:
    return GRADES.index(book[0]) + 1


def unit_id(book: str, unit: int, prefix: str = "M") -> str:
    return f"{prefix}{grade_no(book)}{'A' if book[1] == '上' else 'B'}-{unit:02d}"


def skeleton_cards(book: str, units: list[str], edition: str, subject: str = "math") -> list[KnowledgePoint]:
    spec, g = SPECS[subject], grade_no(book)
    return [KnowledgePoint(
        id=unit_id(book, u, spec.prefix), subject=spec.name, stage="小学" if g <= 6 else "初中",
        grade=f"{book[0]}年级", term=book[1], edition=edition, unit=u,
        unit_name=name if re.match(r"(Starter )?Unit", name) else f"第{u}单元 {name}",
        order=book_base(book) + u * 100, name=name, level="unit", status="skeleton", source="教材目录（toc.json）",
    ) for u, name in enumerate(units, start=1)]


def points_path(book: str, root: Path = MATH) -> Path:
    return root / "points" / f"{book}.jsonl"


def write_cards(cards: list[KnowledgePoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cards:
            f.write(c.model_dump_json(exclude_none=False) + "\n")


def init_skeletons(toc: dict | None = None, root: Path | None = None, force: bool = False,
                   subject: str = "math") -> list[str]:
    """为目录中的每册生成骨架卡片。已经起草或核对过的册不覆盖（除非 force）。返回写入的册。"""
    toc = toc or load_toc(subject=subject)
    root = root or SPECS[subject].root
    written = []
    for book, units in toc["books"].items():
        path = points_path(book, root)
        if path.exists() and not force:
            existing = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if any(r.get("status") != "skeleton" for r in existing):
                continue
        write_cards(skeleton_cards(book, units, toc["edition"], subject), path)
        written.append(book)
    return written


# ---------------- ③④ 模型起草 + 代码校验 ----------------

class UnitDraft(BaseModel):
    unit: int
    name: str
    description: str = Field(description="一两句话概括本单元学什么")
    key_points: list[str] = Field(description="2～8 条主要内容，每条一个短语")
    boundary: str = Field(description="本单元不考什么（尤其是后续单元、后续年级才学的内容）")
    aliases: list[str] = Field(default_factory=list, description="老师可能的其他叫法")
    signals: list[str] = Field(default_factory=list, description="本单元首次出现的专有术语，0～4 个")
    standard_refs: list[str] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list, description="英语：本单元核心词汇")
    patterns: list[str] = Field(default_factory=list, description="英语：本单元核心句型")


class BookDraft(BaseModel):
    units: list[UnitDraft]


SYSTEM = """你是熟悉人教版小学数学教材和《义务教育数学课程标准（2022年版）》的教研员。
现在要为一册教材的每个单元写一张“单元卡片”，供出卷系统判断考查范围。
要求：
1. 按给出的单元目录逐个写，unit 和 name 与目录完全一致，不增不减。
2. key_points 写本单元的主要学习内容（2～8 条短语），只写这一册这一单元真正学的内容，不要写后面才学的。
3. boundary 写本单元“不考什么”，只写容易被误出到这一单元的后续内容，例如“不涉及两位数除法”“不含小数”；
   不要写前面单元的内容（前面学过的内容都可以用）。
4. aliases 写老师可能的其他说法（如“表内除法”也叫“用口诀求商”），0～4 个。
5. signals 写本单元第一次出现、之前各册和本册前面各单元都没有的专有术语（如“余数”“周长”“面积”“小数点”），
   0～4 个，每个至少两个字；拿不准是不是第一次出现就不要写。它们将用于发现“更低年级的卷子里出现了本单元内容”。
6. standard_refs 从给出的课标条目 id 中选择对应的条目（1～2 个）。
7. 只输出 json：{"units": [{"unit": 1, "name": "...", "description": "...", "key_points": ["..."],
   "boundary": "...", "aliases": [], "signals": [], "standard_refs": ["P2-NUM"]}]}"""


SYSTEM_EN = """你是熟悉人教版英语教材（小学 PEP、初中 Go for it!）和《义务教育英语课程标准（2022年版）》的教研员。
现在要为一册教材的每个单元写一张“单元卡片”，供出卷系统判断考查范围。
要求：
1. 按给出的单元目录逐个写，unit 和 name 与目录完全一致，不增不减。
2. description 和 key_points 用中文写本单元的话题和主要学习内容（2～8 条短语，如“询问并回答时间”“一般过去时的肯定句”）。
3. vocabulary 写本单元要求掌握的核心单词或词组，全部小写英文（5～40 个），只写这一单元新学的，不要写之前学过的。
4. patterns 写本单元的核心句型（1～6 个），如 "What time is it? It's ..."。
5. boundary 用中文写本单元“不考什么”，只写后续单元、后续年级才学的语法，如“不涉及一般过去时”；
   不要写前面单元的内容（前面学过的内容都可以用）。
6. aliases 写老师可能的其他说法（中文或英文），0～4 个。
7. signals 留空列表（英语不按单词判断超纲；语法级别和阅读难度由程序按课标检查）。
8. standard_refs 从给出的课标条目 id 中选择对应的条目（1～2 个）。
9. 只输出 json：{"units": [{"unit": 1, "name": "...", "description": "...", "key_points": ["..."], "vocabulary": ["..."],
   "patterns": ["..."], "boundary": "...", "aliases": [], "signals": [], "standard_refs": ["E1-VOC"]}]}"""

MAX_VOCAB = 40
_EN_WORD = re.compile(r"^[a-z][a-z'\- ]*[a-z.!?]?$")


def _earlier_text(kb: KnowledgeBase, book: str) -> str:
    base = book_base(book)
    return "".join(f"{p.name}{''.join(p.key_points)}{''.join(p.aliases)} {' '.join(p.vocabulary)} "
                   for p in kb.points if p.order < base)


def validate_book(d: BookDraft, book: str, units: list[str], kb: KnowledgeBase, subject: str = "math") -> list[str]:
    problems = []
    if [u.name for u in d.units] != units:
        problems.append(f"单元必须与目录完全一致（{len(units)} 个，按顺序）：{units}；你给出的是：{[u.name for u in d.units]}")
        return problems
    band = SPECS[subject].bands.get(grade_no(book), "")
    allowed = {c.id for c in kb.clauses if c.band.startswith(band)}
    earlier = _earlier_text(kb, book)
    english = subject == "english"
    if english:  # 英语信号词按“单词”比较，避免 "go" 命中 "good"
        earlier_words = set(re.findall(r"[a-z']+", earlier.lower()))
    for i, u in enumerate(d.units, start=1):
        tag = f"第{i}单元“{u.name}”"
        if u.unit != i:
            problems.append(f"{tag} 的 unit 应为 {i}")
        if not 2 <= len(u.key_points) <= 8:
            problems.append(f"{tag} 的 key_points 应为 2～8 条，现在 {len(u.key_points)} 条")
        if not u.boundary.strip():
            problems.append(f"{tag} 缺少 boundary（不考什么）")
        bad = [r for r in u.standard_refs if r not in allowed]
        if bad or not u.standard_refs:
            problems.append(f"{tag} 的 standard_refs 必须从 {sorted(allowed)} 中选 1～2 个，{bad or '现在为空'} 不可用")
        if english:
            if len(u.vocabulary) < 5:   # 多于 40 个由代码截断，不必让模型重写（六下 Unit 4 实测给了 41 个，3 次都没改对）
                problems.append(f"{tag} 的 vocabulary 至少 5 个，现在 {len(u.vocabulary)} 个")
            bad_words = [w for w in u.vocabulary if not _EN_WORD.match(w)]
            if bad_words:
                problems.append(f"{tag} 的 vocabulary 必须是小写英文单词或词组：{bad_words[:5]} 不合格")
            if not 1 <= len(u.patterns) <= 6:
                problems.append(f"{tag} 的 patterns 应为 1～6 个")
            same_words = {w for x in d.units[:i - 1] for w in " ".join(x.vocabulary).lower().split()}
            not_en = [s for s in u.signals if not re.fullmatch(r"[A-Za-z' \-]+", s)]
            old = [s for s in u.signals if s.lower() in earlier_words | same_words]
            if not_en:
                problems.append(f"{tag} 的信号词必须是英文：{not_en}")
            if old:
                problems.append(f"{tag} 的信号词 {old} 在之前的册或本册前面的单元已经出现过，请删掉")
            continue
        short = [s for s in u.signals if len(s) < 2]
        if short:
            problems.append(f"{tag} 的信号词 {short} 只有一个字，容易一词多义（如人民币的“角”和图形的“角”），请删掉或换成更具体的术语")
        same_book = "".join(f"{x.name}{''.join(x.key_points)}{''.join(x.aliases)}" for x in d.units[:i - 1])
        old = [s for s in u.signals if s in earlier or s in same_book]
        if old:
            problems.append(f"{tag} 的信号词 {old} 在之前的年级或本册前面的单元已经出现过，不是本单元首次出现的术语，请删掉")
    return problems


def build_book(llm: LLMClient, book: str, kb: KnowledgeBase, toc: dict | None = None,
               tracer: Tracer | None = None, model_name: str = "", subject: str = "math") -> list[KnowledgePoint]:
    toc = toc or load_toc(subject=subject)
    units = toc["books"][book]
    band = SPECS[subject].bands.get(grade_no(book), "")
    clauses = [c for c in kb.clauses if c.band.startswith(band)]
    user = "\n".join([
        f"教材：{toc['edition']} {book[0]}年级{book[1]}册",
        "单元目录：", *[f"{i}. {n}" for i, n in enumerate(units, start=1)],
        f"可引用的课标条目（{band}）：", *[f"- {c.id}（{c.topic}）{c.text}" for c in clauses],
    ])
    draft = structured_call(llm, step="kb_build", system=SYSTEM_EN if subject == "english" else SYSTEM, user=user,
                            schema=BookDraft, validate=lambda d: validate_book(d, book, units, kb, subject),
                            tracer=tracer).value
    cards = skeleton_cards(book, units, toc["edition"], subject)
    stamp = datetime.now().strftime("%Y-%m-%d")
    return [c.model_copy(update={
        "description": u.description.strip(), "key_points": [k.strip() for k in u.key_points],
        "boundary": u.boundary.strip(), "aliases": u.aliases, "signals": u.signals,
        "standard_refs": u.standard_refs, "status": "draft",
        "vocabulary": [w.strip().lower() for w in u.vocabulary][:MAX_VOCAB],
        "patterns": [x.strip() for x in u.patterns],
        **({"signals": []} if subject == "english" else {}),   # 英语不按单词判超纲，信号词一律不用
        "source": f"目录（toc.json）+ 模型起草（{model_name or llm.model}，{stamp}），未人工抽检",
    }) for c, u in zip(cards, draft.units)]


def build_books(llm: LLMClient, books: list[str], progress=print, root: Path | None = None,
                tracer: Tracer | None = None, subject: str = "math") -> dict[str, str]:
    """按年级顺序逐册起草：后面的册校验信号词时，能看到前面已经起草好的内容。"""
    toc = load_toc(subject=subject)
    root = root or SPECS[subject].root
    result = {}
    for book in sorted(books, key=book_base):
        kb = load_kb(subject, root.parent)
        progress(f"起草 {book}（{len(toc['books'][book])} 个单元）...")
        try:
            cards = build_book(llm, book, kb, toc, tracer, subject=subject)
        except Exception as e:  # 单册失败不影响其他册，骨架卡片保留
            result[book] = f"失败：{e}"
            progress(f"  {book} 失败：{e}")
            continue
        write_cards(cards, points_path(book, root))
        problems = load_kb(subject, root.parent).check()
        result[book] = "完成" if not problems else f"完成，但知识库校验有 {len(problems)} 个问题"
        progress(f"  {result[book]}")
    return result


# ---------------- ⑤ 人工抽检 ----------------

AUDIT_HEADER = ["id", "位置", "单元", "主要内容", "不考什么", "信号词", "结论（填 对 / 错）", "错在哪里"]


def sample_for_audit(kb: KnowledgeBase, n: int = 15, seed: int | None = None, out_dir: Path | None = None) -> Path:
    """从模型起草（draft）的单元卡片中随机抽取 n 张，生成核对表（Excel 可直接打开）。"""
    pool = [p for p in kb.points if p.level == "unit" and p.status == "draft"]
    if not pool:
        raise ValueError("没有待抽检的卡片：请先运行 kb build 起草")
    out_dir = out_dir or KB_ROOT / kb.subject / "audit"
    rng = random.Random(seed)
    picked = sorted(rng.sample(pool, min(n, len(pool))), key=lambda p: p.order)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"audit_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(AUDIT_HEADER)
        for p in picked:
            w.writerow([p.id, p.location, p.name, "；".join(p.key_points), p.boundary, "、".join(p.signals), "", ""])
    return path


def audit_report(path: Path, kb: KnowledgeBase | None = None, mark_checked: bool = True,
                 root: Path = MATH) -> dict:
    """统计抽检结果；判为“对”的卡片标记为 checked（人工核对过）。"""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    verdict = AUDIT_HEADER[6]
    filled = [r for r in rows if r[verdict].strip() in ("对", "错")]
    wrong = [r for r in filled if r[verdict].strip() == "错"]
    report = {"file": path.name, "sampled": len(rows), "filled": len(filled), "wrong": len(wrong),
              "accuracy": (len(filled) - len(wrong)) / len(filled) if filled else None,
              "errors": [{"id": r["id"], "unit": r["单元"], "note": r[AUDIT_HEADER[7]]} for r in wrong]}
    if mark_checked and filled:
        ok = {r["id"] for r in filled if r[verdict].strip() == "对"}
        for book_file in (root / "points").glob("*.jsonl"):
            cards = [KnowledgePoint(**json.loads(l)) for l in book_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if any(c.id in ok for c in cards):
                write_cards([c.model_copy(update={"status": "checked"}) if c.id in ok else c for c in cards], book_file)
    log = (path.parent / "audit_history.csv")
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["时间", "抽检文件", "抽取", "已填写", "错误", "准确率"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), path.name, len(rows), len(filled), len(wrong),
                    "" if report["accuracy"] is None else f"{report['accuracy']:.3f}"])
    return report
