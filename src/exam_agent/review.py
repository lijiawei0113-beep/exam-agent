"""审核与返修闭环。

一轮审核：
  1. 盲解 Agent 只看题干作答（信息隔离）；
  2. 精确求解器计算盲解列出的方程，与命题员的答案比对；检查整数等可验证约束；
  3. 教学审核 Agent 只审新写的题（老师没让动的题不挑刺）。
返修：只把有 must_fix 问题的题带着具体问题交回命题员重出，再只审核这些题。

停止条件（显式记录在 report.stop_reason 中）：
  - passed：没有 must_fix 问题；
  - max_rounds：达到最大返修轮数；
  - no_progress：上一轮的问题一个都没解决（同一道题的同一类问题仍在），继续返修只是在烧 token。
    注意按“题 + 问题类型”判断，而不是按题数：实测第 7 题的算术错误修好了，却冒出“与第 6 题重复”，
    按题数算会误判为没有进展并停止（见 evals/real_defects.jsonl d016）。
没修好的问题不会被藏起来，而是作为 remaining 如实展示给老师。
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from .agents.base import AgentFailure, Tracer
from .agents.reviewer import review_pedagogy
from .agents.solver import BlindSolution, blind_solve
from .agents.writer import fill_slots
from .domain import Blueprint, ExamRequest, QType, Question, Verification, VStatus, integer_required
from .llm import LLMClient
from .tools.linear import extract_numbers, fmt
from .verify import route

MAX_REPAIR_ROUNDS = 2


class Finding(BaseModel):
    slot_id: str
    source: Literal["tool", "blind", "pedagogy", "kb"]
    severity: Literal["must_fix", "suggest"]
    code: str
    message: str


@dataclass
class ReviewReport:
    rounds: int = 0
    initial: list[Finding] = field(default_factory=list)
    remaining: list[Finding] = field(default_factory=list)
    suggestions: list[Finding] = field(default_factory=list)
    repaired: set[str] = field(default_factory=set)
    status: dict[str, Verification] = field(default_factory=dict)  # 每道被审核题的最终验证状态
    stop_reason: str = ""

    @property
    def tool_verified(self) -> set[str]:
        return {sid for sid, v in self.status.items() if v.status == VStatus.TOOL}

    @property
    def must_fix_initial(self) -> int:
        return len({f.slot_id for f in self.initial if f.severity == "must_fix"})


def check_question(q: Question, blind: BlindSolution | None, req: ExamRequest) -> tuple[list[Finding], Verification]:
    """单题核验：验证路由给出状态和工具结果，再叠加盲解发现的题干问题、选项不一致、整数约束。"""
    findings: list[Finding] = []

    def add(source, code, message, severity="must_fix"):
        findings.append(Finding(slot_id=q.slot_id, source=source, severity=severity, code=code, message=message))

    if blind is not None:
        if blind.issue.strip():
            add("blind", "ambiguous", f"独立作答时发现题干问题：{blind.issue.strip()}")
        if getattr(blind, "leak", "").strip():
            add("blind", "answer_leak", f"题干的中文提示直接给出了答案：{blind.leak.strip()}。请去掉翻译或改成只提示情境")
        amb = [a.strip() for a in getattr(blind, "ambiguous_items", []) if a.strip()]
        if amb:
            add("blind", "multiple_answers", "独立作答时发现不止一个选项能成立：" + "；".join(amb)
                + "。请修改题目或选项，保证只有一个正确答案")
        if q.qtype == QType.CHOICE and blind.choice.strip():
            picked = blind.choice.strip().upper()[:1]
            if picked != q.answer:
                add("blind", "choice_mismatch", f"独立作答选了 {picked}，但答案键是 {q.answer}，请核对题目和答案")

    r = route(q, blind)
    v = r.verification
    if r.mismatch:
        add(r.source, r.code, r.mismatch)
    elif r.tool_value is not None and v.status == VStatus.UNVERIFIED:
        add("tool", "answer_unparsed", v.detail, "suggest")

    if integer_required(req, q.qtype):
        values = [r.tool_value] if r.tool_value is not None else extract_numbers(q.answer)
        bad = [x for x in values if x.denominator != 1]
        if bad:
            add("tool", "non_integer", f"答案 {fmt(bad[0])} 不是整数，违反约束“答案为整数”")
        elif not values:
            add("tool", "integer_unverified", "未能确定答案数值，无法核验整数约束", "suggest")

    if any(f.severity == "must_fix" for f in findings):
        v = v.model_copy(update={"status": VStatus.PROBLEM,
                                 "detail": "；".join(f.message for f in findings if f.severity == "must_fix")})
    return findings, v


def check_one(q: Question, blind: BlindSolution | None, req: ExamRequest) -> tuple[list[Finding], bool]:
    """兼容旧接口：返回 (问题列表, 答案是否经工具独立核验)。"""
    findings, v = check_question(q, blind, req)
    return findings, v.status == VStatus.TOOL


_SELF_CLEARED = re.compile(r"无问题|没有问题|不构成|符合范围|符合要求|并无不妥|可以接受")
_WORD_SCOPE = re.compile(r"词汇|单词|词表|生词")


def pedagogy_severity(issue, req: ExamRequest) -> str:
    """把老师定下的审核政策落到代码里，不只靠提示词：
    - 审核意见自己写着“无问题 / 不构成错误”，却标成 must_fix → 降为建议（d029：白白返修一轮）；
    - 英语：超纲（语法级别、生词、短文长度）由代码按课标和老师给定的规则判定，审核 Agent 的“超纲 / 超出某单元范围”
      意见一律降为建议（d033：它把“出卷范围内其他单元的句型”判为超纲，如 Unit 2 的题里用了 has）。"""
    if issue.severity != "must_fix":
        return issue.severity
    if _SELF_CLEARED.search(issue.message):
        return "suggest"
    if req.subject == "英语" and (re.search(r"超纲|超出", issue.category + issue.message)
                                  and re.search(r"范围|Unit|单元|词汇|单词|句型|语法|时态", issue.message)):
        return "suggest"
    return "must_fix"


def review_round(llm: LLMClient, req: ExamRequest, bp: Blueprint, questions: dict[str, Question],
                 verify: set[str], pedagogy: set[str], tracer: Tracer | None
                 ) -> tuple[list[Finding], dict[str, Verification]]:
    findings: list[Finding] = []
    status: dict[str, Verification] = {}
    if verify:
        blind = blind_solve(llm, [questions[s] for s in sorted(verify)], tracer)
        for sid in sorted(verify):
            fs, v = check_question(questions[sid], blind.get(sid), req)
            findings += fs
            status[sid] = v
        if tracer:
            tracer.log("tool_check", status={sid: f"{v.status.value}｜{v.method}" for sid, v in status.items()},
                       blind={sid: {"eq": b.equations, "expr": b.expression, "ans": b.final_answer, "issue": b.issue}
                              for sid, b in blind.items()})
    if bp.signals and req.subject != "英语":   # 超纲信号词（英语不按单词判超纲）：范围之后才出现的专有术语，代码直接匹配，有据可查
        from .grounding import find_signals

        for sid in sorted(verify):
            q = questions[sid]
            for word, src in find_signals(q.stem + " ".join(q.options), bp.signals):
                findings.append(Finding(slot_id=sid, source="kb", severity="must_fix", code="out_of_scope_signal",
                                        message=f"题目出现“{word}”，这是 {src} 才学的内容，超出本卷范围"))
                if sid in status:
                    status[sid] = status[sid].model_copy(update={"status": VStatus.PROBLEM})
    if req.subject == "英语":   # 英语：语法按课标分级、阅读按年级控制难度；个别生词不判错
        from .agents.requirement import CN_GRADES
        from .tools.english import check_question as check_english, scope_end_book

        g = CN_GRADES.index(req.grade[:1]) + 1 if req.grade[:1] in CN_GRADES else 0
        end = scope_end_book(req.grade, req.term)
        for sid in sorted(verify):
            q = questions[sid]
            for i in check_english(q.qtype.value, q.stem, q.options, q.answer, g, end):
                findings.append(Finding(slot_id=sid, source="kb", severity="suggest" if i.severity == "info" else i.severity,
                                        code=i.code, message=i.message))
                if i.severity == "must_fix" and sid in status:
                    status[sid] = status[sid].model_copy(update={"status": VStatus.PROBLEM})
    if pedagogy:
        slots = {s.slot_id: s for s in bp.slots}
        for i in review_pedagogy(llm, req, slots, questions, pedagogy, tracer):
            findings.append(Finding(slot_id=i.slot_id, source="pedagogy", severity=pedagogy_severity(i, req),
                                    code=i.category, message=i.message))
    if tracer:
        tracer.log("review_findings", findings=[f.model_dump() for f in findings])
    return findings, status


def review_and_repair(
    llm: LLMClient,
    req: ExamRequest,
    bp: Blueprint,
    questions: dict[str, Question],
    verify: set[str],
    pedagogy: set[str],
    tracer: Tracer | None = None,
    max_rounds: int = MAX_REPAIR_ROUNDS,
    keep_stem: set[str] = frozenset(),
) -> tuple[dict[str, Question], ReviewReport]:
    """questions 会被返修后的题替换。verify：需要核验的题；pedagogy：需要教学审核的题（新写的题）。
    keep_stem：题干由老师指定（局部改字）的题，返修时只修正答案和解析，不整题重出。"""
    questions = dict(questions)
    report = ReviewReport()
    slots = {s.slot_id: s for s in bp.slots}
    prev_keys: set[tuple[str, str]] | None = None

    while True:
        findings, status = review_round(llm, req, bp, questions, verify, pedagogy, tracer)
        report.status.update(status)
        if report.rounds == 0:
            report.initial = findings
        must = [f for f in findings if f.severity == "must_fix"]
        report.suggestions = [f for f in report.suggestions if f.slot_id not in verify | pedagogy] + \
                             [f for f in findings if f.severity == "suggest"]
        bad = sorted({f.slot_id for f in must}, key=lambda s: slots[s].position)

        if not bad:
            report.stop_reason, report.remaining = "passed", []
            break
        if report.rounds >= max_rounds:
            report.stop_reason, report.remaining = "max_rounds", must
            break
        keys = {(f.slot_id, f.code) for f in must}
        if prev_keys is not None and keys <= prev_keys:   # 没有任何旧问题被解决
            report.stop_reason, report.remaining = "no_progress", must
            break

        notes = {}
        for sid in bad:
            msgs = "；".join(f.message for f in must if f.slot_id == sid)
            notes[sid] = f"原题“{questions[sid].stem[:60]}”（答案：{questions[sid].answer[:20]}）未通过审核：{msgs}。请重新出一道题，避免这些问题；计算类题目请先定解再构造方程，并填写 check 供程序验算。"
        others = [q for sid, q in questions.items() if sid not in bad]
        try:
            from .agents.writer import fix_answer_keep_stem

            new_q = {}
            for sid in [x for x in bad if x in keep_stem]:
                msgs = "；".join(f.message for f in must if f.slot_id == sid)
                new_q[sid] = fix_answer_keep_stem(llm, questions[sid], slots[sid], req, msgs, tracer)
            rest = [slots[s] for s in bad if s not in keep_stem]
            if rest:
                new_q.update(fill_slots(llm, req, rest, existing=others, notes=notes, tracer=tracer))
        except AgentFailure as e:
            report.stop_reason, report.remaining = "repair_failed", must
            if tracer:
                tracer.log("repair_failed", problems=e.problems)
            break
        questions.update(new_q)
        report.repaired |= set(new_q)
        report.rounds += 1
        prev_keys = keys
        verify = pedagogy = set(new_q)   # 下一轮只审核返修过的题

    # 教学审核仍有 must_fix 的题、或工具没覆盖到的题，最终状态以“未解决问题”为准
    for sid in {f.slot_id for f in report.remaining}:
        msgs = "；".join(f.message for f in report.remaining if f.slot_id == sid)
        old = report.status.get(sid, Verification(status=VStatus.UNVERIFIED))
        report.status[sid] = old.model_copy(update={"status": VStatus.PROBLEM, "detail": msgs})
    if tracer:
        tracer.log("review_done", rounds=report.rounds, stop_reason=report.stop_reason,
                   repaired=sorted(report.repaired), remaining=[f.model_dump() for f in report.remaining])
    return questions, report


NOT_REVIEWED = Verification(status=VStatus.NOT_REVIEWED, method="本次未开启审核")


def stamp_status(questions: dict[str, Question], report: ReviewReport | None,
                 unreviewed: set[str] = frozenset()) -> dict[str, Question]:
    """把审核得到的验证状态写到题目上（随版本快照存入数据库）。
    unreviewed：本次新写但没有审核的题，标为“未审核”；其余题保留原状态。"""
    out = dict(questions)
    for sid, q in questions.items():
        flagged = q.vstatus == VStatus.PROBLEM and (q.verification.method or "").startswith("命题自查")
        if flagged and report is not None and report.status.get(sid, NOT_REVIEWED).status != VStatus.PROBLEM:
            continue   # 命题时多轮返修仍未通过、保留交给老师的题：审核没改动它，标记不能被覆盖（d046）
        if report is not None and sid in report.status:
            out[sid] = q.model_copy(update={"verification": report.status[sid]})
        elif sid in unreviewed and q.vstatus != VStatus.PROBLEM:   # 命题时就标了“存在未解决问题”的不能被覆盖
            out[sid] = q.model_copy(update={"verification": NOT_REVIEWED})
    return out
