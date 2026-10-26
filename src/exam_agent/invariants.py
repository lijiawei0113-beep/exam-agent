"""不变量校验：LLM 只负责提议，这里的纯 Python 规则负责裁决。

每条违例都带有可直接反馈给 LLM 的中文说明，用于驱动返修。
"""
from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel

from .domain import VStatus, Blueprint, ExamRequest, PaperSnapshot, QType, allowed_types


class Violation(BaseModel):
    code: str
    message: str
    location: str = ""  # 如 slot_id，方便只返修出错的部分

    def __str__(self) -> str:
        loc = f"[{self.location}] " if self.location else ""
        return f"{loc}{self.message}"


CHOICE_LETTERS = ("A", "B", "C", "D")
# d036：命题员写“根据图片提示写出单词”，但系统不出图。提示词早有“不要出现如图”，仍被违反，所以由代码检查
_PICTURE = re.compile(r"(根据|看|如|观察|见)\s*(下|上|右|左)?\s*图|图片|图中|插图|picture|look at the photo", re.I)


def check_blueprint(req: ExamRequest, bp: Blueprint) -> list[Violation]:
    v: list[Violation] = []
    n = len(bp.slots)
    if n != req.question_count:
        v.append(Violation(code="count", message=f"题目数量为 {n}，要求 {req.question_count} 道"))

    total = sum(s.score for s in bp.slots)
    if total != req.total_score:
        v.append(Violation(code="total", message=f"各题分值之和为 {total}，要求满分 {req.total_score}"))

    allowed = allowed_types(req.subject)
    wrong = sorted({s.qtype.value for s in bp.slots if s.qtype not in allowed})
    if wrong:
        v.append(Violation(code="type_subject", message=f"{req.subject}不使用这些题型：{wrong}，"
                                                         f"可用题型为 {'、'.join(t.value for t in allowed)}"))

    if req.subject == "英语":
        from .agents.requirement import CN_GRADES
        from .tools.english import GRADE_LEVEL, LIMITS

        level = GRADE_LEVEL.get(CN_GRADES.index(req.grade[:1]) + 1 if req.grade[:1] in CN_GRADES else 0)
        n_read = sum(s.qtype == QType.READING for s in bp.slots)
        if level and n_read > LIMITS[level].reading_items:
            v.append(Violation(code="reading_count", message=f"{req.grade}英语卷阅读理解最多 {LIMITS[level].reading_items} 篇，"
                                                             f"现在 {n_read} 篇，请改成选择题或填空题"))
        if sum(s.qtype == QType.WRITING for s in bp.slots) > 1:
            v.append(Violation(code="writing_count", message="书面表达最多 1 题"))

    present = {s.qtype for s in bp.slots}
    for t in req.required_types:
        if t not in present:
            v.append(Violation(code="missing_type", message=f"缺少要求的题型：{t.value}"))

    positions = sorted(s.position for s in bp.slots)
    if positions != list(range(1, n + 1)):
        v.append(Violation(code="positions", message=f"题号应为 1..{n} 连续不重复，实际为 {positions}"))

    # 命题惯例：同一题型每题分值相同（如选择题每题 4 分）。
    # 这也让“加一道题、总分不变”之类的修改有明确的重新配分规则。
    scores_by_type: dict[QType, set[int]] = {}
    for s in bp.slots:
        scores_by_type.setdefault(s.qtype, set()).add(s.score)
    for t, scores in scores_by_type.items():
        if len(scores) > 1:
            v.append(Violation(code="uneven_type_score",
                               message=f"{t.value}各题分值不一致 {sorted(scores)}，同一题型每题分值应相同"))

    ids = Counter(s.slot_id for s in bp.slots)
    for sid, c in ids.items():
        if c > 1:
            v.append(Violation(code="dup_slot", message=f"slot_id 重复：{sid}", location=sid))
    return v


def check_paper(paper: PaperSnapshot) -> list[Violation]:
    """整卷一致性：先查蓝图，再查每道题是否与其 slot 吻合。"""
    v = check_blueprint(paper.request, paper.blueprint)
    slots = {s.slot_id: s for s in paper.blueprint.slots}
    seen: Counter[str] = Counter(q.slot_id for q in paper.questions)

    for sid in slots:
        if seen[sid] == 0:
            v.append(Violation(code="unfilled", message="该 slot 没有对应题目", location=sid))
        elif seen[sid] > 1:
            v.append(Violation(code="dup_question", message="该 slot 对应了多道题", location=sid))

    for q in paper.questions:
        s = slots.get(q.slot_id)
        if s is None:
            v.append(Violation(code="orphan", message=f"题目 {q.qid} 不属于任何 slot", location=q.slot_id))
            continue
        if q.qtype != s.qtype:
            v.append(Violation(code="type_mismatch", message=f"题型应为 {s.qtype.value}", location=s.slot_id))
        if q.score != s.score:
            v.append(Violation(code="score_mismatch", message=f"分值应为 {s.score}", location=s.slot_id))
        fmt = check_question_format(q.qtype, q.stem, q.options, q.answer, s.slot_id)
        if q.vstatus == VStatus.PROBLEM:
            # 已标为“存在未解决问题”、交给老师处理的题：引用图片不再阻止整卷保存（d046：兜底只在命题阶段生效，
            # 整卷校验仍把它判为不合格，整卷照样失败）
            fmt = [x for x in fmt if x.code != "no_picture"]
        v.extend(fmt)
    return v


def check_question_format(qtype: QType, stem: str, options: list[str], answer: str, location: str) -> list[Violation]:
    v: list[Violation] = []
    if not stem.strip():
        v.append(Violation(code="empty_stem", message="题干为空", location=location))
    if not answer.strip():
        v.append(Violation(code="empty_answer", message="答案为空", location=location))
    if qtype == QType.CHOICE:
        if len(options) != 4:
            v.append(Violation(code="options", message=f"选择题需要 4 个选项，实际 {len(options)} 个", location=location))
        if answer.strip().upper() not in CHOICE_LETTERS:
            v.append(Violation(code="choice_answer", message="选择题答案必须是 A/B/C/D 之一", location=location))
    elif options:
        v.append(Violation(code="options", message=f"{qtype.value}不应包含选项", location=location))
    if _PICTURE.search(stem):
        v.append(Violation(code="no_picture", message="题干引用了图片（如“根据图片”“看图”“如图”），但试卷不支持插图；"
                                                    "请改为文字描述或中文提示", location=location))
    if qtype == QType.FILL and "____" not in stem and "（ ）" not in stem and "( )" not in stem:
        v.append(Violation(code="fill_blank", message="填空题题干中需要有空位（用 ____ 表示）", location=location))
    return v


def check_preserved(old: PaperSnapshot, new: PaperSnapshot, allowed_slots: set[str]) -> list[Violation]:
    """局部修改的核心保证：不在 allowed_slots 中的题，内容指纹必须完全一致。
    （分值可能因整卷重新配分而变化，由 diff 单独向老师报告。）"""
    v: list[Violation] = []
    new_by_slot = {q.slot_id: q for q in new.questions}
    for q in old.questions:
        if q.slot_id in allowed_slots:
            continue
        nq = new_by_slot.get(q.slot_id)
        if nq is None or nq.content_fingerprint() != q.content_fingerprint():
            v.append(Violation(code="unexpected_change", message="未被要求修改的题目发生了变化", location=q.slot_id))
    return v
