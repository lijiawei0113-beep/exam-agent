"""教学审核 Agent：判断工具判断不了的东西——是否超纲、表述是否适合该年级、
是否与蓝图要求相符、是否与其他题重复。

为了避免“为挑错而挑错”导致无限返修，审核意见分两级：
- must_fix：明确的错误，会触发返修；
- suggest：风格或偏好，只告知老师，不触发返修。
"""
from __future__ import annotations

import re

from typing import Literal

from pydantic import BaseModel

from ..domain import BlueprintSlot, ExamRequest, Question
from ..llm import LLMClient
from .base import Tracer, structured_call


class ReviewIssue(BaseModel):
    slot_id: str
    severity: Literal["must_fix", "suggest"]
    category: str
    message: str


class ReviewOutput(BaseModel):
    issues: list[ReviewIssue]


SYSTEM = """你是一名有经验的中小学教研员，负责审核试卷题目。数学计算的正确性已由其他程序核验，你不需要验算。
你只审核“需要审核的题”，其他题仅作为参照，用来判断是否重复。
检查以下方面：
- 超纲：超出出卷范围的课程要求；题目有“范围说明”时以它为准，并在 message 中写明依据。
  注意：范围说明是这道题的考查重点，不是限制——出卷范围内的其他单元、前面单元和前面年级学过的内容都可以使用，
  不算超纲；范围说明里的“不考”只指后续才学的内容。
  英语：个别生词、不在单元词表里的词都不算超纲，不要因为单词提出 must_fix；语法级别和阅读难度已由程序按课标检查；
- 答案泄露：题干里的中文提示或翻译直接给出了答案（不懂英语也能选对），或者要填的结果已经写在题干里
  （如“盘子里有 3 个苹果，苹果有 ____ 个”），属于 must_fix；
  考查的是明显更低年级的内容、与范围说明不符，也属于与蓝图不符；
- 表述：题干是否清楚、符合该年级学生的阅读水平、是否存在不合常理的情境数据（如人数为小数）；
- 蓝图符合度：题型、知识点、难度是否与该题的蓝图要求一致；
- 重复：是否与其他题考查同一内容、同一情境。

severity 规则：
- must_fix 只用于明确的错误：超纲、情境不合常理、与蓝图明显不符、与其他题实质重复、表述会导致学生误解。
- suggest 用于可改可不改的建议。没有问题的题不要输出。
只输出 json，示例：
{"issues": [{"slot_id": "S10", "severity": "must_fix", "category": "重复",
  "message": "与第 9 题都是和差倍分问题且情境相似，应换成其他类型的应用题"}]}
没有问题时输出 {"issues": []}"""


def review_pedagogy(
    llm: LLMClient,
    req: ExamRequest,
    slots: dict[str, BlueprintSlot],
    questions: dict[str, Question],
    targets: set[str],
    tracer: Tracer | None = None,
) -> list[ReviewIssue]:
    lines = [f"出卷范围：{req.scope_text}，学科：{req.subject}"]
    lines.append("需要审核的题：")
    for sid in sorted(targets, key=lambda s: slots[s].position):
        s, q = slots[sid], questions[sid]
        lines.append(f"[{sid}] 第{s.position}题 蓝图要求：{s.qtype.value}，知识点“{s.knowledge_point}”，"
                     f"难度{s.difficulty.value}，意图“{s.intent}”")
        if s.scope_note:
            # 英语不给审核看单元词表：按单词挑超纲违背“个别生词可以出现”（d029）
            note = re.sub(r"；本单元词汇：[^；]*", "", s.scope_note) if req.subject == "英语" else s.scope_note
            lines.append(f"    范围说明（来自课程知识库）：{note}")
        lines.append(f"    题干：{q.stem}")
        if q.options:
            lines.append("    选项：" + "；".join(f"{l}. {o}" for l, o in zip("ABCD", q.options)))
    others = [sid for sid in questions if sid not in targets]
    if others:
        lines.append("其他题（仅供判断重复）：")
        for sid in sorted(others, key=lambda s: slots[s].position):
            lines.append(f"[{sid}] 第{slots[sid].position}题 {slots[sid].knowledge_point}：{questions[sid].stem[:50]}")

    out = structured_call(llm, step="pedagogy_reviewer", system=SYSTEM, user="\n".join(lines),
                          schema=ReviewOutput, tracer=tracer).value
    return [i for i in out.issues if i.slot_id in targets]
