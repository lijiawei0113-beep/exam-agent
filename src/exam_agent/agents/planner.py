"""规划 Agent：根据 ExamRequest 产出双向细目表式的出卷蓝图。

分工：
- 模型负责语义决策：每道题的题型、知识点、难度、认知层次、考查意图，以及建议分值。
- 代码负责算术与编号：按题型重新排序编号，由求解器把分值配成“同类同分、总分正确”。
- 只有代码无法自行修正的问题（题数不对、缺题型、题型配比凑不出满分）才反馈给模型重做。

实测教训：让模型同时满足“题数 + 总分 + 同类同分”，连续 3 次都没凑对（见 evals/real_defects.jsonl d009）。
"""
from __future__ import annotations

import re

from ..domain import DEFAULT_TYPES, Blueprint, plan_types, ExamRequest, QType, allowed_types
from ..grounding import Candidates
from ..invariants import check_blueprint
from ..llm import LLMClient
from ..scoring import current_scores, feasible_count_options, rebalance
from .base import Tracer, structured_call

SYSTEM = """你是中小学命题助手中的“规划员”，负责制定出卷蓝图（双向细目表）。
要求：
1. slots 数量必须恰好等于题数。请先数清楚再输出。
2. 老师要求的每种题型至少出现一次；题型配比要像一份真实的单元测试卷，
   客观题（选择、填空）和主观题（计算、应用）都要有一定数量。
3. score 填你建议的每题分值即可（选择题、填空题常见 3～5 分，计算题、应用题常见 8～15 分），
   最终分值会由程序统一调整为“同类题同分、总分等于满分”，你不需要自己凑总分。
4. 难度默认按 基础:中等:提高 约 6:3:1 分配，除非老师另有要求；难题放在同类题靠后位置。
5. difficulty 只能是 基础/中等/提高；cognitive_level 只能是 了解/理解/掌握/运用；
   qtype 只能从给出的“本学科可用题型”中选择。英语卷通常：选择题考词汇语法，填空题考单词拼写和词形变化，
   阅读理解每题是一篇短文加 3～5 个小题，书面表达最多 1 题。
6. knowledge_point 写具体的子知识点（如“移项”“去括号”“去分母”），覆盖单元主要内容，避免重复。
   如果给出了“考查范围”，每个 slot 还要填 kp_id（从候选列表中选），不得出范围以外的内容。
7. position 按出现顺序从 1 编号；slot_id 用 S1、S2…（程序会重新整理编号）。
8. 只输出 json，格式示例：
{"slots": [{"slot_id": "S1", "position": 1, "qtype": "选择题", "score": 4,
  "knowledge_point": "方程的概念", "difficulty": "基础", "cognitive_level": "了解",
  "intent": "辨认一元一次方程", "kp_id": "M7A-05-02"}],
 "rationale": "一句话说明题型与难度分配思路"}
rationale 只写思路，不要写题数、分数、比例等数字（这些由程序统计，实测自述的数字常与蓝图不一致）。"""


def check_grounding(bp: Blueprint, cands: Candidates | None) -> list[str]:
    """每道题必须挂在候选知识点上——这是“只考范围内内容”的硬约束，由代码裁决。"""
    if cands is None:
        return []
    bad = [f"第{s.position}题（{s.knowledge_point}）的 kp_id“{s.kp_id or '空'}”" for s in bp.slots if s.kp_id not in cands.ids]
    if bad:
        return ["以下题目没有从候选知识点中选择 kp_id，或选了范围外的 id：" + "；".join(bad)
                + "。kp_id 只能取候选列表中方括号里的 id"]
    return []


def max_counts(req: ExamRequest) -> dict[QType, int]:
    """按学科、年级限制某些题型的数量（英语阅读篇数、书面表达题数）。"""
    if req.subject != "英语":
        return {}
    from ..tools.english import GRADE_LEVEL, LIMITS
    from .requirement import CN_GRADES

    level = GRADE_LEVEL.get(CN_GRADES.index(req.grade[:1]) + 1 if req.grade[:1] in CN_GRADES else 0)
    return {QType.READING: LIMITS[level].reading_items, QType.WRITING: 1} if level else {QType.WRITING: 1}


def check_disguised_reading(req: ExamRequest, bp: Blueprint) -> list[str]:
    """d026：阅读篇数被限制后，模型把 6 篇短文放进了选择题。阅读类内容必须用阅读理解题型。"""
    if req.subject != "英语":
        return []
    bad = [f"第{s.position}题（{s.qtype.value}：{s.knowledge_point}）" for s in bp.slots
           if s.qtype not in (QType.READING, QType.WRITING)   # 书面表达本来就是写短文（e2e 第二轮误判 2 次）
           and re.search(r"阅读|短文|语篇", s.knowledge_point + s.intent)]
    return [f"这些题实际是阅读短文，却标成了其他题型：{'；'.join(bad)}。阅读请用阅读理解题型，并遵守篇数上限；"
            "其他题型考单词、句型、交际用语"] if bad else []


def check_concentration(bp: Blueprint, n_points: int = 0) -> list[str]:
    """d030：20 题里有 5 道都是“字母 X-X 书写”，只换了字母。去掉字母、数字、符号后知识点相同的题不能太多。"""
    if len(bp.slots) < 10:   # 小卷子（如专项练习 4 题）允许集中考一个知识点
        return []
    groups: dict[str, list[int]] = {}
    for s in bp.slots:
        key = re.sub(r"[A-Za-z0-9\s\-—～~、，,.（）()'’]+", "", s.knowledge_point)
        if key:
            groups.setdefault(key, []).append(s.position)
    limit = max(3, len(bp.slots) // 5)
    if n_points:   # 范围内知识点很少（如“可能性”单元只有一两个），重复是必然的（d046）
        limit = max(limit, -(-len(bp.slots) // n_points) + 1)
    over = [f"“{k}”（第{'、'.join(map(str, v))}题）" for k, v in groups.items() if len(v) > limit]
    return [f"这些知识点重复出题太多：{'；'.join(over)}，同一知识点最多 {limit} 题，请换成其他单元或其他考查角度"] if over else []


def finalize_blueprint(req: ExamRequest, bp: Blueprint, cands: Candidates | None = None) -> list[str]:
    """就地修正编号与分值；返回只有模型才能解决的问题。"""
    grounding = check_grounding(bp, cands) + check_disguised_reading(req, bp) + check_concentration(bp, len(cands.points) if cands else 0)
    if grounding:
        return grounding
    if len(bp.slots) != req.question_count:
        return [f"题目数量为 {len(bp.slots)}，要求恰好 {req.question_count} 道，请增删题目后重新输出"]

    bp.resequence()
    for i, s in enumerate(bp.slots, start=1):
        s.slot_id = f"S{i}"

    counts: dict[QType, int] = {}
    for s in bp.slots:
        counts[s.qtype] = counts.get(s.qtype, 0) + 1
    assign = rebalance(counts, req.total_score, current_scores(bp))
    if assign is None:
        now = "、".join(f"{t.value} {c} 题" for t, c in counts.items())
        opts = feasible_count_options(req.question_count, req.total_score, list(counts), max_counts=max_counts(req))
        tip = ("；可行的配比有：" + "；".join(_fmt(o) for o in opts[:3])) if opts else ""
        return [f"当前题型配比（{now}）在合理分值范围内无法凑成满分 {req.total_score}{tip}"]
    for s in bp.slots:
        s.score = assign[s.qtype]
    return [str(v) for v in check_blueprint(req, bp)]


def _fmt(counts: dict[QType, int]) -> str:
    return "、".join(f"{t.value} {c} 题" for t, c in counts.items())


def plan_blueprint(llm: LLMClient, req: ExamRequest, tracer: Tracer | None = None,
                   cands: Candidates | None = None) -> Blueprint:
    types = plan_types(req)
    opts = feasible_count_options(req.question_count, req.total_score, types, max_counts=max_counts(req))
    user = "出卷要求（json）：\n" + req.model_dump_json(indent=2)
    user += f"\n本学科可用题型：{'、'.join(t.value for t in allowed_types(req.subject))}"
    if req.subject == "英语":
        from .writer import _english_rules
        user += "\n" + _english_rules(req)
    if cands is not None:
        user += ("\n\n考查范围（来自课程知识库，" + cands.how + "）。每道题的 kp_id 必须从下列方括号里的 id 中选择，"
                 "knowledge_point 写该卡片范围内更具体的内容；不要出范围以外的内容；"
                 "题目尽量覆盖不同单元，按各单元内容多少分配题量：\n"
                 + "\n".join(f"- {p.brief()}" for p in cands.points))
    if opts:
        user += ("\n\n程序已算好、能够凑成满分的题型题数配比（请从中选择一个最像真实单元卷的）：\n"
                 + "\n".join(f"- {_fmt(o)}" for o in opts))
    bp = structured_call(
        llm, step="planner", system=SYSTEM, user=user, schema=Blueprint,
        validate=lambda bp: finalize_blueprint(req, bp, cands), tracer=tracer,
    ).value
    if cands is not None:
        by_id = {p.id: p for p in cands.points}
        for s in bp.slots:
            p = by_id[s.kp_id]
            s.scope_note = (f"{p.location}·{p.name}" + (f"；主要内容：{'、'.join(p.key_points)}" if p.key_points else "")
                            + (f"；句型：{' / '.join(p.patterns)}" if p.patterns else "")
                            + (f"；本单元词汇：{', '.join(p.vocabulary)}" if p.vocabulary else "")
                            + (f"；不考（后续才学的内容）：{p.boundary}" if p.boundary else ""))
        bp.scope = cands.how
        bp.signals = dict(cands.signals)
        bp.vocabulary = list(cands.vocabulary)
        if tracer:
            tracer.log("grounding", how=cands.how, candidates=len(cands.points),
                       covered=sorted({s.kp_id for s in bp.slots}))
    if tracer:
        tracer.log("planner_scores", scores={s.qtype.value: s.score for s in bp.slots})
    return bp
