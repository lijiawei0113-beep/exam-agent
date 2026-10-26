"""命题 Agent：为指定的若干 slot 生成题目。

只为“被要求的 slot”出题，这个接口在后续局部修改阶段会被直接复用。
题型、分值、知识点、难度一律从 slot 复制，LLM 只负责题干、选项、答案和解析。
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..domain import BlueprintSlot, ExamRequest, QType, Question, integer_required
from ..invariants import check_question_format
from ..llm import LLMClient
from ..tools.arith import evaluate, matches_answer, remainder_division
from ..tools.linear import ToolError, extract_numbers, fmt, solve
from .base import AgentFailure, Tracer, structured_call


class ToolCheck(BaseModel):
    equations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    target: str = ""
    expression: str = ""   # 一个算式就能算出答案时填写（小学题常见）


class QuestionDraft(BaseModel):
    slot_id: str
    stem: str
    options: list[str] = Field(default_factory=list)
    answer: str
    analysis: str
    check: ToolCheck | None = None


class WriterOutput(BaseModel):
    questions: list[QuestionDraft]


SYSTEM = """你是中小学命题助手中的“命题员”。按给定的蓝图 slot 逐一出题。
要求：
1. 每个 slot 恰好出一道题，slot_id 原样返回，不要多出也不要遗漏。
2. 严格匹配 slot 的题型、知识点、难度和考查意图，语言符合该年级学生的认知水平。
3. 选择题：options 恰好 4 个，内容不带“A.”等字母前缀；answer 只写一个字母 A/B/C/D；干扰项要有迷惑性。
4. 填空题：题干中用 ____ 表示空位；options 为空列表。
5. 计算题、应用题：options 为空列表；answer 写最终结果，analysis 写完整解题步骤。
   answer 中的数值写精确值（如 x = 3、12/7 小时），不要写近似小数。
   【反向构造】出解方程题和应用题时，先确定一个简洁的解（七年级一般为整数），再围绕它构造方程或情境，
   最后把解代回原方程检验。不要先随手写方程再去算解。
【英语】题目内容用英文，作答说明可以用中文（如“用所给词的适当形式填空”）；用词不超出该年级、该范围学过的词汇和句型。
   选择题不要在题干后附上能直接看出答案的中文翻译（如“The apple is ____.（这个苹果是红色的。）”）；
   需要中文时只描述情境（如“见到同学时打招呼”），让学生自己判断英文。
   阅读理解：stem 写一篇短文（小学 50～120 词，七年级 120～200 词），后面接 3～5 个带编号的小题，
   每个小题的 A～D 选项直接写在 stem 里；options 为空列表；answer 写成“1. B 2. C 3. A”；analysis 逐题说明依据。
   书面表达：stem 写写作要求（话题、要点、词数）；answer 写一篇范文；analysis 写评分要点。
6. 数学式用纯文本书写，如 3x + 5 = 2x - 1，分数写成 (x+1)/2。
7. 题干必须自包含：不要出现“下列各数”“如图”等引用，除非题干里确实给出了这些内容。
8. 必须遵守“全卷约束”；不要与“已有题目”重复。
   slot 里有 scope_note 时，它是这道题的考查范围（主要内容、不考什么），题目不得超出这个范围，
   也不要用更低年级的内容凑数。
9. 【供程序验算】如果题目最终所求就是某个方程（组）的解（解方程题、求参数题、应用题、求解的填空题），
   填写 check：equations 写方程（只用数字、字母、+ - * / ( ) 和 =），unknowns 写未知数，target 写所求未知数。
   程序会用求解器精确验算 answer。不需要设未知数、一个算式就能算出答案时（小学计算题、解决问题），
   check 只填 expression，如 {"expression": "(120-30)÷3"}，只用数字、+ - × ÷ ( ) 和 %。
   所求是中间步骤或表达式、或者是概念类选择题时，check 为 null。
10. 只输出 json，字符串中的换行用 \\n 表示。格式示例：
{"questions": [{"slot_id": "S1", "stem": "下列方程中，是一元一次方程的是（ ）",
  "options": ["x + y = 1", "2x - 3 = 5", "x² = 4", "1/x = 2"], "answer": "B",
  "analysis": "一元一次方程只含一个未知数且次数为 1，故选 B。", "check": null},
 {"slot_id": "S6", "stem": "解方程：3(x - 2) = x + 4", "options": [], "answer": "x = 5",
  "analysis": "去括号得 3x - 6 = x + 4，移项得 2x = 10，所以 x = 5。",
  "check": {"equations": ["3*(x - 2) = x + 4"], "unknowns": ["x"], "target": "x"}}]}"""


def _english_rules(req: ExamRequest) -> str:
    from ..tools.english import grade_rules_text
    from .requirement import CN_GRADES

    return grade_rules_text(CN_GRADES.index(req.grade[:1]) + 1) if req.grade[:1] in CN_GRADES else ""


def _build_user(req: ExamRequest, slots: list[BlueprintSlot], existing: list[Question],
                feedback: list[str] | None, notes: dict[str, str] | None = None) -> str:
    lines = [
        f"出卷范围：{req.scope_text}，学科：{req.subject}",
        f"整体难度要求：{req.difficulty_note or '无'}",
        "全卷约束：" + ("；".join(
            f"{c.text}（作用于：{'、'.join(t.value for t in c.scope_types) or '全卷'}）" for c in req.constraints
        ) or "无"),
        "其他要求：" + ("；".join(req.extra_requirements) or "无"),
        *([_english_rules(req)] if req.subject == "英语" else []),
        "需要出题的 slot（json）：",
        "[" + ",\n".join(s.model_dump_json() for s in slots) + "]",
    ]
    if notes:
        lines.append("针对个别 slot 的具体要求：")
        lines += [f"- {sid}：{n}" for sid, n in notes.items() if sid in {s.slot_id for s in slots}]
    if existing:
        lines.append("已有题目（避免重复）：")
        lines += [f"- {q.slot_id} {q.qtype.value}：{q.stem[:60]}" for q in existing]
    if feedback:
        lines.append("上一轮这些题没有通过检查，请针对性修正：")
        lines += [f"- {f}" for f in feedback]
    return "\n".join(lines)


def self_check(d: QuestionDraft, s: BlueprintSlot, req: ExamRequest) -> str:
    """命题时当场用求解器验算。工具解不了就不下结论（交给之后的独立审核）。"""
    if d.check is None or s.qtype == QType.CHOICE:
        return ""
    if d.check.equations:
        try:
            sol = solve(d.check.equations, d.check.unknowns)
        except ToolError:
            return ""
        target = d.check.target or (d.check.unknowns[0] if d.check.unknowns else "")
        value, what = sol.get(target), f"你给出的方程解得 {target} = "
    elif d.check.expression:
        expr = d.check.expression
        try:
            value = evaluate(expr)
        except ToolError:
            return ""
        if not matches_answer(expr, value, d.answer):
            return (f"[{s.slot_id}] 求解器验算：你给出的算式 {expr} = {fmt(value)}，但答案写的是“{d.answer}”。"
                    "请重新出这道题：先定一个简洁的答案，再构造题目，并保证答案与解析一致")
        if integer_required(req, s.qtype) and value.denominator != 1 and remainder_division(expr) is None:
            return f"[{s.slot_id}] 求解器验算：结果为 {fmt(value)}，不是整数，违反“答案为整数”的约束，请重新构造"
        return ""
    else:
        return ""
    if value is None:
        return ""
    if value not in extract_numbers(d.answer):
        return (f"[{s.slot_id}] 求解器验算：{what}{fmt(value)}，但答案写的是“{d.answer}”。"
                "请重新出这道题：先定一个简洁的答案，再构造题目，并保证答案与解析一致")
    if integer_required(req, s.qtype) and value.denominator != 1:
        return f"[{s.slot_id}] 求解器验算：解为 {fmt(value)}，不是整数，违反“答案为整数”的约束，请重新构造"
    return ""


def english_self_check(d: QuestionDraft, s: BlueprintSlot, req: ExamRequest) -> str:
    """英语题命题时就用代码检查语法级别、短文词数、生词比例，带着具体数字当场返修，不等到审核阶段。
    （d034：模型数不准词数，三年级短文要求 50～200 词，返修两次仍只写了 33、40 词。）"""
    if req.subject != "英语":
        return ""
    from ..tools.english import check_question as check_en, english_words, passage_of, scope_end_book
    from .requirement import CN_GRADES

    g = CN_GRADES.index(req.grade[:1]) + 1 if req.grade[:1] in CN_GRADES else 0
    bad = [i for i in check_en(s.qtype.value, d.stem, d.options, d.answer, g, scope_end_book(req.grade, req.term))
           if i.severity == "must_fix"]
    if not bad:
        return ""
    msg = "；".join(i.message for i in bad)
    if any(i.code == "reading_length" for i in bad):
        n = len(english_words(passage_of(d.stem)))
        msg += f"（你写的短文是 {n} 词，请按句子数控制：每句 5～8 词，" \
               f"三四年级写 10～20 句，五六年级 15～35 句，七年级 20～45 句）"
    return f"[{s.slot_id}] 程序检查：{msg}。请重写这道题"


class Expansion(BaseModel):
    passage: str = Field(description="扩写后的完整短文，段与段之间用换行分隔")


_ITEM_LINE = re.compile(r"^[(（]?\d+\s*[.、)）]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", x).strip().lower() for x in _SENT.split(text.strip()) if x.strip()]


def keeps_original(original: str, new: str) -> list[str]:
    """原文每一句都必须按原顺序、一字不改地出现在扩写稿里；返回缺失或被改动的句子。"""
    new_s, i, missing = _sentences(new), 0, []
    for sent in _sentences(original):
        while i < len(new_s) and new_s[i] != sent:
            i += 1
        if i == len(new_s):
            missing.append(sent)
            i = 0
        else:
            i += 1
    return missing


def replace_passage(stem: str, passage: str) -> str:
    """用扩写稿替换题干里的短文行；作答说明、编号小题和选项原样保留。"""
    from ..tools.english import passage_of

    lines = stem.splitlines()
    first_item = next((i for i, l in enumerate(lines) if _ITEM_LINE.match(l.strip())), len(lines))
    idx = [i for i in range(first_item) if passage_of(lines[i])]
    if not idx:
        return stem
    new = [l.strip() for l in passage.strip().splitlines() if l.strip()]
    return "\n".join(lines[:idx[0]] + new + [l for i, l in enumerate(lines[idx[0]:], idx[0]) if i not in idx])


SYSTEM_EXPAND = """你是小学/初中英语命题老师。下面这篇阅读短文太短，请扩写到规定长度。
要求：
- 原文的每一句都必须保留，一字不改，顺序不变；
- 新句子可以插在原句之间（补充细节、动作、感受），也可以在结尾前补充情节，要让整篇读起来连贯自然，不要在结尾硬加句子，不要重复已有的意思；
- 不能加入与已有小题答案矛盾的信息，也不能让任何小题出现第二个正确答案；
- 用词简单，语法不超出给定年级的范围；按要求的段数分段，段与段之间换行。
输出 JSON：{"passage": "扩写后的完整短文"}"""


def expand_reading(llm: LLMClient, d: QuestionDraft, s: BlueprintSlot, req: ExamRequest,
                   tracer: Tracer | None = None) -> QuestionDraft | None:
    """短文词数不足时的兜底：在保留原句的前提下扩写（代码核对原句都在、重新数词），而不是整题重写。
    整题重写会改动小题和答案，且模型仍然写不长（d038）；只在结尾续写又容易前后不连贯。"""
    from ..tools.english import GRADE_LEVEL, LIMITS, english_words, grade_rules_text, passage_of
    from .requirement import CN_GRADES

    g = CN_GRADES.index(req.grade[:1]) + 1 if req.grade[:1] in CN_GRADES else 0
    lim = LIMITS[GRADE_LEVEL.get(g, 1)]
    original = passage_of(d.stem)
    n = len(english_words(original))

    def validate(e: Expansion) -> list[str]:
        problems = []
        missing = keeps_original(original, e.passage)
        if missing:
            problems.append("这些原句被删掉或改动了，必须一字不改地保留：" + " | ".join(missing[:3]))
        m = len(english_words(e.passage))
        if not lim.passage_min <= m <= lim.passage_max:
            problems.append(f"扩写稿 {m} 词，需要在 {lim.passage_min}～{lim.passage_max} 词之间（目标约 {lim.target} 词）")
        return problems

    user = (f"{grade_rules_text(g)}\n"
            f"短文现在 {n} 词，需要扩写到约 {lim.target} 词：{lim.shape}。\n\n"
            f"原短文：\n{original}\n\n小题（不要改动，也不要写进短文）：\n"
            + "\n".join(l for l in d.stem.splitlines() if _ITEM_LINE.match(l.strip()) or re.match(r"^[A-D]\s*[.、]", l.strip()))
            + f"\n\n答案：{d.answer}")
    ext = structured_call(llm, step="writer_expand", system=SYSTEM_EXPAND, user=user, schema=Expansion,
                          validate=validate, tracer=tracer).value
    out = d.model_copy(update={"stem": replace_passage(d.stem, ext.passage)})
    if tracer:
        tracer.log("writer_expand_done", slot_id=s.slot_id, before=n, after=len(english_words(passage_of(out.stem))))
    return out


def _too_short(d: QuestionDraft, s: BlueprintSlot, req: ExamRequest) -> bool:
    if req.subject != "英语" or s.qtype != QType.READING:
        return False
    from ..tools.english import GRADE_LEVEL, LIMITS, english_words, passage_of
    from .requirement import CN_GRADES

    g = CN_GRADES.index(req.grade[:1]) + 1 if req.grade[:1] in CN_GRADES else 0
    return len(english_words(passage_of(d.stem))) < LIMITS[GRADE_LEVEL.get(g, 1)].passage_min


class WriteResult(BaseModel):
    questions: list[Question]
    problems: list[str]  # 未通过格式检查或被遗漏的 slot 的问题说明
    rejected: dict[str, Question] = Field(default_factory=dict)   # 格式合格、但工具 / 年级规则检查未通过的稿子
    corrected: list[str] = Field(default_factory=list)            # 由代码按精确计算更正了答案的 slot


class AnalysisFix(BaseModel):
    analysis: str


class AnswerFix(BaseModel):
    answer: str
    analysis: str
    check: ToolCheck | None = None


def fix_answer_keep_stem(llm, q: Question, s: BlueprintSlot, req: ExamRequest, problem: str,
                         tracer: Tracer | None = None) -> Question:
    """老师改过字的题未通过核对时，题干是老师定的，不能整题重出（e2e 第二轮：返修把整题换掉，老师改的数字没了）。
    只修正答案和解析：题干本身是算式就由代码算；否则让模型按原题干重新作答，并用求解器验算它给的算式。"""
    d = QuestionDraft(slot_id=q.slot_id, stem=q.stem, options=q.options, answer=q.answer, analysis=q.analysis)
    d2, fixed = correct_computed_answer(llm, d, s, tracer)
    if fixed:
        return q.model_copy(update={"answer": d2.answer, "analysis": d2.analysis, "verification": None})
    out = structured_call(
        llm, step="writer_fix_answer",
        system="你是中小学命题老师。下面这道题的题干由老师指定，一个字都不能改。现在的答案没有通过核对，"
               "请按题干重新解答，给出正确答案和解析；计算类题目在 check 里写出算式或方程供程序验算。只输出 json："
               "{\"answer\": \"...\", \"analysis\": \"...\", \"check\": {\"expression\": \"...\"}}",
        user=f"题干：{q.stem}\n选项：{q.options}\n原答案：{q.answer}\n核对发现的问题：{problem}",
        schema=AnswerFix, tracer=tracer).value
    draft = d.model_copy(update={"answer": out.answer, "analysis": out.analysis, "check": out.check})
    if self_check(draft, s, req):
        raise AgentFailure("writer_fix_answer", [self_check(draft, s, req)])
    return q.model_copy(update={"answer": out.answer.strip().upper() if s.qtype == QType.CHOICE else out.answer.strip(),
                                "analysis": out.analysis.strip(), "verification": None})


def correct_computed_answer(llm, d: QuestionDraft, s: BlueprintSlot, tracer: Tracer | None = None):
    """题干本身就是一个算式（“计算：-2² + (-3)² × (-1/3) - (-8) ÷ 4”）时，答案完全由题目决定：
    题目没有问题，错的只是模型算的答案。直接用精确计算结果更正答案，只让模型重写解析，不再整题重出。
    （d039：七年级计算题模型三轮都把答案算错，整份试卷因此失败。）返回 (更正后的稿子, 是否更正)。"""
    from ..verify import from_stem

    if s.qtype not in (QType.CALC, QType.FILL):
        return d, False
    f = from_stem(d.stem)
    if f is None or f.kind != "expression":
        return d, False
    try:
        value = evaluate(f.text[0])
    except ToolError:
        return d, False
    if matches_answer(f.text[0], value, d.answer):
        return d, False
    answer = fmt(value)
    try:
        fixed = structured_call(
            llm, step="writer_fix_analysis",
            system="你是数学老师。下面这道计算题的正确答案已由程序精确算出，请按正确的运算顺序写出解析，"
                   "最后结果必须等于给定答案。只输出 json：{\"analysis\": \"...\"}",
            user=f"题目：{d.stem}\n正确答案：{answer}\n原解析（有错误，仅供参考）：{d.analysis}",
            schema=AnalysisFix, tracer=tracer).value.analysis
    except AgentFailure:
        fixed = f"按运算顺序计算，结果为 {answer}。"
    if tracer:
        tracer.log("writer_answer_corrected", slot_id=s.slot_id, stated=d.answer, correct=answer)
    return d.model_copy(update={"answer": answer, "analysis": fixed, "check": None}), True


def write_questions(
    llm: LLMClient,
    req: ExamRequest,
    slots: list[BlueprintSlot],
    existing: list[Question] | None = None,
    feedback: list[str] | None = None,
    tracer: Tracer | None = None,
    notes: dict[str, str] | None = None,
) -> WriteResult:
    """结构/JSON 错误在 structured_call 内重试；单题质量问题不整批重来，
    合格的题直接收下，不合格的 slot 交由上层带反馈单独返修。"""
    existing = existing or []
    by_id = {s.slot_id: s for s in slots}

    # 英语阅读理解每篇单独调用：和整份卷挤在一个回答里时，模型会压缩每道题的篇幅（d038）
    reading = [s for s in slots if req.subject == "英语" and s.qtype == QType.READING]
    groups = ([[s for s in slots if s not in reading]] if len(reading) < len(slots) else []) + [[s] for s in reading]
    drafts: list[QuestionDraft] = []
    for group in groups:
        ids = {s.slot_id for s in group}
        fb = [f for f in (feedback or []) if not re.match(r"^\[S\w+\]", f) or f.split("]")[0][1:] in ids]
        user = _build_user(req, group, existing, fb or None, notes)
        if group and group[0] in reading:
            user += "\n这次只写这一篇阅读理解，请把篇幅写足。"
        out = structured_call(llm, step="writer", system=SYSTEM, user=user, schema=WriterOutput, tracer=tracer).value
        drafts += [d for d in out.questions if d.slot_id in ids]

    accepted: dict[str, Question] = {}
    rejected: dict[str, Question] = {}
    corrected: list[str] = []
    problems: list[str] = []
    for d in drafts:
        s = by_id.get(d.slot_id)
        if s is None or d.slot_id in accepted:
            continue  # 未要求的 slot 或重复，直接丢弃
        issues = check_question_format(s.qtype, d.stem, d.options, d.answer, s.slot_id)
        if issues:
            problems += [str(v) for v in issues]
            if all(v.code == "no_picture" for v in issues):
                # 只差“引用了图片”：多轮仍改不掉时保留并标为问题题交给老师，不让整卷失败（e2e 第二轮 r070）
                rejected[s.slot_id] = _to_question(d, s, VStatus_problem("；".join(str(v) for v in issues)))
            continue
        if _too_short(d, s, req):
            try:
                d = expand_reading(llm, d, s, req, tracer) or d
            except AgentFailure:
                pass      # 扩写也不成功就退回到原来的整题返修
        d, fixed = correct_computed_answer(llm, d, s, tracer)
        if fixed:
            corrected.append(s.slot_id)
        tool_problem = self_check(d, s, req) or english_self_check(d, s, req)
        if tool_problem:
            problems.append(tool_problem)
            rejected[s.slot_id] = _to_question(d, s, VStatus_problem(tool_problem))
            continue
        accepted[s.slot_id] = _to_question(d, s)
    for sid in by_id:
        if sid not in accepted and not any(p.startswith(f"[{sid}]") for p in problems):
            problems.append(f"[{sid}] 这道题被遗漏了")
    if tracer:
        tracer.log("writer_check", accepted=sorted(accepted), problems=problems)
    return WriteResult(questions=list(accepted.values()), problems=problems,
                       rejected={k: v for k, v in rejected.items() if k not in accepted}, corrected=corrected)


MAX_WRITE_ROUNDS = 3


def VStatus_problem(message: str):
    from ..domain import Verification, VStatus

    return Verification(status=VStatus.PROBLEM, method="命题自查（多轮返修后仍未通过）", detail=message[:200])


def _to_question(d: QuestionDraft, s: BlueprintSlot, verification=None) -> Question:
    return Question(
        slot_id=s.slot_id, qtype=s.qtype, score=s.score, knowledge_point=s.knowledge_point,
        difficulty=s.difficulty, stem=d.stem.strip(), options=[o.strip() for o in d.options],
        answer=d.answer.strip().upper() if s.qtype == QType.CHOICE else d.answer.strip(),
        analysis=d.analysis.strip(), verification=verification,
    )


def fill_slots(
    llm: LLMClient,
    req: ExamRequest,
    slots: list[BlueprintSlot],
    existing: list[Question] | None = None,
    notes: dict[str, str] | None = None,
    tracer: Tracer | None = None,
    max_rounds: int = MAX_WRITE_ROUNDS,
) -> dict[str, Question]:
    """为给定 slot 出题，最多 max_rounds 轮；每轮只重出上一轮不合格或遗漏的 slot。
    新建试卷与局部修改共用这一个入口。"""
    existing = list(existing or [])
    done: dict[str, Question] = {}
    last_rejected: dict[str, Question] = {}
    pending = list(slots)
    feedback: list[str] | None = None
    for round_no in range(1, max_rounds + 1):
        try:
            result = write_questions(llm, req, pending, existing=existing + list(done.values()),
                                     feedback=feedback, tracer=tracer, notes=notes)
        except AgentFailure as e:  # 连 JSON 结构都没出对
            if tracer:
                tracer.log("writer_round_failed", round=round_no, problems=e.problems)
            feedback = e.problems
            continue
        for q in result.questions:
            done[q.slot_id] = q
        last_rejected.update(result.rejected)
        pending = [s for s in slots if s.slot_id not in done]
        feedback = result.problems
        if not pending:
            return done
    # 多轮返修后，个别题格式合格、只是工具或年级规则检查没通过：保留最后一稿，标为“存在未解决问题”交给老师，
    # 而不是让整份试卷失败（d039）。连格式都不合格的题没法放进试卷，仍然报错。
    missing = [s.slot_id for s in pending if s.slot_id not in last_rejected]
    if missing:
        raise AgentFailure("writer", [f"以下 slot 始终未能生成：{missing}"] + (feedback or []))
    for s in pending:
        done[s.slot_id] = last_rejected[s.slot_id]
    if tracer:
        tracer.log("writer_kept_with_problem", slots=[s.slot_id for s in pending], problems=feedback)
    return done
