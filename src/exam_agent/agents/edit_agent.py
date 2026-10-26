"""修改与讨论 Agent（ReAct）：先按需查阅试卷、历史版本和知识库，再决定回答、追问还是提交修改方案。

为什么在这里用 ReAct，而出卷流程不用：
- 出卷流程的每一步做什么是确定的（规划 → 命题 → 审核），由代码编排更可靠；
- 老师的修改和提问千变万化：“讲苹果的那题”“第二篇阅读第 3 小题”“把第 3 题改回原来的”“答案为什么是 B”，
  要看哪些内容事先无法确定，适合让模型自己决定查什么。

分工的边界：
- 模型只能调用只读工具（读题、搜索、统计、看历史、查知识库、程序核对），最后用 submit_plan 提交方案；
- 方案仍由 editing.apply_ops 用代码校验和执行（题号、分值守恒、历史版本是否存在、改字是否唯一定位），
  校验不通过就把具体问题作为工具结果返回给模型，让它修正——模型从不直接写数据库。

停止条件：最多 MAX_STEPS 轮；同一工具同样参数重复调用视为原地打转；方案连续 MAX_SUBMITS 次不通过就停止；
到达上限时强制要求提交，仍不提交就向老师追问。
上下文控制：模型一开始只看到试卷索引（每题一行），全文按需读取；较早的工具结果在后续轮次中压缩为一行摘要。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from ..domain import PaperSnapshot, QType, type_list_text
from .base import Tracer
from .editor import EditPlan

MAX_STEPS = 8          # 一次对话最多几轮“思考 → 调用工具”
MAX_SUBMITS = 3        # 方案最多被代码退回几次
KEEP_FULL = 3          # 最近几条工具结果保留全文，更早的压缩成摘要
OBS_LIMIT = 1500       # 单条工具结果的最大字数
MAX_TOOL_CALLS = 10    # 一次对话最多调用几次只读工具（一轮可以并行调用多个，所以单限轮数不够；e1：统计类问题曾查到 24 次）


@dataclass
class EditWorkspace:
    store: Any
    paper_id: str
    paper: PaperSnapshot
    version: int
    grounding: Any = None

    def q_at(self, pos: int):
        return self.paper.question_by_position(pos)

    def slot_at(self, pos: int):
        return self.paper.blueprint.slot_at(pos)


# ---------------------------------------------------------------- 工具实现（只读）

def _fmt_question(ws: EditWorkspace, pos: int, q=None, slot=None) -> str:
    q = q or ws.q_at(pos)
    s = slot or ws.slot_at(pos)
    v = q.verification
    lines = [f"第{pos}题 | {q.qtype.value} | {q.score}分 | 难度 {s.difficulty.value} | 知识点：{s.knowledge_point}"
             + (f"（{s.kp_id}）" if s.kp_id else ""),
             f"命题意图：{s.intent}", "题干：", q.stem]
    if q.options:
        lines += ["选项："] + [f"{'ABCD'[i]}. {o}" for i, o in enumerate(q.options)]
    lines += [f"答案：{q.answer}", f"解析：{q.analysis}",
              f"验证状态：{v.status.value}（{v.method}{'；' + v.detail if v.detail else ''}）" if v else "验证状态：未审核"]
    return "\n".join(lines)


def tool_get_question(ws: EditWorkspace, position: int) -> str:
    return _fmt_question(ws, int(position))


def _question_text(q, s) -> str:
    return " ".join([q.stem, " ".join(q.options), q.answer, q.analysis, s.knowledge_point, s.intent])


def tool_search_questions(ws: EditWorkspace, query: str) -> str:
    """全文搜索（题干、选项、答案、解析、知识点）：原文直接包含的排在前面，其余按 BM25 打分。"""
    from ..retrieval.bm25 import BM25

    slots = sorted(ws.paper.blueprint.slots, key=lambda s: s.position)
    qs = {q.slot_id: q for q in ws.paper.questions}
    texts = [_question_text(qs[s.slot_id], s) for s in slots]
    exact = [i for i, t in enumerate(texts) if query.lower() in t.lower()]
    ranked = exact + [i for i, sc in BM25(texts).scores(query) if i not in exact and sc > 0][: max(0, 5 - len(exact))]
    if not ranked:
        return f"没有题目包含或接近“{query}”"
    out = []
    for i in ranked[:6]:
        t = texts[i]
        k = t.lower().find(query.lower())
        snippet = t[max(0, k - 30): k + len(query) + 30] if k >= 0 else t[:60]
        out.append(f"第{slots[i].position}题（{slots[i].qtype.value}）{'【原文包含】' if i in exact else ''}："
                   f"…{snippet.replace(chr(10), ' ')}…")
    return "\n".join(out)


def tool_paper_stats(ws: EditWorkspace) -> str:
    from ..tools.english import GRADE_LEVEL, LIMITS, english_words, passage_of
    from .requirement import CN_GRADES

    p = ws.paper
    slots = sorted(p.blueprint.slots, key=lambda s: s.position)
    qs = {q.slot_id: q for q in p.questions}
    by_type: dict[str, list[int]] = {}
    for s in slots:
        by_type.setdefault(f"{s.qtype.value}（每题 {s.score} 分）", []).append(s.position)
    lines = [f"共 {len(slots)} 题，满分 {p.total_score}（要求 {p.request.total_score}）",
             "题型：" + "；".join(f"{k} 第{'、'.join(map(str, v))}题" for k, v in by_type.items()),
             "难度：" + "，".join(f"{d} {n} 题" for d, n in Counter(s.difficulty.value for s in slots).items())]
    choice = [(s.position, qs[s.slot_id].answer) for s in slots if s.qtype == QType.CHOICE]
    if choice:
        seq = [a for _, a in choice]
        run = max((len(m.group(0)) for m in re.finditer(r"(.)\1*", "".join(seq))), default=0)
        lines.append("选择题答案：" + " ".join(f"{pos}:{a}" for pos, a in choice)
                     + f"；分布 {dict(Counter(seq))}；最长连续相同 {run} 个")
    kp: dict[str, list[int]] = {}
    for s in slots:
        kp.setdefault(s.knowledge_point, []).append(s.position)
    dup = {k: v for k, v in kp.items() if len(v) > 1}
    lines.append(f"知识点 {len(kp)} 个" + ("；重复考查：" + "；".join(f"{k} 第{'、'.join(map(str, v))}题"
                                                             for k, v in dup.items()) if dup else "；没有重复"))
    st: dict[str, list[int]] = {}
    for s in slots:
        st.setdefault(qs[s.slot_id].vstatus.value, []).append(s.position)
    lines.append("验证状态：" + "；".join(f"{k} 第{'、'.join(map(str, v))}题" for k, v in st.items()))
    need = [s.position for s in slots if qs[s.slot_id].vstatus.value in ("待老师确认", "存在未解决问题", "未审核")]
    lines.append("需要老师人工确认的题：" + ("第" + "、".join(map(str, need)) + "题" if need else "无")
                 + "（“工具验证”“独立复核一致”“老师已确认”都不需要再确认）")
    if p.request.subject == "英语":
        g = CN_GRADES.index(p.request.grade[:1]) + 1 if p.request.grade[:1] in CN_GRADES else 0
        lim = LIMITS[GRADE_LEVEL.get(g, 1)]
        reads = [f"第{s.position}题 {len(english_words(passage_of(qs[s.slot_id].stem)))} 词"
                 for s in slots if s.qtype == QType.READING]
        if reads:
            lines.append(f"阅读短文词数（要求 {lim.passage_min}～{lim.passage_max}）：" + "，".join(reads))
    if p.request.constraints:
        lines.append("持续约束：" + "；".join(c.text for c in p.request.constraints))
    return "\n".join(lines)


def tool_get_history(ws: EditWorkspace, position: int | None = None) -> str:
    """版本历史。给出题号时，列出这道题在哪些版本里内容发生了变化（找“原来的”是哪一版）。"""
    hist = ws.store.history(ws.paper_id)
    if position is None:
        return "\n".join(f"v{h['version']}（{h['operation']}）：{h['summary']}" for h in hist[-10:])
    sid = ws.slot_at(int(position)).slot_id
    lines, last = [], None
    for h in hist:
        snap, _ = ws.store.load(ws.paper_id, h["version"])
        q = next((x for x in snap.questions if x.slot_id == sid), None)
        fp = q.content_fingerprint() if q else None
        if fp != last:
            what = f"{q.qtype.value}：{q.stem[:40].replace(chr(10), ' ')}… 答案 {q.answer[:15]}" if q else "（这一版还没有这道题）"
            lines.append(f"v{h['version']}（{h['operation']}：{h['summary'][:40]}）→ 第{position}题内容：{what}")
            last = fp
    lines.append(f"当前是 v{ws.version}。以上只列出这道题内容发生变化的版本；“改回原来的”通常指最近一次变化之前的那一版")
    return "\n".join(lines)


def tool_get_question_version(ws: EditWorkspace, position: int, version: int) -> str:
    sid = ws.slot_at(int(position)).slot_id
    snap, _ = ws.store.load(ws.paper_id, int(version))
    q = next((x for x in snap.questions if x.slot_id == sid), None)
    if q is None:
        return f"第 {version} 版里没有这道题"
    return f"【v{version} 中的这道题】\n" + _fmt_question(ws, int(position), q, snap.blueprint.slot(sid))


REL = {"current": "本次出卷范围内", "prior": "前面已学过（可以使用）", "later": "后面才学（超纲）"}


def tool_lookup_scope(ws: EditWorkspace, query: str = "", position: int | None = None) -> str:
    """查课程知识库：知识点在哪册哪单元、相对本卷范围是本单元 / 已学 / 超纲。"""
    g = ws.grounding
    sub = g.by_subject.get(ws.paper.request.subject) if g is not None and hasattr(g, "by_subject") else g
    if sub is None:
        return "当前没有加载这个学科的课程知识库，无法查询范围"
    cands = sub.candidates(ws.paper.request)
    if cands is None:
        return "课程知识库没有覆盖这个年级"
    lines = [f"本卷范围：{cands.scope.label()}"]
    if position is not None:
        s = ws.slot_at(int(position))
        if s.kp_id and s.kp_id in sub.kb.by_id:
            p = sub.kb.by_id[s.kp_id]
            lines.append(f"第{position}题对应卡片 {p.id} {p.location}{'' if p.name in p.location else '·' + p.name}：{REL[sub.kb.relation(p.id, cands.scope)]}")
            if s.scope_note:
                lines.append(f"范围说明：{s.scope_note[:200]}")
        query = query or f"{s.knowledge_point} {ws.q_at(int(position)).stem[:60]}"
    if query:
        hits = sub.retriever.search(query, 4, sub.mode, where={"kind": "kp"})
        for h in hits:
            p = h.doc.obj
            if p.unit is None:
                continue
            name = "" if p.name in p.location else f"·{p.name}"
            lines.append(f"- {p.id} {p.location}{name}：{REL[sub.kb.relation(p.id, cands.scope)]}；"
                         f"{'；'.join(p.key_points[:2])[:80]}")
    return "\n".join(lines)


def tool_check_question(ws: EditWorkspace, position: int) -> str:
    """用程序工具重新核对一道题：算式 / 方程精确计算、英语语法级别与短文字数、格式。不调用模型。"""
    from ..invariants import check_question_format
    from ..verify import route

    pos = int(position)
    q = ws.q_at(pos)
    out = [str(v) for v in check_question_format(q.qtype, q.stem, q.options, q.answer, q.slot_id)]
    r = route(q, None)
    out.append(f"工具核对：{r.verification.status.value}（{r.verification.method}"
               f"{'；' + r.verification.detail if r.verification.detail else ''}）" + (f"；问题：{r.mismatch}" if r.mismatch else ""))
    if ws.paper.request.subject == "英语":
        from ..tools.english import check_question as check_en, scope_end_book
        from .requirement import CN_GRADES

        req = ws.paper.request
        g = CN_GRADES.index(req.grade[:1]) + 1 if req.grade[:1] in CN_GRADES else 0
        issues = check_en(q.qtype.value, q.stem, q.options, q.answer, g, scope_end_book(req.grade, req.term))
        out += [f"英语检查（{i.severity}）：{i.message}" for i in issues] or ["英语检查：语法级别、短文字数、生词比例均符合"]
    return "\n".join(out)


def _spec(name: str, desc: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


POS = {"type": "integer", "description": "当前卷面题号"}
OP_DOC = ("修改操作列表。每个操作形如 {\"op\": 操作名, ...}：\n"
          "rewrite（重出，positions，可带 difficulty / knowledge_point / instruction）；"
          "retype（改题型，positions 1 个，new_qtype）；add（新增同一题型的题，new_qtype、knowledge_point，count 为新增几道，默认 1）；"
          "resize（老师说“扩大到 N 题”“一共要 N 道”而且保留现有题时用：count=N 为调整后的总题数，"
          "新增几道、各是什么题型由程序计算，不要自己算差几道）；remove（删除，positions）；"
          "rescore（scores 如 {\"选择题\": 4}）；set_total（total）；"
          "add_constraint（constraint 如 {\"kind\": \"integer_answer\", \"text\": \"答案为整数\", \"scope_types\": [\"计算题\"]}，"
          "不能程序检验的用 kind=free_text）；"
          "restore（把一道题恢复成旧版本：positions 1 个，from_version）；"
          "patch（局部改字，不重出整题：positions 1 个，patches 为 [{\"field\": \"stem|options|answer|analysis\", "
          "\"old\": 原文中恰好出现一次的一段, \"new\": 新文字}]；改了题干数字时必须同时改答案和解析）")
TOOLS = [
    _spec("get_question", "读取一道题的完整内容：题干、选项、答案、解析、知识点、验证状态。回答关于某道题的问题前必须先读。",
          {"position": POS}, ["position"]),
    _spec("search_questions", "按内容在全卷中搜索题目（题干、选项、答案、解析、知识点、阅读短文），用于“讲苹果的那题”“答案是 12 的题”这类说法。",
          {"query": {"type": "string"}}, ["query"]),
    _spec("paper_stats", "全卷统计：题型与分值、难度分布、选择题答案分布、知识点重复、验证状态、英语短文词数、持续约束。"),
    _spec("get_history", "版本历史。给出 position 时，列出这道题内容发生变化的各个版本，用于“改回原来的”“上一版那道”。",
          {"position": POS}),
    _spec("get_question_version", "读取某道题在指定历史版本中的完整内容。", {"position": POS, "version": {"type": "integer"}},
          ["position", "version"]),
    _spec("lookup_scope", "查课程知识库：某知识点或某道题属于哪册哪单元，相对本卷范围是本单元、已学过还是超纲。",
          {"query": {"type": "string", "description": "知识点或关键词"}, "position": POS}),
    _spec("check_question", "用程序工具重新核对一道题（精确计算 / 解方程、英语语法级别与短文字数、格式），不经过模型。",
          {"position": POS}, ["position"]),
    _spec("submit_plan", "结束本轮，提交结果。只回答或讨论时 action=answer；需要老师澄清时 action=clarify；"
                         "老师要求修改或同意了你的建议时 action=edit 并给出 ops；要导出 PDF 时 action=export。", {
        "action": {"type": "string", "enum": ["edit", "answer", "clarify", "export"]},
        "ops": {"type": "array", "items": {"type": "object"}, "description": OP_DOC},
        "reply": {"type": "string", "description": "对老师说的话"}}, ["action", "reply"]),
]
SUBMIT_ONLY = [t for t in TOOLS if t["function"]["name"] == "submit_plan"]
FORCE_SUBMIT = {"type": "function", "function": {"name": "submit_plan"}}
IMPL: dict[str, Callable[..., str]] = {
    "get_question": tool_get_question, "search_questions": tool_search_questions, "paper_stats": tool_paper_stats,
    "get_history": tool_get_history, "get_question_version": tool_get_question_version,
    "lookup_scope": tool_lookup_scope, "check_question": tool_check_question,
}
LABEL = {"get_question": "查看第 {position} 题全文", "search_questions": "在试卷中搜索“{query}”",
         "paper_stats": "统计全卷", "get_history": "查看版本历史", "get_question_version": "查看第 {position} 题的 v{version}",
         "lookup_scope": "查询课程知识库", "check_question": "用程序工具核对第 {position} 题",
         "submit_plan": "形成答复 / 修改方案"}


SYSTEM = """你是中小学试卷编制助手中的“试卷修改与讨论助手”。老师正在和你讨论一份已经生成的试卷，
可能是提问（这题考什么、答案为什么是 B、难度合不合适、有没有超纲）、请你评价和提建议，也可能是要求修改。

工作方式：先想清楚需要看什么，再调用工具查阅，最后调用 submit_plan 结束。
1. 你一开始只看到每题一行的索引。凡是要谈论某道题的具体内容（解释、评价、判断对错、改字），必须先用 get_question 读全文，
   不能凭索引或记忆编造。老师用内容指代题目（“讲苹果的那题”“第二篇阅读”）时用 search_questions 定位。
2. 判断是否超纲用 lookup_scope；判断答案或计算是否正确用 check_question（程序核对，比你自己算可靠）；
   谈整体情况（难度分布、答案分布、重复、哪些题要老师确认、短文词数）用 paper_stats，它已经汇总好了，
   不要为此逐题 get_question；“改回原来的 / 上一版那道”先用 get_history 找到版本。
   一次对话最多查阅 """ + str(MAX_TOOL_CALLS) + """ 次，够用就提交。
3. 先判断老师这句话是“要求修改”还是“提问 / 请你评价”：
   - 含有“换、改、删、加、恢复、改回、改成、调整”等动词的祈使句，就是要求修改：能确定是哪道题时直接提交 action=edit，
     不要再反问“要不要改”“怎么改”，也不要只分析不动手。“换一道”“改简单些”这类没说具体怎么改的，用 rewrite，
     把你判断的方向写进 instruction（如“换一道考同一知识点、难度为基础的新题”）。
   - 提问或请你评价时，action=answer，给出有依据的回答；可以提出具体修改建议并问老师是否照此修改，但不要擅自改卷。
     老师随后表示同意（“好”“就这么改”），按你上一轮的建议提交 action=edit。
   - 老师要求修改，但这句话、题目内容和最近的对话都不能唯一确定是哪道题（如“这个不太好，改一下”“那个阅读改一下”
     而卷中有两篇阅读），必须 action=clarify，绝不能猜一道题去改。
   - clarify 只用于“缺了老师的信息就无法继续”；已经回答了问题、只是顺带问一句“要不要我改”时，用 answer。
4. 修改方式：老师给出了具体的原文和新文字（“278 改成 268”“选项 D 改成 30”），用 patch；要换一道新题用 rewrite；
   恢复旧版用 restore。patch 的 old 必须是原文中的一段，不要带选项字母前缀（写“18”，不要写“D. 18”）。
   题号是当前卷面题号。
5. 分值重配、题数总分守恒由程序完成，除非老师明确指定分值，不要用 rescore。
   老师说的是“调整后一共多少题”时用 resize，不要用多个 add 自己凑数量。
   老师提出对整卷持续有效的要求（“计算题答案都要是整数”）时，一律提交 add_constraint 记录下来，
   即使现有题目已经满足也要记录（以后改题、加题都要遵守）；程序会自动核验现有题目。
   action=edit 的 reply 说明将要做的修改，不要在 reply 里再征求同意（征求同意就用 answer 且不给 ops）。
   老师要求“增加某类题”但题数不变时，用 retype 把一道最合适的现有题改成该题型，并在 reply 说明选了哪题、为什么。
6. 指代不明、无法判断时 action=clarify 追问。题号明确、无需查阅就能确定的修改，直接提交，不要多余地调用工具。
7. submit_plan 被程序退回时，按退回原因修正后重新提交；如果需要老师决定，向老师原样转述退回原因。
   不要自己做加减乘除去算分值、总分或答案（模型心算不可靠）：总分、分值用 paper_stats，答案用 check_question，
   程序给出的数字原样引用，不要编造数字来解释原因；没有真正执行的操作，不能说“已经做了”。
8. difficulty 只能是 基础/中等/提高；题型只能是本学科的可用题型（""" + type_list_text() + """）。"""


def paper_index(paper: PaperSnapshot) -> str:
    r = paper.request
    qs = {q.slot_id: q for q in paper.questions}
    lines = [f"试卷：{paper.title}；{r.subject}；题数 {r.question_count}；满分 {r.total_score}",
             "持续约束：" + ("；".join(c.text for c in r.constraints) or "无"), "题目索引（全文用 get_question 读取）："]
    for s in sorted(paper.blueprint.slots, key=lambda s: s.position):
        q = qs[s.slot_id]
        lines.append(f"第{s.position}题 | {s.qtype.value} | {s.score}分 | {s.difficulty.value} | {s.knowledge_point} | "
                     f"{q.vstatus.value} | {q.stem[:30].replace(chr(10), ' ')}")
    return "\n".join(lines)


def _digest(name: str, args: dict, text: str) -> str:
    first = text.splitlines()[0] if text else ""
    return f"（较早的工具结果已压缩）{name}({json.dumps(args, ensure_ascii=False)}) → {first[:80]}"


@dataclass
class AgentRun:
    messages: list[dict] = field(default_factory=list)
    digests: dict[int, str] = field(default_factory=dict)   # 消息下标 → 压缩后的内容

    def view(self) -> list[dict]:
        """发给模型的消息：最近 KEEP_FULL 条工具结果保留全文，更早的换成一行摘要。"""
        tool_idx = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        old = set(tool_idx[:-KEEP_FULL]) if len(tool_idx) > KEEP_FULL else set()
        return [({**m, "content": self.digests[i]} if i in old and i in self.digests else m)
                for i, m in enumerate(self.messages)]


def _parse_plan(args: str) -> tuple[EditPlan | None, str]:
    try:
        data = json.loads(args or "{}")
        return EditPlan.model_validate(data), ""
    except (ValueError, ValidationError) as e:
        return None, f"方案格式不正确：{str(e)[:300]}"


def run_edit_agent(llm, ws: EditWorkspace, message: str, validate: Callable[[EditPlan], list[str]] | None = None,
                   tracer: Tracer | None = None, history_turns: int = 4) -> EditPlan:
    turns = ws.store.recent_turns(ws.paper_id, limit=history_turns)
    user = [paper_index(ws.paper)]
    if turns:
        user.append("最近的对话：")
        user += [f"{'老师' if h['role'] == 'user' else '助手'}：{h['content'][:500]}" for h in turns]
    user.append(f"老师的新消息：{message}")
    run = AgentRun([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(user)}])
    seen: set[str] = set()
    submits = repeats = 0

    def record(turn) -> None:
        if tracer:
            tracer.totals["llm_calls"] += 1
            tracer.totals["prompt_tokens"] += turn.prompt_tokens
            tracer.totals["completion_tokens"] += turn.completion_tokens

    tool_calls = 0
    for step in range(1, MAX_STEPS + 2):
        final = step > MAX_STEPS or tool_calls >= MAX_TOOL_CALLS
        if final:
            run.messages.append({"role": "user", "content": "已达到查阅上限。请根据已有信息立即调用 submit_plan；"
                                                            "信息不足就用 action=clarify 向老师说明还需要什么。"})
        turn = llm.chat_tools(run.view(), SUBMIT_ONLY, FORCE_SUBMIT) if final else llm.chat_tools(run.view(), TOOLS)
        record(turn)
        if not turn.tool_calls:
            # 模型直接用文字回复：让它用 submit_plan 归类（回答 / 追问 / 修改），避免把追问或该执行的修改当成回答
            text = turn.content.strip()
            if tracer:
                tracer.log("edit_agent_step", round=step, tool="reply", label="形成答复")
            run.messages.append({"role": "assistant", "content": text})
            run.messages.append({"role": "user", "content": "请把上面的内容通过 submit_plan 提交：老师要求修改就 action=edit "
                                                            "并给出 ops；你在向老师提问就 action=clarify；否则 action=answer。"
                                                            "reply 保持上面的内容。"})
            turn2 = llm.chat_tools(run.view(), SUBMIT_ONLY, FORCE_SUBMIT)
            record(turn2)
            call = next((c for c in turn2.tool_calls if c.name == "submit_plan"), None)
            plan, err = _parse_plan(call.arguments) if call else (None, "未提交")
            if plan and not (validate(plan) if validate else []):
                return plan
            return EditPlan(action="answer", reply=text or "抱歉，我没能理解，请换个说法。")
        run.messages.append(turn.as_message())
        for call in turn.tool_calls:
            try:
                args = json.loads(call.arguments or "{}")
            except ValueError:
                args = None
            if call.name == "submit_plan":
                plan, err = _parse_plan(call.arguments)
                problems = [err] if err else (validate(plan) if validate else [])
                if tracer:
                    tracer.log("edit_agent_step", round=step, tool="submit_plan", label=LABEL["submit_plan"],
                               problems=problems)
                if not problems:
                    return plan
                submits += 1
                result = "方案未通过程序校验，请修正后重新提交：" + "；".join(problems)
                if submits >= MAX_SUBMITS:
                    return EditPlan(action="clarify", reply="我没能把这个修改整理成可执行的方案（" +
                                    "；".join(problems)[:150] + "）。能换个说法，或说得更具体一些吗？")
            elif call.name not in IMPL or args is None:
                tool_calls += 1
                result = f"没有这个工具或参数不是 JSON：{call.name}"
            else:
                tool_calls += 1
                key = f"{call.name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                if key in seen:
                    repeats += 1
                    result = "这个查询刚才已经做过，结果同上。请根据已有信息继续，或调用 submit_plan。"
                else:
                    seen.add(key)
                    try:
                        result = IMPL[call.name](ws, **args)
                    except (KeyError, TypeError, ValueError) as e:
                        result = f"工具调用出错：{e}"
                    result = result[:OBS_LIMIT] + ("…（已截断）" if len(result) > OBS_LIMIT else "")
                if tracer:
                    label = LABEL.get(call.name, call.name)
                    try:
                        label = label.format(**{"position": "?", "query": "", "version": "?", **args})
                    except (KeyError, IndexError):
                        pass
                    tracer.log("edit_agent_step", round=step, tool=call.name, label=label, args=args,
                               result=result[:200])
            run.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            run.digests[len(run.messages) - 1] = _digest(call.name, args or {}, result)
        if repeats >= 2:   # 原地打转：直接进入最后一步，强制提交
            run.messages.append({"role": "user", "content": "你在重复同样的查询。请立即调用 submit_plan。"})
            turn = llm.chat_tools(run.view(), SUBMIT_ONLY, FORCE_SUBMIT)
            record(turn)
            call = next((c for c in turn.tool_calls if c.name == "submit_plan"), None)
            plan, err = _parse_plan(call.arguments) if call else (None, "未提交")
            if plan and not (validate(plan) if validate else []):
                return plan
            return EditPlan(action="clarify", reply="我没能确定要怎么处理，能说得更具体一些吗？")
    return EditPlan(action="clarify", reply="我没能确定要怎么处理，能说得更具体一些吗？")
