"""修改意图规划 Agent：把老师的一句话翻译成结构化的修改操作（EditPlan）。

它只决定“改什么”，不直接改题：
- 语义判断（改哪道、换成什么、难度怎么变）交给模型；
- 题号定位、分值重配、一致性校验交给确定性代码（editing.py）。
模型看到的是试卷摘要（来自数据库的当前版本），而不是整卷全文，以控制上下文长度。
"""
from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, Field

from ..domain import Difficulty, PaperConstraint, PaperSnapshot, QType, type_list_text
from ..llm import LLMClient
from .base import Tracer, structured_call

OpName = Literal["rewrite", "retype", "add", "remove", "rescore", "set_total", "add_constraint", "restore", "patch",
                 "resize"]


class TextPatch(BaseModel):
    """局部修改：在题目的某个字段里把一段原文替换成新文字（原文必须恰好出现一次）。"""
    field: Literal["stem", "options", "answer", "analysis"]
    old: str
    new: str


class EditOp(BaseModel):
    op: OpName
    positions: list[int] = Field(default_factory=list, description="涉及的当前卷面题号")
    new_qtype: QType | None = None
    difficulty: Difficulty | None = None
    knowledge_point: str | None = None
    instruction: str = Field(default="", description="交给命题员的具体要求")
    scores: dict[QType, int] = Field(default_factory=dict, description="rescore：题型→每题分值")
    total: int | None = None
    count: int | None = Field(default=None, description="add：一次新增几道同类题；resize：调整后的总题数")
    constraint: PaperConstraint | None = None
    from_version: int | None = Field(default=None, description="restore：把这道题恢复成第几版的内容")
    patches: list[TextPatch] = Field(default_factory=list, description="patch：逐处文字替换")


class EditPlan(BaseModel):
    action: Literal["edit", "answer", "clarify", "export"]
    ops: list[EditOp] = Field(default_factory=list)
    reply: str = Field(description="对老师说的话：edit 时说明将做什么；answer/clarify 时是回答或追问")


SYSTEM = """你是中小学命题助手中的“修改规划员”。老师正在修改一份已经生成的试卷。
你的任务是把老师的话翻译成结构化的修改操作，只输出 json。

可用操作（op）：
- rewrite：重出指定题（题型和分值不变）。positions 为题号列表；可同时修改 difficulty、knowledge_point；
  instruction 写清楚新题要满足的要求。
- retype：把一道题改成另一种题型。positions 只能有 1 个；必须给出 new_qtype；可选 difficulty、knowledge_point。
- add：新增一道题（题数加 1）。必须给出 new_qtype 和 knowledge_point，可选 difficulty。
- remove：删除指定题（题数减少）。
- rescore：老师明确指定某类题型的每题分值时使用，scores 形如 {"选择题": 4}。
- set_total：老师修改满分时使用，给出 total。
- add_constraint：老师提出对后续持续有效的要求时使用，如“计算题答案都要是整数”，
  constraint 形如 {"kind": "integer_answer", "text": "答案为整数", "scope_types": ["计算题"]}；
  不能由程序检验的要求用 kind=free_text。

规则：
1. 题号是试卷摘要中的当前题号。只改老师提到的题；老师说“其他题别动”时尤其不能多改。
2. 老师要求“增加某类题”但同时要求题数不变时，不要用 add，而是选一道最合适的现有题用 retype 改成该题型，
   并在 reply 中说明选了哪题、为什么。选题优先级：
   (a) 本身就是实际情境问题、却被标成其他题型的题（改动最小）；
   (b) 题型占比过高的那类题中，与其他题考查内容重复的题。
   不要选考查独特知识点的题；新题的 knowledge_point 不能与其他题重复。
3. add_constraint 之后，程序会自动用求解工具核验现有题目，并只修改不符合约束的题。
   reply 中直接说明将这样做，不要反问老师是否需要修改。
4. 分值的重新分配由程序自动完成，保证满分不变且同类题分值相同；除非老师明确指定分值，否则不要用 rescore。
5. 老师只是提问、不需要改卷时，action=answer，在 reply 中直接回答。
   老师要求导出、打印、生成 PDF、要学生版或答案版时，action=export，ops 为空，reply 说明将导出两份 PDF。
6. 指代不明确、无法判断要改哪道题时，action=clarify，在 reply 中追问。
7. difficulty 只能是 基础/中等/提高；题型只能是本学科的可用题型（""" + type_list_text() + """）。
8. 输出 json 示例：
{"action": "edit",
 "ops": [{"op": "rewrite", "positions": [3], "difficulty": "基础",
          "instruction": "原题偏难，换成一道只需一步移项即可求解的基础题"}],
 "reply": "好的，我将把第 3 题换成一道基础题，其他题保持不变。"}"""


def summarize_paper(paper: PaperSnapshot) -> str:
    """给规划员看的试卷摘要：结构信息 + 题干开头，不含完整解析，控制 token。"""
    r = paper.request
    lines = [
        f"试卷：{paper.title}；题数 {r.question_count}；满分 {r.total_score}；"
        f"老师要求的题型：{'、'.join(t.value for t in r.required_types) or '未指定'}",
        "持续约束：" + ("；".join(c.text for c in r.constraints) or "无"),
        "当前各题：",
    ]
    qs = {q.slot_id: q for q in paper.questions}
    for s in sorted(paper.blueprint.slots, key=lambda s: s.position):
        q = qs[s.slot_id]
        lines.append(f"第{s.position}题 | {s.qtype.value} | {s.score}分 | {s.difficulty.value} | "
                     f"{s.knowledge_point} | {q.stem[:50]} | 答案：{q.answer[:15]}")
    return "\n".join(lines)


def plan_edit(
    llm: LLMClient,
    paper: PaperSnapshot,
    message: str,
    history: list[dict] | None = None,
    validate: Callable[[EditPlan], list[str]] | None = None,
    tracer: Tracer | None = None,
) -> EditPlan:
    parts = [summarize_paper(paper)]
    if history:
        parts.append("最近的对话：")
        parts += [f"{'老师' if h['role'] == 'user' else '助手'}：{h['content'][:120]}" for h in history]
    parts.append(f"老师的新消息：{message}")
    return structured_call(
        llm, step="edit_planner", system=SYSTEM, user="\n".join(parts),
        schema=EditPlan, validate=validate, tracer=tracer,
    ).value
