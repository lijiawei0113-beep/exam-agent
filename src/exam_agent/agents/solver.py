"""盲解 Agent：只看题干（和选项），看不到标准答案，独立作答。

它的价值来自信息隔离：
- 写不出方程或觉得条件不足 → 题干可能有歧义或缺条件（如“下列各数”却没给数）；
- 它列出的方程交给精确求解器计算，得到与命题员无关的独立答案；
- 选择题独立选出的选项与答案键不一致 → 答案键或题目有问题。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..domain import Question
from ..llm import LLMClient
from .base import Tracer, structured_call


class BlindSolution(BaseModel):
    slot_id: str
    equations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    target: str = ""
    expression: str = Field(default="", description="不需要设未知数、一个算式就能算出答案时填写")
    choice: str = ""
    final_answer: str = ""
    issue: str = Field(default="", description="题干有歧义、缺条件或自相矛盾时说明；没有问题留空")
    ambiguous_items: list[str] = Field(default_factory=list,
                                       description="除了你选的答案，还有别的选项也能成立时，写明哪一题、哪几个选项、依据")
    option_exprs: dict[str, str] = Field(default_factory=dict, description="比较类选择题：每个选项对应的算式")
    pick: str = Field(default="", description="比较类选择题：选最大填 max，选最小填 min")
    leak: str = Field(default="", description="题干的中文提示或翻译直接给出了答案时说明")


class BlindOutput(BaseModel):
    solutions: list[BlindSolution]


SYSTEM = """你是一名认真的学生，正在独立做题。你看不到标准答案。
对每道题：
0. 如果不需要设未知数、列一个算式就能算出最终答案（小学计算题、解决问题常见），
   在 expression 中写这个算式，只用数字、+ - × ÷ ( ) 和 %，不写单位和中文，
   例如“每盒 6 支，4 盒一共几支”写 "6×4"，“17 个苹果每人分 5 个，能分给几人、还剩几个”写 "17÷5"。
   expression 和 equations 选一种写即可。
1. 如果题目可以通过列方程求解（包括解方程题、填空题、应用题、求参数的题），
   在 equations 中写出方程（每个方程恰好一个等号），unknowns 写所设的未知数（单个字母），
   target 写题目最终要求的那个未知数。方程只用数字、字母、+ - * / ( ) 和 =，不写单位和中文，
   例如应用题“设经过 t 小时相遇”写成 "60*t + 80*t = 240"。
   如果题目给出了“已知某数是方程的解”，把这个解代入后再列方程。
   注意：只有当题目最终所求的就是某个未知数的值时才写 equations；
   如果所求是中间步骤（如“移项得 5x - 2x = 6 + ____”）、代数式或其他内容，equations 留空，
   只在 final_answer 中写你的答案。
2. 选择题在 choice 中写你选的字母（A/B/C/D）；如果正确选项是一个能算出来的数（如“3 分 = 多少秒”），
   同时在 expression 写算式（如 "3×60"）。
3. final_answer 写你的最终答案。英语填空题写你填的单词（多个空用“; ”分隔）；
   阅读理解写成“1. B 2. C 3. A”；书面表达不用写，final_answer 留空。
【逐项排查】选择题和阅读理解的每个小题，做完后再逐个看其他选项：如果另一个选项也能成立
   （例如短文同时写了 "I'm hungry" 和 "I'm happy"，问 "How does he feel?"；或两个选项意思相同），
   在 ambiguous_items 里写明“第几小题：哪几个选项都能成立，依据是……”。干扰项只是有迷惑性、但明确错误的，不要报告。
【答案泄露】英语题如果题干里的中文提示或翻译直接说出了答案（例如 "The apple is ____.（这个苹果是红色的。）"
   选项里有 red，不懂英语也能选对），在 leak 中说明；只提示词义、需要学生自己写出或辨认英文的（如“鼻子：____”）不算。
   数学题也一样：要求填写的结果已经直接写在题干里（如“盘子里有 3 个苹果，苹果有 ____ 个”），在 leak 中说明。
【比较类选择题】如果题目是比较几个量谁最大 / 最小（如统计表里哪组人数最多），
   在 option_exprs 中为每个选项写出它对应数量的算式（如 {"A": "8+10", "B": "6+9", "C": "12+7"}），
   pick 填 max 或 min；程序会精确计算后自动选出答案。
4. 如果题干有歧义、缺少必要条件、自相矛盾，或者引用了题中不存在的内容（如“下列各数”却没有给出），
   在 issue 中具体说明；题目没问题时 issue 留空。不要因为题目简单就报告问题。
5. 只输出 json，示例：
{"solutions": [{"slot_id": "S9", "equations": ["60*t + 80*t = 240"], "unknowns": ["t"], "target": "t",
  "expression": "", "choice": "", "final_answer": "12/7 小时", "issue": ""},
 {"slot_id": "S3", "equations": [], "unknowns": [], "target": "", "expression": "(120-30)÷3",
  "choice": "", "final_answer": "30 元", "issue": ""}]}"""


def blind_solve(llm: LLMClient, questions: list[Question], tracer: Tracer | None = None) -> dict[str, BlindSolution]:
    lines = []
    for q in questions:
        lines.append(f"[{q.slot_id}]（{q.qtype.value}）{q.stem}")
        for letter, opt in zip("ABCD", q.options):
            lines.append(f"    {letter}. {opt}")
    ids = {q.slot_id for q in questions}

    def validate(out: BlindOutput) -> list[str]:
        missing = ids - {s.slot_id for s in out.solutions}
        return [f"遗漏了这些题：{sorted(missing)}"] if missing else []

    out = structured_call(llm, step="blind_solver", system=SYSTEM, user="\n".join(lines),
                          schema=BlindOutput, validate=validate, tracer=tracer).value
    return {s.slot_id: s for s in out.solutions if s.slot_id in ids}
