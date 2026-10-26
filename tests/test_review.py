"""阶段三：工具验题与审核返修闭环。测试用例取自真实运行中出现过的问题。"""
import json

from exam_agent.agents.solver import BlindSolution
from exam_agent.domain import ConstraintKind, ExamRequest, PaperConstraint, QType, Question
from exam_agent.editing import apply_edit
from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.review import check_one
from exam_agent.store import PaperStore

from fixtures import GOOD_BLUEPRINT, Q1, Q2, REQUEST_REPLY, q

j = lambda o: json.dumps(o, ensure_ascii=False)
REQ = ExamRequest.model_validate(json.loads(REQUEST_REPLY)["request"])
INT_REQ = REQ.model_copy(update={"constraints": [
    PaperConstraint(kind=ConstraintKind.INTEGER_ANSWER, text="计算题答案为整数", scope_types=[QType.CALC])]})


def calc(stem, answer, slot="S4"):
    return Question(slot_id=slot, qtype=QType.CALC, score=10, knowledge_point="k", difficulty="中等",
                    stem=stem, answer=answer, analysis="略")


def blind(slot="S4", eq=None, unknowns=("x",), choice="", issue=""):
    return BlindSolution(slot_id=slot, equations=list(eq or []), unknowns=list(unknowns),
                         target=unknowns[0] if unknowns else "", choice=choice, issue=issue)


# ---------- 单题核验 ----------

def test_tool_catches_wrong_answer_from_real_run():
    # 真实运行 p_dba47cbc 第 9 题：答案写 1.5 小时，实际是 12/7 小时
    qq = calc("甲乙两地相距240千米……经过多少小时两车相遇？", "1.5小时")
    fs, verified = check_one(qq, blind(eq=["60*t + 80*t = 240"], unknowns=("t",)), REQ)
    assert [f.code for f in fs] == ["wrong_answer"] and not verified
    assert "12/7" in fs[0].message


def test_correct_answer_is_verified_without_findings():
    fs, verified = check_one(calc("解方程：3x + 5 = 2x - 1", "x = -6"), blind(eq=["3x + 5 = 2x - 1"]), REQ)
    assert fs == [] and verified


def test_integer_constraint_flags_fraction_answer():
    # 真实运行第 7 题：x = 22/5
    qq = calc("解方程：(2x-1)/3 - (x+2)/4 = 1", "x = 22/5")
    fs, _ = check_one(qq, blind(eq=["(2x-1)/3 - (x+2)/4 = 1"]), INT_REQ)
    assert [f.code for f in fs] == ["non_integer"]


def test_integer_constraint_respects_scope():
    app = calc("应用题", "2.5 小时").model_copy(update={"qtype": QType.APPLICATION})
    fs, _ = check_one(app, blind(eq=["2*t = 5"], unknowns=("t",)), INT_REQ)
    assert fs == []   # 约束只作用于计算题


def test_tool_does_not_accuse_when_it_cannot_solve():
    fs, verified = check_one(calc("解方程：x² = 4", "x = 2 或 -2"), blind(eq=["x^2 = 4"]), REQ)
    assert fs == [] and not verified


def test_blind_solver_flags_missing_information():
    # 真实运行 p_607acfac 第 5 题：“检验下列各数”却没给出任何数
    fill = Question(slot_id="S2", qtype=QType.FILL, score=5, knowledge_point="k", difficulty="基础",
                    stem="检验下列各数，____ 是方程 4x - 3 = 2x + 1 的解。", answer="2", analysis="略")
    fs, _ = check_one(fill, blind("S2", issue="题目说“下列各数”，但没有给出任何数"), REQ)
    assert [f.code for f in fs] == ["ambiguous"]


def test_blind_choice_mismatch():
    choice = Question(slot_id="S1", qtype=QType.CHOICE, score=5, knowledge_point="k", difficulty="基础",
                      stem="s", options=["a", "b", "c", "d"], answer="A", analysis="略")
    fs, _ = check_one(choice, blind("S1", choice="C"), REQ)
    assert [f.code for f in fs] == ["choice_mismatch"]


# ---------- 返修闭环（端到端）----------

WRONG_S4 = q("S4", "解方程：60t + 80t = 240", "t = 1.5")
GOOD_S3 = q("S3", "解方程：3x + 5 = 2x - 1", "x = -6")


def blind_reply(*items):
    return j({"solutions": [{"slot_id": s, "equations": eq, "unknowns": u, "target": u[0] if u else "",
                             "choice": c, "final_answer": "", "issue": ""} for s, eq, u, c in items]})


BLIND_ALL = [("S1", [], [], "B"), ("S2", ["2x = 6"], ["x"], ""), ("S3", ["3x + 5 = 2x - 1"], ["x"], "")]


def test_create_paper_repairs_wrong_answer_then_passes():
    llm = ScriptedLLM([
        REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": [Q1, Q2, GOOD_S3, WRONG_S4]}),
        blind_reply(*BLIND_ALL, ("S4", ["60t + 80t = 240"], ["t"], "")),      # 第 0 轮：盲解
        j({"issues": []}),                                                      # 第 0 轮：教学审核
        j({"questions": [q("S4", "解方程：60t + 80t = 280", "t = 2")]}),        # 只返修 S4
        blind_reply(("S4", ["60t + 80t = 280"], ["t"], "")),                    # 第 1 轮：只审 S4
        j({"issues": []}),
    ])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分")
    r = res.review
    assert r.stop_reason == "passed" and r.rounds == 1 and r.repaired == {"S4"}
    assert [f.code for f in r.initial] == ["wrong_answer"]
    assert r.tool_verified == {"S2", "S3", "S4"}
    assert next(x for x in res.paper.questions if x.slot_id == "S4").answer == "t = 2"
    assert llm.replies == []
    # 返修提示里带着求解器给出的具体证据
    assert "12/7" in llm.calls[5][-1]["content"]
    # 第 1 轮盲解只看到 S4
    assert "[S1]" not in llm.calls[6][-1]["content"]


def test_review_stops_when_repair_makes_no_progress():
    still_wrong = q("S4", "解方程：60t + 80t = 240", "t = 1.7")
    llm = ScriptedLLM([
        REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": [Q1, Q2, GOOD_S3, WRONG_S4]}),
        blind_reply(*BLIND_ALL, ("S4", ["60t + 80t = 240"], ["t"], "")), j({"issues": []}),
        j({"questions": [still_wrong]}),
        blind_reply(("S4", ["60t + 80t = 240"], ["t"], "")), j({"issues": []}),
    ])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分")
    assert res.review.stop_reason == "no_progress"
    assert [f.code for f in res.review.remaining] == ["wrong_answer"]   # 没修好就如实报告
    assert res.version == 1                                               # 仍然保存，由老师决定


def test_scenario_integer_constraint_fixes_only_offending_question():
    """面试第三句：“计算题答案都要是整数”——更新约束、调用求解工具、只改不符合的题。"""
    frac_s4 = q("S4", "解方程：(2x-1)/3 - (x+2)/4 = 1", "x = 22/5")
    store = PaperStore(":memory:")
    paper = create_paper(ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": [Q1, Q2, GOOD_S3, frac_s4]})]),
                         store, "七年级一元一次方程小测，4 道题，满分 30 分", review=False).paper
    before = {x.slot_id: x.content_fingerprint() for x in paper.questions}

    llm = ScriptedLLM([
        j({"action": "edit", "ops": [{"op": "add_constraint", "constraint": {
            "kind": "integer_answer", "text": "计算题答案为整数", "scope_types": ["计算题"]}}],
           "reply": "已记录，并会核验现有计算题、修改答案不是整数的题。"}),
        # 回溯核验范围内的两道计算题（不做教学审核：老师没让动它们）
        blind_reply(("S3", ["3x + 5 = 2x - 1"], ["x"], ""), ("S4", ["(2x-1)/3 - (x+2)/4 = 1"], ["x"], "")),
        j({"questions": [q("S4", "解方程：(2x-1)/3 - (x+2)/4 = 0", "x = 2")]}),   # 只返修 S4
        blind_reply(("S4", ["(2x-1)/3 - (x+2)/4 = 0"], ["x"], "")),
        j({"issues": []}),                                                         # 新写的题才做教学审核
    ])
    res = apply_edit(llm, store, paper.paper_id, "计算题答案都要是整数")

    after = {x.slot_id: x.content_fingerprint() for x in res.paper.questions}
    assert {s for s in before if before[s] != after[s]} == {"S4"}
    assert res.review.stop_reason == "passed" and res.review.repaired == {"S4"}
    assert [c.kind for c in res.changes] == ["request", "content"]
    assert "新增约束" in res.changes[0].detail
    assert llm.replies == []


# ---------- 命题时的工具自查 ----------

def test_writer_self_check_catches_wrong_answer_before_review():
    """真实运行 p_5c54cb2f 第 9 题：3 - (x-2)/4 = (2x+1)/6，命题员写 41/8，实际 40/7。
    命题时附上方程，求解器当场发现，只让这一题带着正确值重写，不必等到审核阶段。"""
    eq = "3 - (x - 2)/4 = (2x + 1)/6"
    wrong = {**q("S4", f"解方程：{eq}", "x = 41/8"), "check": {"equations": [eq], "unknowns": ["x"], "target": "x"}}
    fixed = {**q("S4", "解方程：3(x - 2) = x + 4", "x = 5"),
             "check": {"equations": ["3(x - 2) = x + 4"], "unknowns": ["x"], "target": "x"}}
    llm = ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": [Q1, Q2, GOOD_S3, wrong]}),
                       j({"questions": [fixed]})])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    assert next(x for x in res.paper.questions if x.slot_id == "S4").answer == "x = 5"
    retry = llm.calls[3][-1]["content"]
    assert "40/7" in retry and '"S4"' in retry and '"S3"' not in retry.split("需要出题的 slot")[1].split("已有题目")[0]


def test_writer_self_check_enforces_integer_constraint():
    from exam_agent.agents.writer import QuestionDraft, self_check
    from exam_agent.domain import Blueprint
    slot = Blueprint.model_validate_json(GOOD_BLUEPRINT).slot("S4")
    d = QuestionDraft.model_validate({**q("S4", "解方程：5x = 22", "x = 22/5"),
                                      "check": {"equations": ["5x = 22"], "unknowns": ["x"], "target": "x"}})
    assert self_check(d, slot, REQ) == ""                      # 无约束：答案正确即可
    assert "不是整数" in self_check(d, slot, INT_REQ)            # 有约束：分数解不合格


def test_writer_self_check_stays_silent_when_tool_cannot_solve():
    from exam_agent.agents.writer import QuestionDraft, self_check
    from exam_agent.domain import Blueprint
    slot = Blueprint.model_validate_json(GOOD_BLUEPRINT).slot("S4")
    d = QuestionDraft.model_validate({**q("S4", "解方程：x² = 4", "x = ±2"),
                                      "check": {"equations": ["x^2 = 4"], "unknowns": ["x"], "target": "x"}})
    assert self_check(d, slot, REQ) == ""


def test_new_problem_after_repair_counts_as_progress():
    """真实运行 p_748b5eba：第 7 题算错 → 返修后算对了，但与第 6 题重复 → 应继续返修，而不是判定没有进展。"""
    dup_s4 = q("S4", "解方程：3x + 5 = 2x - 1", "x = -6")           # 和 S3 一模一样
    fixed_s4 = q("S4", "解方程：2(x - 1) = x + 3", "x = 5")
    llm = ScriptedLLM([
        REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": [Q1, Q2, GOOD_S3, WRONG_S4]}),
        blind_reply(*BLIND_ALL, ("S4", ["60t + 80t = 240"], ["t"], "")), j({"issues": []}),   # 第 0 轮：算错
        j({"questions": [dup_s4]}),
        blind_reply(("S4", ["3x + 5 = 2x - 1"], ["x"], "")),                                     # 第 1 轮：算对了
        j({"issues": [{"slot_id": "S4", "severity": "must_fix", "category": "重复", "message": "与第3题相同"}]}),
        j({"questions": [fixed_s4]}),                                                              # 第 2 轮返修
        blind_reply(("S4", ["2(x - 1) = x + 3"], ["x"], "")), j({"issues": []}),
    ])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分")
    assert res.review.stop_reason == "passed" and res.review.rounds == 2
    assert next(x for x in res.paper.questions if x.slot_id == "S4").answer == "x = 5"
