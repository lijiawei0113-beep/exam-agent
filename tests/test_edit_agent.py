"""ReAct 修改与讨论 Agent：按内容定位、先读后答、恢复旧版、局部改字、停止条件与上下文压缩。

基准试卷（来自 fixtures）：第1题选择 | 第2题填空“方程 2x = 6 的解是 x = ____” | 第3、4题解方程
"""
import json

from exam_agent.agents import edit_agent
from exam_agent.agents.edit_agent import EditWorkspace, run_edit_agent, tool_check_question, tool_paper_stats
from exam_agent.editing import apply_edit
from exam_agent.llm import ScriptedLLM, tool_reply
from exam_agent.pipeline import create_paper
from exam_agent.store import PaperStore

from fixtures import ALL_QUESTIONS, GOOD_BLUEPRINT, REQUEST_REPLY, q

j = lambda o: json.dumps(o, ensure_ascii=False)
TEXT = "七年级一元一次方程小测，4 道题，满分 30 分"


def setup():
    store = PaperStore(":memory:")
    res = create_paper(ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS]), store, TEXT, review=False)
    return store, res.paper


def submit(action="edit", reply="好的", **op):
    return tool_reply(("submit_plan", {"action": action, "ops": [op] if op else [], "reply": reply}))


def tool_msgs(call):
    return [m["content"] for m in call if m["role"] == "tool"]


def test_locates_question_by_content_then_edits():
    store, paper = setup()
    llm = ScriptedLLM([tool_reply(("search_questions", {"query": "2x = 6"})),
                       tool_reply(("get_question", {"position": 2})),
                       submit(op="rewrite", positions=[2], instruction="换一道系数不同的填空题"),
                       j({"questions": [q("S2", "方程 3x = 9 的解是 x = ____", "3")]})])
    res = apply_edit(llm, store, paper.paper_id, "把考 2x=6 的那道题换一下", review=False)
    assert "第2题" in tool_msgs(llm.calls[1])[0] and "【原文包含】" in tool_msgs(llm.calls[1])[0]
    assert "解析：" in tool_msgs(llm.calls[2])[-1]                    # 读到了全文
    assert res.action == "edit" and [c.slot_id for c in res.changes] == ["S2"]
    assert "题目索引（全文用 get_question 读取）" in llm.calls[0][1]["content"]


def test_answers_question_after_reading_full_text_without_new_version():
    store, paper = setup()
    llm = ScriptedLLM([tool_reply(("get_question", {"position": 1})),
                       submit(action="answer", reply="第 1 题选 B，因为 2x-3=5 只含一个未知数且次数为 1，其他选项不是一元一次方程。")])
    res = apply_edit(llm, store, paper.paper_id, "第一题为什么选 B？", review=False)
    assert res.action == "answer" and "选 B" in res.reply and res.version == 1
    assert "选项：" in tool_msgs(llm.calls[1])[0]
    assert store.current_version(paper.paper_id) == 1


def test_text_reply_is_classified_through_submit():
    """e1：模型直接用文字向老师追问，曾被当成 answer；现在要求它再用 submit_plan 归类。"""
    store, paper = setup()
    llm = ScriptedLLM(["您说的“那道题”是第几题？", submit(action="clarify", reply="您说的“那道题”是第几题？")])
    plan = run_edit_agent(llm, EditWorkspace(store, paper.paper_id, paper, 1), "把那道题改一下")
    assert plan.action == "clarify" and "submit_plan" in llm.calls[1][-1]["content"]


def test_tool_call_budget_forces_submit():
    """e1：统计类问题曾逐题查阅 24 次。总调用次数到上限就强制提交。"""
    store, paper = setup()
    many = tool_reply(*[("get_question", {"position": i}) for i in (1, 2, 3, 4)])
    more = tool_reply(*[("get_history", {"position": i}) for i in (1, 2, 3, 4)])
    more2 = tool_reply(("paper_stats", {}), ("search_questions", {"query": "x"}))
    llm = ScriptedLLM([many, more, more2, submit(action="answer", reply="都看过了")])
    plan = run_edit_agent(llm, EditWorkspace(store, paper.paper_id, paper, 1), "逐题检查")
    assert plan.action == "answer" and "查阅上限" in llm.calls[3][-1]["content"]


def test_option_patch_tolerates_letter_prefix():
    store, paper = setup()
    patches = [{"field": "options", "old": "B. 2x-3=5", "new": "B. 2x-3=7"}]
    llm = ScriptedLLM([submit(op="patch", positions=[1], patches=patches)])
    res = apply_edit(llm, store, paper.paper_id, "第1题选项B改成 2x-3=7", review=False)
    assert res.paper.question_by_position(1).options[1] == "2x-3=7"


def test_restore_question_to_earlier_version():
    store, paper = setup()
    original = paper.question_by_position(3)
    apply_edit(ScriptedLLM([submit(op="rewrite", positions=[3], instruction="换简单些"),
                            j({"questions": [q("S3", "解方程：x + 1 = 2", "x = 1")]})]),
               store, paper.paper_id, "第3题改简单", review=False)
    llm = ScriptedLLM([tool_reply(("get_history", {"position": 3})),
                       submit(op="restore", positions=[3], from_version=1, reply="已把第 3 题恢复为 v1 的题目")])
    res = apply_edit(llm, store, paper.paper_id, "还是把第三题改回原来的吧", review=False)
    hist = tool_msgs(llm.calls[1])[0]
    assert "v1" in hist and "v2" in hist and "3x + 5" in hist and "x + 1 = 2" in hist
    now = res.paper.question_by_position(3)
    assert now.content_fingerprint() == original.content_fingerprint() and res.version == 3
    assert [c.slot_id for c in res.changes] == ["S3"]


def test_restore_problems_are_fed_back():
    store, paper = setup()
    llm = ScriptedLLM([submit(op="restore", positions=[3], from_version=1),       # 内容与当前相同
                       submit(op="restore", positions=[3], from_version=9),       # 版本不存在
                       submit(action="clarify", reply="第 3 题还没有改过，没有可以恢复的旧版本")])
    res = apply_edit(llm, store, paper.paper_id, "第3题改回原来的", review=False)
    assert "不需要恢复" in tool_msgs(llm.calls[1])[-1] and "不存在第 9 版" in tool_msgs(llm.calls[2])[-1]
    assert res.action == "clarify" and res.version == 1


def test_patch_changes_only_the_given_text():
    store, paper = setup()
    patches = [{"field": "stem", "old": "2x = 6", "new": "2x = 8"}, {"field": "answer", "old": "3", "new": "4"}]
    llm = ScriptedLLM([tool_reply(("get_question", {"position": 2})), submit(op="patch", positions=[2], patches=patches)])
    res = apply_edit(llm, store, paper.paper_id, "第2题改成 2x=8", review=False)
    q2 = res.paper.question_by_position(2)
    assert q2.stem == "方程 2x = 8 的解是 x = ____" and q2.answer == "4"
    assert any("未经核验" in w for w in res.warnings)
    assert [c.slot_id for c in res.changes] == ["S2"] and len(llm.calls) == 2      # 没有调用命题员


def test_patch_with_ambiguous_text_is_rejected():
    store, paper = setup()
    bad = [{"field": "stem", "old": "x", "new": "y"}]                                # 题干中 x 出现两次
    llm = ScriptedLLM([submit(op="patch", positions=[2], patches=bad),
                       submit(op="patch", positions=[2], patches=[{"field": "stem", "old": "2x = 6", "new": "2y = 6"}])])
    res = apply_edit(llm, store, paper.paper_id, "把第2题的 x 改成 y", review=False)
    assert "必须恰好 1 次" in tool_msgs(llm.calls[1])[-1]
    assert "2y = 6" in res.paper.question_by_position(2).stem


def test_patched_question_keeps_teacher_stem_and_only_fixes_answer():
    """改了题干数字却没改答案：工具核对发现答案不对；返修只修正答案和解析，老师改的题干保持不变
    （e2e 第二轮：原来会整题重出，老师改的数字就没了）。"""
    store, paper = setup()
    fix = j({"answer": "x = -7", "analysis": "移项得 x = -7",
             "check": {"equations": ["3x + 6 = 2x - 1"], "unknowns": ["x"], "target": "x"}})
    llm = ScriptedLLM([submit(op="patch", positions=[3], patches=[{"field": "stem", "old": "+ 5", "new": "+ 6"}]),
                       j({"solutions": [{"slot_id": "S3"}]}),                                 # 盲解
                       j({"issues": []}),                                                     # 教学审核
                       fix,                                                                   # 只修答案
                       j({"solutions": [{"slot_id": "S3"}]}), j({"issues": []})])
    res = apply_edit(llm, store, paper.paper_id, "第3题的 5 改成 6", review=True)
    q3 = res.paper.question_by_position(3)
    assert q3.stem == "解方程：3x + 6 = 2x - 1" and q3.answer == "x = -7"
    assert res.review.initial and res.review.initial[0].code == "wrong_answer"
    assert {c.slot_id for c in res.changes} == {"S3"}


def test_repeated_queries_force_a_decision():
    store, paper = setup()
    same = tool_reply(("get_question", {"position": 1}))
    llm = ScriptedLLM([same, same, same, submit(action="answer", reply="第 1 题考一元一次方程的概念")])
    plan = run_edit_agent(llm, EditWorkspace(store, paper.paper_id, paper, 1), "第1题考什么")
    assert plan.action == "answer" and len(llm.calls) == 4
    assert "已经做过" in tool_msgs(llm.calls[2])[-1]
    assert llm.calls[3][-1]["content"].startswith("你在重复同样的查询")


def test_step_limit_forces_submit():
    old, edit_agent.MAX_STEPS = edit_agent.MAX_STEPS, 3
    store, paper = setup()
    steps = [tool_reply(("get_question", {"position": i})) for i in (1, 2, 3)]
    llm = ScriptedLLM(steps + [submit(action="clarify", reply="请问具体想改哪一题？")])
    try:
        plan = run_edit_agent(llm, EditWorkspace(store, paper.paper_id, paper, 1), "看看哪里需要改")
    finally:
        edit_agent.MAX_STEPS = old
    assert plan.action == "clarify" and len(llm.calls) == 4
    assert "查阅上限" in llm.calls[3][-1]["content"]


def test_older_tool_results_are_compressed():
    store, paper = setup()
    llm = ScriptedLLM([tool_reply(("get_question", {"position": i})) for i in (1, 2, 3, 4)]
                      + [submit(action="answer", reply="四道题都看过了")])
    run_edit_agent(llm, EditWorkspace(store, paper.paper_id, paper, 1), "逐题看一遍")
    last = tool_msgs(llm.calls[4])
    assert last[0].startswith("（较早的工具结果已压缩）") and all("解析：" in t for t in last[1:])


def test_stats_and_program_check():
    store, paper = setup()
    ws = EditWorkspace(store, paper.paper_id, paper, 1)
    stats = tool_paper_stats(ws)
    assert "选择题答案：1:B" in stats and "共 4 题，满分 30" in stats
    assert "工具验证" in tool_check_question(ws, 3)
    bad = paper.model_copy(deep=True)
    bad.questions[2].answer = "x = 6"
    assert "问题" in tool_check_question(EditWorkspace(store, paper.paper_id, bad, 1), 3)


def test_lookup_scope_without_kb_is_honest():
    store, paper = setup()
    out = edit_agent.tool_lookup_scope(EditWorkspace(store, paper.paper_id, paper, 1), "一元一次方程")
    assert "没有加载" in out


def test_oneshot_planner_still_available():
    store, paper = setup()
    plain = j({"action": "answer", "ops": [], "reply": "这份卷共 4 题"})
    llm = ScriptedLLM([plain])
    res = apply_edit(llm, store, paper.paper_id, "一共几题", review=False, planner="oneshot")
    assert res.reply == "这份卷共 4 题" and "当前各题" in llm.calls[0][-1]["content"]


# ---------- 修改 / 讨论评测的数据与判分 ----------

def test_edit_eval_dataset_is_consistent():
    from exam_agent.experiments.edit_eval import TAG_LABEL, load_ecases, make_store

    cases = load_ecases()
    assert len(cases) >= 30 and all(c.tag in TAG_LABEL for c in cases)
    assert {c.split for c in cases} == {"dev", "holdout"} and len({c.id for c in cases}) == len(cases)
    for pid in {c.paper for c in cases}:
        store = make_store(pid)
        paper, v = store.load(pid)
        n = len(paper.blueprint.slots)
        assert paper.total_score == paper.request.total_score
        for c in cases:
            if c.paper == pid:
                assert all(1 <= p <= n for p in c.positions) and (c.from_version or 1) <= v
    hist = make_store("p_evalmath").history("p_evalmath")
    assert [h["version"] for h in hist] == [1, 2, 3]


def test_edit_eval_judge_and_run_with_scripted_model():
    from exam_agent.experiments.edit_eval import load_ecases, run_one

    case = next(c for c in load_ecases() if c.id == "m09")
    llm = ScriptedLLM([tool_reply(("get_history", {"position": 3})),
                       submit(op="restore", positions=[3], from_version=1)])
    r = run_one(llm, case, "react")
    assert r.ok and r.tool_calls == 1 and r.llm_calls == 2
    wrong = run_one(ScriptedLLM([submit(op="rewrite", positions=[3])]), case, "react")
    assert not wrong.ok and "restore" in wrong.reason
    # m10：第 6 题在 v1、v2 内容相同，恢复到 v1 与恢复到 v2 等价
    m10 = next(c for c in load_ecases() if c.id == "m10")
    same = run_one(ScriptedLLM([submit(op="restore", positions=[6], from_version=1)]), m10, "react")
    assert same.ok


# ---------- d042：扩大试卷由程序算新增数量；回复与动作一致；回复附程序执行结果 ----------

def _five_calc_paper():
    from exam_agent.domain import Blueprint, BlueprintSlot, ExamRequest, PaperSnapshot, QType, Question

    req = ExamRequest(stage="初中", grade="七年级", subject="数学", term="全年", topic="", question_count=5,
                      total_score=100, required_types=[QType.CALC])
    slots = [BlueprintSlot(slot_id=f"S{i}", position=i, qtype=QType.CALC, score=20, knowledge_point="k",
                           difficulty="基础", cognitive_level="掌握", intent="x") for i in range(1, 6)]
    qs = [Question(slot_id=f"S{i}", qtype=QType.CALC, score=20, knowledge_point="k", difficulty="基础",
                   stem=f"计算：{i} + {i} =", answer=str(2 * i), analysis="略") for i in range(1, 6)]
    store = PaperStore(":memory:")
    paper = PaperSnapshot(paper_id="p_five", title="t", request=req, blueprint=Blueprint(slots=slots), questions=qs)
    store.create_paper(paper, "新建")
    return store, paper


def test_resize_adds_exact_number_and_keeps_old_questions():
    from exam_agent.agents.editor import EditPlan
    from exam_agent.editing import apply_ops

    _, paper = _five_calc_paper()
    out = apply_ops(paper, EditPlan.model_validate({"action": "edit", "reply": "好", "ops": [
        {"op": "resize", "count": 20}]}))
    assert out.problems == [] and len(out.blueprint.slots) == 20 and len(out.regenerate) == 15
    assert out.request.question_count == 20 and sum(s.score for s in out.blueprint.slots) == 100
    kinds = {s.qtype.value for s in out.blueprint.slots}
    assert {"选择题", "填空题", "计算题"} <= kinds
    shrink = apply_ops(paper, EditPlan.model_validate({"action": "edit", "reply": "好", "ops": [
        {"op": "resize", "count": 3}]}))
    assert "只用于扩大" in shrink.problems[0]


def test_add_with_count():
    from exam_agent.agents.editor import EditPlan
    from exam_agent.editing import apply_ops

    store, paper = setup()
    out = apply_ops(paper, EditPlan.model_validate({"action": "edit", "reply": "好", "ops": [
        {"op": "add", "new_qtype": "填空题", "knowledge_point": "移项", "count": 2}]}))
    assert out.problems == [] and len(out.blueprint.slots) == 6 and len(out.regenerate) == 2


def test_edit_reply_must_not_ask_for_confirmation_and_gets_program_summary():
    store, paper = setup()
    asking = submit(op="rewrite", positions=[3], reply="我打算重出第 3 题，请回复“照此执行”")
    llm = ScriptedLLM([asking, submit(op="rewrite", positions=[3], reply="好的，将重出第 3 题"),
                       j({"questions": [q("S3", "解方程：x + 2 = 5", "x = 3")]})])
    res = apply_edit(llm, store, paper.paper_id, "第3题换一道", review=False)
    assert "征求老师同意" in tool_msgs(llm.calls[1])[-1]
    assert res.reply.startswith("好的，将重出第 3 题") and "程序执行结果：现在共 4 题，满分 30" in res.reply


def test_answer_that_claims_constraint_recorded_is_sent_back():
    store, paper = setup()
    claim = submit(action="answer", reply="已把您的要求记为整卷持续约束，以后都会遵守")
    real = submit(op="add_constraint", reply="好的，将把“计算题答案为整数”记为持续约束并核验现有题",
                  constraint={"kind": "integer_answer", "text": "答案为整数", "scope_types": ["计算题"]})
    llm = ScriptedLLM([claim, real])
    res = apply_edit(llm, store, paper.paper_id, "计算题答案都要是整数", review=False)
    assert "不会执行任何修改" in tool_msgs(llm.calls[1])[-1]
    assert res.action == "edit" and res.paper.request.constraints[0].text == "答案为整数"
