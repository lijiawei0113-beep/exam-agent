"""阶段二：对话式局部修改。

基准试卷（来自 fixtures）：
  第1题 S1 选择题 5分 | 第2题 S2 填空题 5分 | 第3题 S3 计算题 10分 | 第4题 S4 计算题 10分，满分 30
"""
import json

import pytest

from exam_agent.agents.base import AgentFailure
from exam_agent.agents.editor import EditPlan
from exam_agent.editing import apply_edit, apply_ops, undo
from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.store import PaperStore

from fixtures import ALL_QUESTIONS, GOOD_BLUEPRINT, REQUEST_REPLY


def j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def make_store_with_paper():
    store = PaperStore(":memory:")
    res = create_paper(ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS]), store, "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    return store, res.paper


def plan(*ops, action="edit", reply="好的"):
    return EditPlan.model_validate({"action": action, "ops": list(ops), "reply": reply})


def content_fps(paper):
    return {q.slot_id: q.content_fingerprint() for q in paper.questions}


# ---------- 纯函数：apply_ops ----------

def test_rewrite_only_touches_target_slot():
    _, paper = make_store_with_paper()
    a = apply_ops(paper, plan({"op": "rewrite", "positions": [3], "difficulty": "基础"}))
    assert a.problems == []
    assert a.regenerate == {"S3"}
    assert a.blueprint.slot("S3").difficulty.value == "基础"
    assert "原题为" in a.notes["S3"]          # 命题员会拿到原题，避免换汤不换药


def test_invalid_position_is_reported():
    _, paper = make_store_with_paper()
    a = apply_ops(paper, plan({"op": "rewrite", "positions": [9]}))
    assert a.problems and "不存在的第 9 题" in a.problems[0]


def test_retype_keeps_count_and_total_and_moves_to_type_group():
    _, paper = make_store_with_paper()
    # 第3题（S3 计算题）改成应用题：题数仍为 4，满分仍为 30
    a = apply_ops(paper, plan({"op": "retype", "positions": [3], "new_qtype": "应用题",
                               "knowledge_point": "行程问题"}))
    assert a.problems == []
    bp = a.blueprint
    assert len(bp.slots) == 4 and sum(s.score for s in bp.slots) == 30
    assert [(s.slot_id, s.qtype.value) for s in bp.slots] == [
        ("S1", "选择题"), ("S2", "填空题"), ("S4", "计算题"), ("S3", "应用题")]   # 应用题排到最后
    assert bp.slot("S3").position == 4 and bp.slot("S4").position == 3


def test_removing_last_of_required_type_is_rejected():
    _, paper = make_store_with_paper()
    # 老师要求过“选择题”，删掉唯一的选择题会违反需求
    a = apply_ops(paper, plan({"op": "remove", "positions": [1]}))
    assert any("缺少要求的题型：选择题" in p for p in a.problems)


# ---------- 端到端：apply_edit ----------

def test_scenario_make_q3_easier_others_untouched():
    store, paper = make_store_with_paper()
    before = content_fps(paper)
    llm = ScriptedLLM([
        j({"action": "edit", "ops": [{"op": "rewrite", "positions": [3], "difficulty": "基础",
                                      "instruction": "换成一步移项的基础题"}],
           "reply": "好的，第 3 题换成基础题，其他题不动。"}),
        j({"questions": [{"slot_id": "S3", "stem": "解方程：x + 2 = 5", "answer": "x = 3",
                          "analysis": "移项得 x = 3"}]}),
    ])
    res = apply_edit(llm, store, paper.paper_id, "第 3 题太难，换一道基础题，其他题别动", review=False)

    assert res.version == 2
    after = content_fps(res.paper)
    assert {sid for sid in before if before[sid] != after[sid]} == {"S3"}
    assert [c.kind for c in res.changes] == ["content"]
    # 命题员只被要求出 S3，其他题只以“避免重复”的摘要出现
    writer_prompt = llm.calls[1][-1]["content"]
    slot_section = writer_prompt.split("需要出题的 slot")[1].split("针对个别")[0]
    assert '"S3"' in slot_section and '"S1"' not in slot_section


def test_writer_cannot_sneak_in_changes_to_other_questions():
    store, paper = make_store_with_paper()
    llm = ScriptedLLM([
        j({"action": "edit", "ops": [{"op": "rewrite", "positions": [3]}], "reply": "好"}),
        # 命题员“顺手”把 S1 也改了 —— 这条必须被丢弃
        j({"questions": [
            {"slot_id": "S3", "stem": "解方程：x + 2 = 5", "answer": "x = 3", "analysis": "略"},
            {"slot_id": "S1", "stem": "被偷偷改掉的选择题", "options": ["a", "b", "c", "d"],
             "answer": "A", "analysis": "略"}]}),
    ])
    res = apply_edit(llm, store, paper.paper_id, "换掉第3题", review=False)
    s1 = next(q for q in res.paper.questions if q.slot_id == "S1")
    assert s1.stem != "被偷偷改掉的选择题"


def test_scenario_add_application_keep_count_and_total():
    store, paper = make_store_with_paper()
    llm = ScriptedLLM([
        j({"action": "edit", "ops": [{"op": "retype", "positions": [4], "new_qtype": "应用题",
                                      "knowledge_point": "行程问题", "instruction": "相遇问题"}],
           "reply": "保持 4 题 30 分：把第 4 题（与第 3 题考法重复）改为应用题。"}),
        j({"questions": [{"slot_id": "S4", "stem": "甲乙两地相距 300 千米……几小时相遇？",
                          "answer": "3 小时", "analysis": "设 x 小时，60x + 40x = 300"}]}),
    ])
    res = apply_edit(llm, store, paper.paper_id, "增加一道应用题，仍然保持 4 题、30 分", review=False)
    p = res.paper
    assert len(p.questions) == 4 and p.total_score == 30
    assert {q.qtype.value for q in p.questions} == {"选择题", "填空题", "计算题", "应用题"}


def test_planner_gets_feedback_on_bad_plan():
    store, paper = make_store_with_paper()
    llm = ScriptedLLM([
        j({"action": "edit", "ops": [{"op": "rewrite", "positions": [7]}], "reply": "好"}),
        j({"action": "edit", "ops": [{"op": "rewrite", "positions": [3]}], "reply": "好"}),
        j({"questions": [{"slot_id": "S3", "stem": "解方程：x - 1 = 2", "answer": "x = 3", "analysis": "略"}]}),
    ])
    res = apply_edit(llm, store, paper.paper_id, "改一下最后那道", review=False)
    assert res.version == 2
    assert "不存在的第 7 题" in llm.calls[1][-1]["content"]


def test_answer_action_does_not_create_version():
    store, paper = make_store_with_paper()
    llm = ScriptedLLM([j({"action": "answer", "ops": [], "reply": "第 3 题考查移项。"})])
    res = apply_edit(llm, store, paper.paper_id, "第三题考什么？", review=False)
    assert res.action == "answer" and res.version == 1 and store.current_version(paper.paper_id) == 1


def test_stale_base_version_is_rejected():
    from exam_agent.store import VersionConflict
    store, paper = make_store_with_paper()
    edited = paper.model_copy(deep=True)
    store.commit_version(edited, expected_version=1, operation="edit", summary="另一个窗口的修改")
    with pytest.raises(VersionConflict):
        store.commit_version(edited, expected_version=1, operation="edit", summary="基于旧版本的修改")


def test_undo_twice_walks_back_through_history():
    store, paper = make_store_with_paper()
    for stem in ("解方程：x + 1 = 2", "解方程：x + 3 = 4"):
        apply_edit(ScriptedLLM([
            j({"action": "edit", "ops": [{"op": "rewrite", "positions": [3]}], "reply": "好"}),
            j({"questions": [{"slot_id": "S3", "stem": stem, "answer": "x = 1", "analysis": "略"}]}),
        ]), store, paper.paper_id, "换第3题", review=False)
    # v1 原题 → v2 → v3；撤销两次应回到 v1 的内容
    _, target1, _ = undo(store, paper.paper_id)
    snap2, target2, v = undo(store, paper.paper_id)
    assert (target1, target2, v) == (2, 1, 5)
    assert content_fps(snap2) == content_fps(paper)
    with pytest.raises(ValueError):
        undo(store, paper.paper_id)


def test_constraint_without_review_warns_honestly():
    store, paper = make_store_with_paper()
    llm = ScriptedLLM([j({"action": "edit", "ops": [{"op": "add_constraint", "constraint": {
        "kind": "integer_answer", "text": "计算题答案为整数", "scope_types": ["计算题"]}}], "reply": "好"})])
    res = apply_edit(llm, store, paper.paper_id, "计算题答案都要是整数", review=False)
    assert res.paper.request.constraints[0].text == "计算题答案为整数"
    assert res.warnings and "未开启审核" in res.warnings[0]
