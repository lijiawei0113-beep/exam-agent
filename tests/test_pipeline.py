import json

from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.store import PaperStore

from fixtures import ALL_QUESTIONS, BAD_BLUEPRINT, GOOD_BLUEPRINT, PARTLY_BAD_QUESTIONS, Q1, Q4, REQUEST_REPLY


def test_happy_path():
    llm = ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS])
    res = create_paper(llm, PaperStore(":memory:"), "出一份一元一次方程小测，4 道题，满分 30", review=False)
    assert res.paper.total_score == 30
    assert len(res.paper.questions) == 4
    assert res.version == 1
    assert res.totals["llm_calls"] == 3
    # 分值来自蓝图而不是 LLM
    assert {q.slot_id: q.score for q in res.paper.questions} == {"S1": 5, "S2": 5, "S3": 10, "S4": 10}


def test_planner_scores_are_fixed_by_code_without_retry():
    # BAD_BLUEPRINT 的分值之和是 28：以前要让模型重做，现在由求解器直接配成 30 分
    llm = ScriptedLLM([REQUEST_REPLY, BAD_BLUEPRINT, ALL_QUESTIONS])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    assert res.paper.total_score == 30 and res.totals["llm_calls"] == 3
    by_type = {}
    for s in res.paper.blueprint.slots:
        by_type.setdefault(s.qtype.value, set()).add(s.score)
    assert all(len(v) == 1 for v in by_type.values())       # 同类同分


def test_planner_gets_feedback_when_count_is_wrong():
    three = json.loads(GOOD_BLUEPRINT)
    three["slots"] = three["slots"][:3]
    llm = ScriptedLLM([REQUEST_REPLY, json.dumps(three, ensure_ascii=False), GOOD_BLUEPRINT, ALL_QUESTIONS])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    assert len(res.paper.questions) == 4
    assert "题目数量为 3" in llm.calls[2][-1]["content"]


def test_planner_prompt_lists_feasible_type_counts():
    llm = ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS])
    create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    assert "能够凑成满分的题型题数配比" in llm.calls[1][-1]["content"]


def test_writer_only_regenerates_failed_slots():
    # 第一轮：S2、S3 合格；S1 只有 3 个选项；S4 被遗漏
    fixed = json.dumps({"questions": [Q1, Q4]}, ensure_ascii=False)
    llm = ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, PARTLY_BAD_QUESTIONS, fixed])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    assert len(res.paper.questions) == 4 and llm.replies == []

    retry_prompt = llm.calls[3][-1]["content"]
    slot_section = retry_prompt.split("需要出题的 slot")[1].split("已有题目")[0]
    assert '"S1"' in slot_section and '"S4"' in slot_section
    assert '"S2"' not in slot_section and '"S3"' not in slot_section   # 合格的题不重出
    assert "4 个选项" in retry_prompt and "遗漏" in retry_prompt         # 带着具体反馈


def test_trace_is_recorded():
    store = PaperStore(":memory:")
    res = create_paper(ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS]), store, "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    steps = [e["step"] for e in store.events(paper_id=res.paper.paper_id)]
    assert steps == ["input", "requirement", "planner", "planner_scores", "writer", "writer_check", "final_check", "committed"]
