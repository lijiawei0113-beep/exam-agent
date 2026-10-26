"""端到端评测框架本身的测试（不调用真实模型）。"""
import csv
import json

from exam_agent.experiments import e2e_eval
from exam_agent.experiments.e2e_eval import check_edit, load_requests, plan_edits, run_e2e, summarize
from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.store import PaperStore

from fixtures import ALL_QUESTIONS, GOOD_BLUEPRINT, REQUEST_REPLY

TEXT = "七年级一元一次方程小测，4 道题，满分 30 分"


def test_dataset_has_100_varied_requests():
    reqs = load_requests()
    assert len(reqs) == 100 and len({r["id"] for r in reqs}) == 100
    groups = {g: sum(r["group"] == g for r in reqs) for g in ("数学", "英语", "边界")}
    assert groups == {"数学": 60, "英语": 25, "边界": 15}
    assert sum(r["expect"] == "reject" for r in reqs) == 6
    assert len({r["text"] for r in reqs}) >= 90           # 说法基本不重复


def test_plan_edits_is_deterministic_and_valid():
    store = PaperStore(":memory:")
    paper = create_paper(ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS]), store, TEXT, review=False).paper
    for rid in ("r001", "r002", "r003", "r004", "r005"):
        a, b = plan_edits(paper, rid), plan_edits(paper, rid)
        assert a == b and a[0][0] == "rewrite" and len(a) == 2
    kinds = {plan_edits(paper, f"r{i:03d}")[1][0] for i in range(1, 11)}
    assert {"restore", "resize", "ask", "constraint"} <= kinds


def test_check_edit_detects_wrong_results():
    from exam_agent.editing import EditResult

    store = PaperStore(":memory:")
    paper = create_paper(ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS]), store, TEXT, review=False).paper
    same = EditResult(action="edit", reply="好", paper=paper, version=2)
    ok, why = check_edit("rewrite", {"k": 3}, paper, same, store, paper.paper_id)
    assert not ok and "没有被替换" in why
    ok, why = check_edit("resize", {"n": 9}, paper, same, store, paper.paper_id)
    assert not ok and "题数应为 9" in why
    ok, _ = check_edit("ask", {"j": 1}, paper, EditResult(action="answer", reply="因为", paper=paper, version=1),
                       store, paper.paper_id)
    assert ok
    ok, why = check_edit("ask", {"j": 1}, paper, same, store, paper.paper_id)
    assert not ok and "只回答" in why


def test_run_e2e_writes_results_report_and_sample(tmp_path=None):
    import tempfile
    from pathlib import Path

    out = Path(tempfile.mkdtemp())
    reqs = [{"id": "r001", "group": "数学", "grade": 7, "text": TEXT, "expect": "paper"},
            {"id": "r002", "group": "边界", "grade": 0, "text": "出一份七年级语文试卷", "expect": "reject"}]
    old = e2e_eval.load_requests
    e2e_eval.load_requests = lambda path=None: reqs
    chinese = json.dumps({"request": {"stage": "初中", "grade": "七年级", "subject": "语文", "topic": "",
                                      "question_count": 20, "total_score": 100, "required_types": []},
                          "assumptions": []}, ensure_ascii=False)
    blind = json.dumps({"solutions": [{"slot_id": "S1", "choice": "B"}, {"slot_id": "S2", "final_answer": "3"},
                                      {"slot_id": "S3"}, {"slot_id": "S4"}]})
    scripts = {"0": [REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS, blind, json.dumps({"issues": []})]}

    def factory():
        return ScriptedLLM(list(scripts["0"]) + [chinese])

    try:
        s, path = run_e2e(factory, workers=1, resume=out, do_edits=False, sample=3)
        s2, _ = run_e2e(lambda: (_ for _ in ()).throw(AssertionError("不应再调用")), workers=1, resume=out,
                        do_edits=False, sample=0)
    finally:
        e2e_eval.load_requests = old
    rows = [json.loads(x) for x in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["id"]: r["outcome"] for r in rows} == {"r001": "paper", "r002": "reject"}
    assert s["create_ok"] == 2 and s["questions"] == 4
    assert "应当拒绝的请求，正确拒绝 | 100%" in (path / "report.md").read_text(encoding="utf-8")
    with open(path / "audit_sample.csv", encoding="utf-8-sig") as f:
        assert len(list(csv.reader(f))) == 4
    assert s2["create_ok"] == 2                          # 续跑：已完成的条目不重跑，工厂函数不会被调用
    assert len((path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_patch_numbers_skip_subitem_labels():
    """试跑 r003：从“（7）5 - 4 = ____”里选中了小题编号 7，模型合理地追问——这是评测指令设计的问题。"""
    from exam_agent.experiments.e2e_eval import _numbers_once

    assert 7 not in _numbers_once("口算。\n（6）4 - 3 = ____\n（7）5 - 4 = ____")
    assert _numbers_once("小明家到学校 650 米，往返一共多少米？") == [650]
