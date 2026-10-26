"""阶段四：知识库构建流水线 + 检索接入出卷流程。"""
import csv
import json

from exam_agent import kbuild
from exam_agent.agents.planner import plan_blueprint
from exam_agent.domain import ExamRequest, VStatus
from exam_agent.grounding import Grounding, parse_unit
from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.retrieval.kb import KnowledgeBase, Scope, load_kb
from exam_agent.store import PaperStore

j = lambda o: json.dumps(o, ensure_ascii=False)
KB = load_kb()
G = Grounding(KB)   # 只用 BM25，不加载模型


def req(**kw):
    base = dict(stage="小学", grade="三年级", subject="数学", term="全年", topic="", question_count=4,
                total_score=30, required_types=["选择题", "填空题", "计算题"])
    base.update(kw)
    return ExamRequest(**base)


# ---------- 范围 ----------

def test_scope_follows_teacher_words():
    c = G.candidates(req())
    assert c.scope.books == ["三上", "三下"] and len(c.points) == 17
    c = G.candidates(req(term="上"), "出一份三年级上册第二单元的卷子")
    assert c.scope.unit == 2 and [p.name for p in c.points] == ["万以内的加法和减法（一）"]
    assert parse_unit("第五章") == 5 and parse_unit("第十单元") == 10


def test_topic_is_located_by_retrieval():
    c = G.candidates(req(stage="初中", grade="七年级", topic="一元一次方程"))
    assert c.scope.books == ["七上"] and c.scope.unit == 5 and "检索" in c.how
    assert all(p.unit == 5 for p in c.points)
    assert "方程组" in c.signals          # 七上第五章的卷子里出现“方程组”就是超纲


def test_uncovered_subject_returns_none():
    assert G.candidates(req(subject="英语")) is None
    assert G.candidates(req(grade="八年级")) is None   # 八年级只有边界卡片，没有单元


# ---------- 规划只能选候选知识点 ----------

def slot(i, qtype, kp_id, score=5):
    return {"slot_id": f"S{i}", "position": i, "qtype": qtype, "score": score, "knowledge_point": "x",
            "difficulty": "基础", "cognitive_level": "掌握", "intent": "x", "kp_id": kp_id}


BP_OK = {"slots": [slot(1, "选择题", "M3A-01"), slot(2, "填空题", "M3A-05"), slot(3, "计算题", "M3A-06", 10),
                   slot(4, "计算题", "M3B-02", 10)], "rationale": ""}


def test_planner_must_pick_candidate_ids():
    bad = {"slots": [slot(1, "选择题", "M2B-07"), *BP_OK["slots"][1:]], "rationale": ""}   # 二年级的单元
    llm = ScriptedLLM([j(bad), j(BP_OK)])
    bp = plan_blueprint(llm, req(), cands=G.candidates(req()))
    assert "M2B-07" in llm.calls[1][-1]["content"]                 # 代码把范围外的 id 打回
    s1 = next(s for s in bp.slots if s.kp_id == "M3A-01")
    assert "时、分、秒" in s1.scope_note and "三年级" in bp.scope
    assert "[M3A-01]" in llm.calls[0][-1]["content"]               # 候选列表交给了模型


# ---------- 端到端：超纲信号词由代码直接发现 ----------

def test_out_of_scope_signal_triggers_repair():
    request = {"request": req(stage="初中", grade="七年级", topic="一元一次方程").model_dump(mode="json"),
               "assumptions": []}
    bp = {"slots": [slot(1, "选择题", "M7A-05-02"), slot(2, "填空题", "M7A-05-03"),
                    slot(3, "计算题", "M7A-05-05", 10), slot(4, "计算题", "M7A-05-06", 10)], "rationale": ""}
    q = lambda s, stem, a, o=None: {"slot_id": s, "stem": stem, "options": o or [], "answer": a, "analysis": "略"}
    qs = [q("S1", "下列是一元一次方程的是（ ）", "B", ["x+y=1", "2x-3=5", "x²=4", "1/x=2"]),
          q("S2", "方程 2x = 6 的解是 x = ____", "3"),
          q("S3", "解方程组：x + y = 3，x - y = 1，求 x", "x = 2"),      # 七上第五章出现“方程组”
          q("S4", "解方程：2(x - 1) = x + 3", "x = 5")]
    blind = lambda *ids: j({"solutions": [{"slot_id": i, "choice": "B" if i == "S1" else ""} for i in ids]})
    llm = ScriptedLLM([j(request), j(bp), j({"questions": qs}), blind("S1", "S2", "S3", "S4"), j({"issues": []}),
                       j({"questions": [q("S3", "解方程：3x + 5 = 2x - 1", "x = -6")]}), blind("S3"), j({"issues": []})])
    res = create_paper(llm, PaperStore(":memory:"), "出一份一元一次方程试卷，4 道题，满分 30 分", grounding=G)
    r = res.review
    assert [f.code for f in r.initial if f.slot_id == "S3"] == ["out_of_scope_signal"]
    assert r.initial[0].source == "kb" and "七年级下册" in r.initial[0].message
    assert r.stop_reason == "passed" and r.repaired == {"S3"}
    assert any("检索" in a for a in res.assumptions)
    assert res.paper.blueprint.scope and all(s.scope_note for s in res.paper.blueprint.slots)
    assert next(x for x in res.paper.questions if x.slot_id == "S3").vstatus == VStatus.TOOL


# ---------- 知识库构建流水线 ----------

def draft_units(units, **override):
    return {"units": [{"unit": i, "name": n, "description": "d", "key_points": ["a", "b"], "boundary": "不含小数",
                       "aliases": [], "signals": [], "standard_refs": ["P2-NUM"], **override}
                      for i, n in enumerate(units, start=1)]}


def test_skeleton_ids_and_order():
    cards = kbuild.skeleton_cards("三上", ["时、分、秒", "测量"], "人教版")
    assert [c.id for c in cards] == ["M3A-01", "M3A-02"] and cards[1].order == 30200
    assert cards[0].status == "skeleton" and cards[0].stage == "小学"


def test_build_validation_feeds_back_concrete_problems():
    units = kbuild.load_toc()["books"]["三上"]
    bad = draft_units(units[:-1])                                         # 少了一个单元
    wrong_ref = draft_units(units, standard_refs=["P3-NUM"])              # 五六年级的课标条目
    good = draft_units(units)
    good["units"][0]["signals"] = ["秒针"]
    llm = ScriptedLLM([j(bad), j(wrong_ref), j(good)])
    cards = kbuild.build_book(llm, "三上", KB, model_name="test")
    assert "必须与目录完全一致" in llm.calls[1][-1]["content"]
    assert "P3-NUM" in llm.calls[2][-1]["content"]
    assert cards[0].status == "draft" and cards[0].signals == ["秒针"] and cards[0].key_points == ["a", "b"]


def test_signal_must_be_new():
    kb = KnowledgeBase("math", kbuild.skeleton_cards("二下", ["有余数的除法"], "e")
                       + [], KB.clauses)
    kb.points[0].key_points = ["余数", "试商"]
    d = kbuild.BookDraft.model_validate(draft_units(["除数是一位数的除法"], signals=["余数"]))
    problems = kbuild.validate_book(d, "三下", ["除数是一位数的除法"], kb)
    assert any("之前的年级或本册前面的单元已经出现过" in p for p in problems)


def test_audit_sample_and_report(tmp_path):
    root = tmp_path / "math"
    cards = [c.model_copy(update={"status": "draft", "key_points": ["k"], "boundary": "b"})
             for c in kbuild.skeleton_cards("三上", ["时、分、秒", "测量", "倍的认识"], "e")]
    kbuild.write_cards(cards, root / "points" / "三上.jsonl")
    kb = KnowledgeBase("math", list(cards), [])
    path = kbuild.sample_for_audit(kb, n=3, seed=1, out_dir=root / "audit")
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    rows[1][6], rows[2][6], rows[3][6], rows[3][7] = "对", "对", "错", "边界写错了"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    rep = kbuild.audit_report(path, root=root)
    assert rep["filled"] == 3 and rep["wrong"] == 1 and abs(rep["accuracy"] - 2 / 3) < 1e-9
    saved = [json.loads(l)["status"] for l in (root / "points" / "三上.jsonl").read_text(encoding="utf-8").splitlines()]
    assert saved.count("checked") == 2 and (root / "audit" / "audit_history.csv").exists()


def test_build_rejects_ambiguous_or_repeated_signals():
    """d020：“认识人民币”的信号词“角”与前面单元“认识图形”里的“角”同字不同义。"""
    units = ["认识图形（二）", "认识人民币"]
    d = kbuild.BookDraft.model_validate(draft_units(units))
    d.units[0].key_points = ["认识角", "整十数"]
    d.units[1].signals = ["角", "整十数"]
    problems = kbuild.validate_book(d, "一下", units, KB)
    assert any("只有一个字" in p for p in problems) and any("本册前面的单元" in p for p in problems)
