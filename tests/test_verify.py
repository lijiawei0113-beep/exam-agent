"""验证路由、验证状态、老师确认、出卷范围（年级 / 册别）。"""
import json
from fractions import Fraction

from exam_agent.agents.requirement import _validate, normalize, normalize_grade, parse_request, term_in_text, RequestDraft
from exam_agent.agents.solver import BlindSolution
from exam_agent.domain import ExamRequest, QType, Question, VStatus
from exam_agent.editing import confirm
from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.review import check_question
from exam_agent.store import PaperStore
from exam_agent.tools.arith import evaluate, parse_remainder_answer, remainder_division
from exam_agent.tools.linear import ToolError
from exam_agent.verify import from_stem, route

from fixtures import GOOD_BLUEPRINT, Q1, Q2, REQUEST_REPLY, q

j = lambda o: json.dumps(o, ensure_ascii=False)
REQ = ExamRequest.model_validate(json.loads(REQUEST_REPLY)["request"])


def mk(stem, answer, qtype=QType.CALC, options=None):
    return Question(slot_id="S1", qtype=qtype, score=5, knowledge_point="k", difficulty="基础",
                    stem=stem, answer=answer, options=options or [], analysis="略")


def bl(**kw):
    return BlindSolution(slot_id="S1", **kw)


# ---------- 精确计算工具 ----------

def test_arith_is_exact():
    assert evaluate("0.1+0.2") == Fraction(3, 10)
    assert evaluate("25%×80") == 20
    assert evaluate("（120－30）÷3") == 30
    assert evaluate("125×8=") == 1000
    try:
        evaluate("3x+1")
        assert False
    except ToolError:
        pass


def test_remainder_division():
    assert remainder_division("17÷5") == (3, 2)
    assert remainder_division("17÷5+1") is None
    assert parse_remainder_answer("3……2") == (3, 2)
    assert parse_remainder_answer("商3余2") == (3, 2)
    assert parse_remainder_answer("3.4") is None


# ---------- 从题干直接提取 ----------

def test_from_stem_only_takes_whole_question_formulas():
    assert from_stem("解方程：3(x - 2) = x + 4").kind == "equation"
    assert from_stem("计算：125×8=____").text == ["125×8"]
    assert from_stem("36÷4=（ ）").kind == "expression"
    assert from_stem("计算下面各题：(1)3+4 (2)5×6") is None        # 多小题不提取
    assert from_stem("小明有 12 元，买了 3 支笔，还剩几元？") is None  # 有文字情境交给盲解


# ---------- 路由与状态 ----------

def test_direct_extraction_beats_blind_modeling():
    # 题干就是方程：即使盲解列错了方程，也以题干为准
    r = route(mk("解方程：2x + 1 = 7", "x = 3"), bl(equations=["2x + 1 = 9"], unknowns=["x"], target="x"))
    assert r.verification.status == VStatus.TOOL and "题干直接提取" in r.verification.method


def test_wrong_arithmetic_is_problem_with_evidence():
    r = route(mk("计算：17÷5=", "3……1"))
    assert r.verification.status == VStatus.PROBLEM and "3……2" in r.mismatch
    r = route(mk("计算：17÷5=", "3.4"))                      # 写成小数也对
    assert r.verification.status == VStatus.TOOL


def test_remainder_answer_in_two_blanks_is_not_false_alarm():
    """真实运行 p_9bcaeb4d 第 12 题（d017）：“每个小朋友分到 ____ 个，还剩 ____ 个”，答案“3，2”。
    只按 17/5 比对会误判为算错，还白白返修了两轮。"""
    q = mk("把 17 个苹果平均分给 5 个小朋友，每个小朋友分到 ____ 个，还剩 ____ 个。", "3，2", QType.FILL)
    r = route(q, bl(expression="17÷5"))
    assert r.verification.status == VStatus.TOOL and not r.mismatch and "3……2" in r.verification.detail
    r = route(q.model_copy(update={"answer": "3，1"}), bl(expression="17÷5"))
    assert r.verification.status == VStatus.PROBLEM
    from exam_agent.agents.writer import QuestionDraft, self_check
    from exam_agent.domain import BlueprintSlot
    slot = BlueprintSlot(slot_id="S1", position=1, qtype=QType.FILL, score=3, knowledge_point="有余数除法",
                         difficulty="中等", cognitive_level="理解", intent="x")
    d = QuestionDraft(slot_id="S1", stem=q.stem, answer="3，2", analysis="", check={"expression": "17÷5"})
    assert self_check(d, slot, REQ) == ""


def test_word_problem_uses_blind_expression():
    q = mk("一盒铅笔 6 支，老师买了 4 盒，一共多少支？", "24 支", QType.APPLICATION)
    assert route(q, bl(expression="6×4")).verification.status == VStatus.TOOL
    r = route(q.model_copy(update={"answer": "20 支"}), bl(expression="6×4"))
    assert r.verification.status == VStatus.PROBLEM and "24" in r.mismatch


def test_without_tool_status_is_honest():
    choice = mk("下面哪个是长方形的特征（ ）", "B", QType.CHOICE, ["a", "b", "c", "d"])
    assert route(choice, bl(choice="B")).verification.status == VStatus.AGREED
    fill = mk("一年有____个月", "12", QType.FILL)
    assert route(fill, bl(final_answer="12个月")).verification.status == VStatus.AGREED
    assert route(mk("看图写出钟面上的时间", "3时")).verification.status == VStatus.UNVERIFIED
    # 工具算不了（二次方程）：不下结论，也不冤枉
    r = route(mk("解方程：x² = 4", "x = 2 或 -2"), bl(equations=["x^2 = 4"], unknowns=["x"], target="x"))
    assert r.verification.status == VStatus.UNVERIFIED and not r.mismatch


def test_blind_issue_marks_problem():
    fs, v = check_question(mk("检验下列各数，哪个是方程的解", "2", QType.FILL), bl(issue="没有给出“下列各数”"), REQ)
    assert [f.code for f in fs] == ["ambiguous"] and v.status == VStatus.PROBLEM


# ---------- 状态随版本保存；老师确认 ----------

def blind_reply(*items):
    return j({"solutions": [{"slot_id": s, "equations": eq, "unknowns": u, "target": u[0] if u else "",
                             "choice": c, "final_answer": "", "issue": ""} for s, eq, u, c in items]})


def test_status_is_stamped_saved_and_confirmable():
    q3 = q("S3", "解方程：3x + 5 = 2x - 1", "x = -6")
    q4 = q("S4", "说一说解方程的步骤", "去括号、移项、合并、系数化为1")      # 工具核对不了
    llm = ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": [Q1, Q2, q3, q4]}),
                       blind_reply(("S1", [], [], "B"), ("S2", ["2x = 6"], ["x"], ""), ("S3", [], [], ""),
                                   ("S4", [], [], "")),
                       j({"issues": []})])
    store = PaperStore(":memory:")
    res = create_paper(llm, store, "七年级一元一次方程小测，4 道题，满分 30 分")
    st = {x.slot_id: x.vstatus for x in res.paper.questions}
    assert st == {"S1": VStatus.AGREED, "S2": VStatus.TOOL, "S3": VStatus.TOOL, "S4": VStatus.UNVERIFIED}
    saved, _ = store.load(res.paper.paper_id)
    assert {x.slot_id: x.vstatus for x in saved.questions} == st         # 状态是试卷状态的一部分，存在数据库里

    pos = {s.slot_id: s.position for s in saved.blueprint.slots}
    paper, v, done = confirm(store, saved.paper_id, [pos["S4"]], note="步骤表述正确")
    s4 = next(x for x in paper.questions if x.slot_id == "S4")
    assert v == 2 and s4.vstatus == VStatus.TEACHER and "待老师确认" in s4.verification.detail
    # 确认不改变题目内容
    assert [x.content_fingerprint() for x in paper.questions] == [x.content_fingerprint() for x in saved.questions]
    assert store.history(saved.paper_id)[-1]["operation"] == "confirm"


def test_no_review_means_not_reviewed():
    llm = ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": [Q1, Q2, q("S3", "解方程：x+1=2", "x = 1"),
                                                                       q("S4", "解方程：x+2=3", "x = 1")]})])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    assert {x.vstatus for x in res.paper.questions} == {VStatus.NOT_REVIEWED}


# ---------- 出卷范围：严格按老师原话 ----------

def test_grade_and_stage_are_normalized_by_code():
    assert normalize_grade("3年级") == "三年级" and normalize_grade("初一") == "七年级"
    d = RequestDraft(request=REQ.model_copy(update={"stage": "初中", "grade": "小学3年级", "topic": ""}))
    r = normalize(d).request
    assert (r.stage, r.grade) == ("小学", "三年级")
    assert r.title == "三年级数学测试卷"
    assert r.model_copy(update={"term": "上"}).title == "三年级上册数学测试卷"
    assert "上下册" in r.scope_text


def test_term_comes_only_from_teacher_words():
    assert term_in_text("出一份小学3年级的数学试卷") == "全年"
    assert term_in_text("三年级上册第二单元") == "上"
    assert term_in_text("三年级下学期期末卷") == "下"
    wrong = RequestDraft(request=REQ.model_copy(update={"term": "上"}))
    assert any("全年" in p for p in _validate(wrong, "出一份小学3年级的数学试卷"))


def test_model_guessing_term_is_sent_back():
    guess = json.loads(REQUEST_REPLY)
    guess["request"].update({"stage": "小学", "grade": "三年级", "topic": "", "term": "上"})
    fixed = json.loads(j(guess))
    fixed["request"]["term"] = "全年"
    llm = ScriptedLLM([j(guess), j(fixed)])
    d = parse_request(llm, "出一份小学3年级的数学试卷，总分100分，共20题")
    assert d.request.term == "全年" and len(llm.calls) == 2
    assert "没有提到上册或下册" in llm.calls[1][-1]["content"]


def test_numeric_choice_is_tool_verified_and_catches_wrong_key():
    q = mk("3 分等于多少秒？", "B", QType.CHOICE, ["30秒", "180秒", "300秒", "18秒"])
    assert route(q, bl(choice="B", expression="3×60")).verification.status == VStatus.TOOL
    r = route(q.model_copy(update={"answer": "C"}), bl(choice="C", expression="3×60"))
    assert r.verification.status == VStatus.PROBLEM and "对应选项 B" in r.mismatch   # 模型一致选错也能发现


def test_multi_part_question_is_not_tool_verified_by_one_expression():
    """e2e 试跑：5 个小题的题，盲解只列了其中一个算式，就显示“工具验证”。"""
    from exam_agent.agents.solver import BlindSolution
    from exam_agent.domain import QType, Question, VStatus
    from exam_agent.verify import is_multi_part, route

    stem = "填一填。\n（1）0 + 4 = ____\n（2）5 - 0 = ____\n（3）0 + 0 = ____"
    q = Question(slot_id="S1", qtype=QType.CALC, score=20, knowledge_point="k", difficulty="基础", stem=stem,
                 answer="（1）4（2）5（3）0", analysis="略")
    one = BlindSolution(slot_id="S1", expression="0 + 4", final_answer="（1）4（2）5（3）0")
    r = route(q, one)
    assert r.verification.status == VStatus.AGREED and "多空题" in r.verification.method
    wrong = BlindSolution(slot_id="S1", expression="0 + 4", final_answer="（1）4（2）5（3）1")
    assert route(q, wrong).verification.status == VStatus.UNVERIFIED
    assert not is_multi_part("17 ÷ 5 = ____……____") and is_multi_part("30 + 20 = ____，45 - 3 = ____。")
