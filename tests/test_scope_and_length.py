"""d037：不支持的学科/年级给出明确提示；题数、满分未指定时用固定默认值。
d038：英语阅读按目标词数单独命题；仍然太短时保留原句扩写，而不是整篇重写。"""
import json

from exam_agent.agents.base import UnsupportedRequest
from exam_agent.agents.requirement import RequestDraft, apply_defaults, check_supported
from exam_agent.agents.writer import QuestionDraft, expand_reading, keeps_original, replace_passage, write_questions
from exam_agent.domain import BlueprintSlot, ExamRequest, QType
from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.store import PaperStore
from exam_agent.tools.english import english_words, passage_of

j = lambda o: json.dumps(o, ensure_ascii=False)


def req(**kw):
    base = dict(stage="小学", grade="三年级", subject="英语", term="上", topic="", question_count=4, total_score=40,
                required_types=["阅读理解"])
    base.update(kw)
    return ExamRequest(**base)


def raises(fn):
    try:
        fn()
    except UnsupportedRequest as e:
        return str(e)
    raise AssertionError("应当报不支持")


def test_unsupported_subject_and_grade_get_clear_message():
    msg = raises(lambda: check_supported(req(subject="语文", stage="初中", grade="七年级")))
    assert "暂不支持语文" in msg and "数学一至七年级" in msg and "英语三至七年级" in msg
    assert "暂不支持二年级" in raises(lambda: check_supported(req(grade="二年级")))
    assert "暂不支持八年级" in raises(lambda: check_supported(req(subject="数学", stage="初中", grade="八年级")))
    check_supported(req())                                    # 支持范围内不报错


def test_chinese_paper_fails_fast_without_calling_planner():
    """p 图 1：语文卷走到规划蓝图才失败，提示“选择题 1 题凑不成满分 1”，老师看不懂。"""
    reply = j({"request": req(subject="语文", stage="初中", grade="七年级", question_count=1, total_score=1,
                              required_types=["选择题"]).model_dump(mode="json"), "assumptions": []})
    llm = ScriptedLLM([reply])
    try:
        create_paper(llm, PaperStore(":memory:"), "出一份七年级语文试卷")
        assert False
    except UnsupportedRequest as e:
        assert "暂不支持语文" in str(e) and "planner" not in str(e)
    assert len(llm.calls) == 1                                 # 只调用了一次需求解析


def test_defaults_when_teacher_gives_no_count_or_score():
    d = RequestDraft(request=req(question_count=1, total_score=1), assumptions=["题数按 1 题", "按上册出题"])
    out = apply_defaults(d, "出一份三年级上册英语试卷")
    assert out.request.question_count == 20 and out.request.total_score == 100
    assert "按上册出题" in out.assumptions and not any("1 题" in a for a in out.assumptions)
    kept = apply_defaults(d, "三年级英语，10 道题，满分 50")
    assert kept.request.question_count == 1 and kept.request.total_score == 1    # 老师说了就不覆盖
    assert apply_defaults(d, "共十五题，总分为100").request.total_score == 1


STEM = ("Read the passage and choose the correct answer.\n"
        "I am Amy. I have a cat. It is white. I like my cat.\n"
        "1. What colour is the cat? A. Black B. White C. Red\n"
        "2. Who has a cat? A. Amy B. Mike C. Sam")
SLOT = BlueprintSlot(slot_id="S1", position=1, qtype=QType.READING, score=10, knowledge_point="k",
                     difficulty="基础", cognitive_level="理解", intent="x")


ORIGINAL = "I am Amy. I have a cat. It is white. I like my cat."
EXPANDED = ("I am Amy. I am nine years old and I live in a small house. I have a cat. Her name is Mimi. "
            "It is white. She has big eyes and a short tail.\n"
            "Every morning Mimi sleeps on my bed. After school we play in the garden together. "
            "My mother gives her some milk and fish every day. Mimi is very happy at home. "
            "I like my cat. She is my best friend.")


def test_keeps_original_detects_dropped_or_changed_sentences():
    assert keeps_original(ORIGINAL, EXPANDED) == []
    assert keeps_original(ORIGINAL, EXPANDED.replace("It is white.", "It is black.")) == ["it is white."]
    assert keeps_original(ORIGINAL, EXPANDED.replace("I like my cat.", "")) == ["i like my cat."]


def test_replace_passage_keeps_instruction_and_items():
    out = replace_passage(STEM, EXPANDED).splitlines()
    assert out[0] == STEM.splitlines()[0] and out[-2:] == STEM.splitlines()[-2:]
    assert out[1].startswith("I am Amy.") and len(out) == 5           # 扩写稿两段 + 说明 + 两道小题


def test_short_passage_is_expanded_keeping_original_sentences():
    d = QuestionDraft(slot_id="S1", stem=STEM, options=[], answer="1. B 2. A", analysis="略")
    changed = EXPANDED.replace("It is white.", "It is black.")
    llm = ScriptedLLM([j({"passage": changed}), j({"passage": EXPANDED})])
    out = expand_reading(llm, d, SLOT, req())
    assert len(english_words(passage_of(out.stem))) >= 50 and out.answer == d.answer and "1. What colour" in out.stem
    assert "it is white." in llm.calls[1][-1]["content"]              # 改了原句 → 带着具体句子重试
    assert "约 105 词" in llm.calls[0][-1]["content"] and "3 段" in llm.calls[0][-1]["content"]


def test_writer_expands_short_reading_before_self_check():
    llm = ScriptedLLM([j({"questions": [{"slot_id": "S1", "stem": STEM, "options": [], "answer": "1. B 2. A",
                                         "analysis": "略"}]}),
                       j({"passage": EXPANDED})])
    res = write_questions(llm, req(), [SLOT])
    assert res.problems == [] and len(res.questions) == 1
    assert len(english_words(passage_of(res.questions[0].stem))) >= 50
    assert "只写这一篇阅读理解" in llm.calls[0][-1]["content"]


# ---------- d039：计算题答案算错时由代码更正；多轮仍不通过的题标为问题题，不让整卷失败 ----------

MREQ = dict(stage="初中", grade="七年级", subject="数学", term="上", topic="", question_count=1, total_score=10,
            required_types=["计算题"])
CSLOT = BlueprintSlot(slot_id="S1", position=1, qtype=QType.CALC, score=10, knowledge_point="有理数的混合运算",
                      difficulty="中等", cognitive_level="掌握", intent="x")
EXPR = "-2² + (-3)² × (-1/3) - (-8) ÷ 4"


def test_wrong_answer_to_pure_calculation_is_corrected_by_tool():
    draft = {"slot_id": "S1", "stem": f"计算：{EXPR}", "options": [], "answer": "-3", "analysis": "先算乘方……得 -3",
             "check": {"expression": EXPR}}
    llm = ScriptedLLM([j({"questions": [draft]}), j({"analysis": "-2²=-4，(-3)²×(-1/3)=-3，-(-8)÷4=2，合计 -5"})])
    res = write_questions(llm, ExamRequest(**MREQ), [CSLOT])
    assert res.problems == [] and res.corrected == ["S1"]
    q = res.questions[0]
    assert q.answer == "-5" and "-5" in q.analysis and "正确答案：-5" in llm.calls[1][-1]["content"]


def test_unresolved_slot_is_kept_and_flagged_instead_of_failing_paper():
    from exam_agent.agents.writer import fill_slots
    from exam_agent.domain import VStatus

    slot = CSLOT.model_copy(update={"qtype": QType.APPLICATION})
    bad = {"slot_id": "S1", "stem": "小明有 12 颗糖，又买了 5 颗，一共有多少颗？", "options": [], "answer": "16 颗",
           "analysis": "12+5=16", "check": {"expression": "12+5"}}
    llm = ScriptedLLM([j({"questions": [bad]})] * 3)
    out = fill_slots(llm, ExamRequest(**{**MREQ, "required_types": ["应用题"]}), [slot])
    q = out["S1"]
    assert q.vstatus == VStatus.PROBLEM and "12+5 = 17" in q.verification.detail


def test_paper_is_saved_with_flagged_question_and_teacher_is_told():
    from fixtures import GOOD_BLUEPRINT, Q1, Q2, REQUEST_REPLY, q as fq
    from exam_agent.domain import VStatus

    bad = {**fq("S4", "小明有 12 颗糖，又买了 5 颗，一共有多少颗？", "16 颗"), "check": {"expression": "12+5"}}
    good = [Q1, Q2, fq("S3", "解方程：x + 1 = 2", "x = 1")]
    llm = ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": good + [bad]}), j({"questions": [bad]}),
                       j({"questions": [bad]})])
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    assert res.paper.question_by_position(4).vstatus == VStatus.PROBLEM
    assert any("第 4 题经多轮返修" in a for a in res.assumptions)


# ---------- d040：题数、满分、题型凑不成一份卷时，规划前就由代码处理 ----------

def _mreq(**kw):
    base = dict(stage="初中", grade="七年级", subject="数学", term="全年", topic="", question_count=5, total_score=100,
                required_types=[])
    base.update(kw)
    return ExamRequest(**base)


def test_five_questions_become_five_calculations_worth_100():
    """老师的规则：没说满分就是 100；5 题凑不成时直接出 5 道计算题，每题 20 分（d040）。"""
    from exam_agent.agents.requirement import resolve_feasibility
    from exam_agent.scoring import feasible_count_options

    d = apply_defaults(RequestDraft(request=_mreq()), "七年纪数学试卷，5题")
    out = resolve_feasibility(d, "七年纪数学试卷，5题")
    assert out.request.total_score == 100 and [t.value for t in out.request.required_types] == ["计算题"]
    assert "题型安排为计算题" in out.assumptions[-1]
    assert feasible_count_options(5, 100, out.request.required_types) == [{QType.CALC: 5}]


def test_score_ranges_follow_teacher_rule():
    from exam_agent.scoring import SCORE_RANGE

    assert (SCORE_RANGE[QType.CHOICE], SCORE_RANGE[QType.FILL]) == (range(1, 7), range(1, 7))
    assert SCORE_RANGE[QType.CALC] == range(10, 21)


def test_impossible_requests():
    from exam_agent.agents.requirement import resolve_feasibility

    msg = raises(lambda: resolve_feasibility(RequestDraft(request=_mreq(required_types=["选择题", "填空题"])),
                                             "5 道选择填空，100 分"))
    assert "5 道题凑不成 100 分" in msg and "选择题 1～6 分" in msg
    d = apply_defaults(RequestDraft(request=_mreq(question_count=3)), "七年级数学 3 题")
    out = resolve_feasibility(d, "七年级数学 3 题")                          # 没说满分：降到能凑成的整十分
    assert out.request.total_score == 60 and "按 60 分出卷" in out.assumptions[-2]
    msg = raises(lambda: resolve_feasibility(RequestDraft(request=_mreq(question_count=3)), "3 题 100 分"))
    assert "3 道题凑不成 100 分" in msg and "计算题 10～20 分" in msg


def test_feasible_request_is_untouched():
    from exam_agent.agents.requirement import resolve_feasibility

    d = RequestDraft(request=_mreq(question_count=20))
    assert resolve_feasibility(d, "七年级数学，20 题，100 分") == d


def test_old_paper_with_out_of_range_scores_can_still_be_edited():
    """d041：v9.5 收紧分值范围后，旧卷（计算题每题 6 分）任何修改都被拒绝；ReAct 还编造了“只有 92 分”的解释。"""
    from exam_agent.agents.editor import EditPlan
    from exam_agent.domain import Blueprint, PaperSnapshot, Question
    from exam_agent.editing import apply_ops

    spec = [(QType.CHOICE, 1)] * 10 + [(QType.FILL, 4)] * 5 + [(QType.CALC, 6)] * 8 + [(QType.APPLICATION, 11)] * 2
    slots, qs = [], []
    for i, (t, sc) in enumerate(spec, 1):
        slots.append(BlueprintSlot(slot_id=f"S{i}", position=i, qtype=t, score=sc, knowledge_point="k",
                                   difficulty="基础", cognitive_level="掌握", intent="x"))
        qs.append(Question(slot_id=f"S{i}", qtype=t, score=sc, knowledge_point="k", difficulty="基础",
                           stem=f"题{i}", options=["1", "2", "3", "4"] if t == QType.CHOICE else [],
                           answer="A" if t == QType.CHOICE else "1", analysis="略"))
    paper = PaperSnapshot(paper_id="p", title="t", request=_mreq(question_count=25, required_types=[]),
                          blueprint=Blueprint(slots=slots), questions=qs)
    assert paper.total_score == 100
    out = apply_ops(paper, EditPlan.model_validate({"action": "edit", "reply": "好", "ops": [
        {"op": "rewrite", "positions": [14], "difficulty": "基础"}]}))
    assert out.problems == [] and [s.score for s in out.blueprint.slots] == [sc for _, sc in spec]
    bad = apply_ops(paper, EditPlan.model_validate({"action": "edit", "reply": "好", "ops": [
        {"op": "set_total", "total": 300}]}))
    assert bad.problems and "修改前卷面总分为 100" in bad.problems[0] and "不要自行计算" in bad.problems[0]


# ---------- e2e 第一轮：题型“必须包含”与“只要”；题数、满分为空 ----------

def test_required_types_mean_must_include_unless_teacher_says_only():
    from exam_agent.agents.requirement import resolve_feasibility
    from exam_agent.domain import plan_types

    r = _mreq(grade="四年级", stage="小学", question_count=20, required_types=["应用题"])
    assert [t.value for t in plan_types(r)] == ["选择题", "填空题", "计算题", "应用题"]
    d = RequestDraft(request=r)
    assert resolve_feasibility(d, "四年级上册数学期中卷，20道题，总分100，要有应用题") == d     # 不再拒绝
    only = apply_defaults(RequestDraft(request=_mreq(question_count=2, required_types=["应用题"])),
                          "七年级数学，只要应用题，2题")
    assert only.request.only_types and plan_types(only.request) == [QType.APPLICATION]
    en = ExamRequest(stage="小学", grade="五年级", subject="英语", term="全年", topic="", question_count=2,
                     total_score=100, required_types=["阅读理解"])
    d2 = apply_defaults(RequestDraft(request=en), "五年级英语，只要阅读理解，2篇")
    assert d2.request.question_count == 2 and d2.request.only_types


def test_null_or_zero_count_and_total_from_model_do_not_crash():
    raw = {"request": {"stage": "小学", "grade": "三年级", "subject": "英语", "term": "上", "topic": "",
                       "question_count": None, "total_score": 0, "required_types": []}, "assumptions": []}
    d = RequestDraft.model_validate(raw)
    assert d.request.question_count == 20 and d.request.total_score == 100
    d = RequestDraft.model_validate({**raw, "request": {**raw["request"], "question_count": "8题",
                                                        "total_score": None}})
    assert d.request.question_count == 8 and d.request.total_score == 100


def test_add_constraint_without_constraint_object_is_built_by_code():
    from exam_agent.agents.editor import EditPlan
    from exam_agent.domain import ConstraintKind
    from exam_agent.editing import apply_ops
    from fixtures import ALL_QUESTIONS, GOOD_BLUEPRINT, REQUEST_REPLY

    store = PaperStore(":memory:")
    paper = create_paper(ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS]), store,
                         "七年级一元一次方程小测，4 道题，满分 30 分", review=False).paper
    out = apply_ops(paper, EditPlan.model_validate({"action": "edit", "reply": "好", "ops": [
        {"op": "add_constraint", "instruction": "计算题答案都要是整数"}]}))
    c = out.new_constraints[0]
    assert out.problems == [] and c.kind == ConstraintKind.INTEGER_ANSWER and [t.value for t in c.scope_types] == ["计算题"]


def test_picture_reference_kept_as_flagged_question_after_rounds():
    """e2e 第二轮 r070：某题三轮都写“看图”，整卷失败。现在保留并标为问题题交给老师。"""
    from exam_agent.agents.writer import fill_slots
    from exam_agent.domain import VStatus

    slot = BlueprintSlot(slot_id="S1", position=1, qtype=QType.FILL, score=5, knowledge_point="k",
                         difficulty="基础", cognitive_level="掌握", intent="x")
    pic = {"slot_id": "S1", "stem": "看图写单词：____", "options": [], "answer": "cat", "analysis": "略"}
    out = fill_slots(ScriptedLLM([j({"questions": [pic]})] * 3), req(required_types=["填空题"]), [slot])
    assert out["S1"].vstatus == VStatus.PROBLEM and "图片" in out["S1"].verification.detail


# ---------- d046：留出集暴露的问题 ----------

def test_flagged_picture_question_does_not_block_whole_paper():
    """引用图片的题多轮改不掉：标为问题题后，整卷校验也要放行（v9.10 只在命题阶段放行，整卷照样失败）。"""
    from fixtures import GOOD_BLUEPRINT, Q1, Q2, REQUEST_REPLY, q as fq
    from exam_agent.domain import VStatus

    pic = fq("S4", "看图列式计算：____", "x = 1")
    good = [Q1, Q2, fq("S3", "解方程：x + 1 = 2", "x = 1")]
    for review in (False, True):
        replies = [REQUEST_REPLY, GOOD_BLUEPRINT, j({"questions": good + [pic]}), j({"questions": [pic]}),
                   j({"questions": [pic]})]
        if review:
            replies += [j({"solutions": [{"slot_id": s} for s in ("S1", "S2", "S3", "S4")]}), j({"issues": []})]
        res = create_paper(ScriptedLLM(replies), PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分",
                           review=review)
        q4 = res.paper.question_by_position(4)
        assert q4.vstatus == VStatus.PROBLEM and "图片" in q4.verification.detail


def test_reply_claiming_done_without_edit_is_rejected():
    from exam_agent.editing import claims_executed

    assert claims_executed("已把您的要求记为整卷持续约束：计算题答案都要是整数")
    assert claims_executed("好的，已为您将第 3 题替换为基础题")
    assert not claims_executed("已核对：第 9～12 题答案都是整数")
    assert not claims_executed("第 1 题选 A，因为 298 + 305 = 603 最接近 600")


def test_small_unit_allows_more_questions_per_point():
    from exam_agent.agents.planner import check_concentration
    from exam_agent.domain import Blueprint

    bp = Blueprint.model_validate({"slots": [
        {"slot_id": f"S{i}", "position": i, "qtype": "选择题", "score": 10, "knowledge_point": "事件发生的确定与不确定",
         "difficulty": "基础", "cognitive_level": "理解", "intent": "x"} for i in range(1, 11)][:4] + [
        {"slot_id": f"S{i}", "position": i, "qtype": "选择题", "score": 10, "knowledge_point": "可能性的大小",
         "difficulty": "基础", "cognitive_level": "理解", "intent": "x"} for i in range(5, 11)][:4] + [
        {"slot_id": f"S{i}", "position": i, "qtype": "选择题", "score": 10, "knowledge_point": f"其他{i}",
         "difficulty": "基础", "cognitive_level": "理解", "intent": "x"} for i in (9, 10)]})
    assert check_concentration(bp)                       # 不知道范围大小：按原规则打回
    assert check_concentration(bp, n_points=2) == []     # “可能性”单元只有 2 个知识点：允许
