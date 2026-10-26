import json

from exam_agent.domain import Blueprint, ExamRequest, PaperSnapshot, Question
from exam_agent.invariants import check_blueprint, check_paper, check_preserved

from fixtures import BAD_BLUEPRINT, GOOD_BLUEPRINT, REQUEST_REPLY


def make_request() -> ExamRequest:
    return ExamRequest.model_validate(json.loads(REQUEST_REPLY)["request"])


def make_paper() -> PaperSnapshot:
    bp = Blueprint.model_validate_json(GOOD_BLUEPRINT)
    qs = []
    for s in bp.slots:
        opts = ["a", "b", "c", "d"] if s.qtype.value == "选择题" else []
        ans = "A" if opts else "1"
        stem = "x = ____" if s.qtype.value == "填空题" else f"题目{s.slot_id}"
        qs.append(Question(slot_id=s.slot_id, qtype=s.qtype, score=s.score, knowledge_point=s.knowledge_point,
                           difficulty=s.difficulty, stem=stem, options=opts, answer=ans, analysis="略"))
    return PaperSnapshot(paper_id="p_test", title="t", request=make_request(), blueprint=bp, questions=qs)


def test_good_blueprint_passes():
    assert check_blueprint(make_request(), Blueprint.model_validate_json(GOOD_BLUEPRINT)) == []


def test_wrong_total_detected():
    codes = [v.code for v in check_blueprint(make_request(), Blueprint.model_validate_json(BAD_BLUEPRINT))]
    assert codes == ["total"]


def test_missing_type_detected():
    req = ExamRequest.model_validate({**make_request().model_dump(), "required_types": ["应用题"]})
    codes = [v.code for v in check_blueprint(req, Blueprint.model_validate_json(GOOD_BLUEPRINT))]
    assert "missing_type" in codes


def test_uneven_score_within_type_detected():
    # 复现第一次真实运行的问题：选择题出现 5 分和 10 分混用
    bp = Blueprint.model_validate_json(GOOD_BLUEPRINT)
    bp.slots[3].score = 5          # 两道计算题变成 10 分和 5 分
    bp.slots[0].score = 10         # 总分仍为 30
    codes = [v.code for v in check_blueprint(make_request(), bp)]
    assert codes == ["uneven_type_score"]


def test_json_with_raw_newline_is_accepted():
    # 复现第一次真实运行的问题：字符串中有原始换行导致 JSON 解析失败、白白重试一次
    from exam_agent.agents.base import structured_call
    from exam_agent.llm import ScriptedLLM
    raw = '{"slots": [], "rationale": "第一行\n第二行"}'
    res = structured_call(ScriptedLLM([raw]), step="t", system="", user="", schema=Blueprint)
    assert res.attempts == 1 and "第二行" in res.value.rationale


def test_valid_paper_passes():
    assert check_paper(make_paper()) == []


def test_score_tampering_detected():
    paper = make_paper()
    paper.questions[0].score = 99
    codes = {v.code for v in check_paper(paper)}
    assert "score_mismatch" in codes


def test_preserved_detects_unrequested_change():
    old = make_paper()
    new = old.model_copy(deep=True)
    new.questions[2].stem = "被偷偷改掉的题干"           # S3，不在允许范围内
    new.questions[1].stem = "允许修改的题 x = ____"      # S2，在允许范围内
    v = check_preserved(old, new, allowed_slots={"S2"})
    assert [x.location for x in v] == ["S3"]


def test_picture_reference_is_rejected():
    """d036：命题员写“根据图片提示写出单词：鼻子 ____”，但试卷不出图。"""
    from exam_agent.domain import QType
    from exam_agent.invariants import check_question_format
    bad = check_question_format(QType.FILL, "根据图片提示写出单词：鼻子 ____", [], "nose", "S13")
    assert [v.code for v in bad] == ["no_picture"]
    assert check_question_format(QType.FILL, "根据中文提示写出单词：鼻子 ____", [], "nose", "S13") == []
    assert check_question_format(QType.CALC, "如图，长方形的长是 5 厘米……求周长", [], "16", "S2")[0].code == "no_picture"
