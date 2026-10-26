"""阶段五：导出学生版和答案版 PDF。"""
import json

import pytest

from exam_agent.editing import apply_edit
from exam_agent.export_pdf import ExportError, _section_title, _sections, export_paper, register_fonts
from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.store import PaperStore

from fixtures import ALL_QUESTIONS, GOOD_BLUEPRINT, REQUEST_REPLY


def make_paper():
    store = PaperStore(":memory:")
    res = create_paper(ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT, ALL_QUESTIONS]), store, "七年级一元一次方程小测，4 道题，满分 30 分", review=False)
    return store, res.paper


def fonts_or_skip():
    try:
        return register_fonts()
    except ExportError as e:
        pytest.skip(str(e))


def test_sections_follow_blueprint_numbering():
    _, paper = make_paper()
    secs = _sections(paper)
    assert [t.value for t, _ in secs] == ["选择题", "填空题", "计算题"]
    assert [no for _, items in secs for no, _ in items] == [1, 2, 3, 4]
    assert _section_title(2, secs[2][0], secs[2][1]) == "三、计算题（本大题共 2 小题，每小题 10 分，共 20 分）"


def test_export_creates_student_and_answer_pdfs(tmp_path):
    fonts_or_skip()
    _, paper = make_paper()
    paper.questions[2].stem = "解方程：x < 3 & x > 1 时……"   # 排版标记字符不能导致崩溃
    student, answers = export_paper(paper, 1, tmp_path)
    for p in (student, answers):
        assert p.exists() and p.read_bytes()[:5] == b"%PDF-" and p.stat().st_size > 2000
    assert "学生版" in student.name and "答案版" in answers.name


def test_question_mark_never_starts_a_line():
    """实测问题：ReportLab 默认会把全角“？”单独折到下一行。"""
    fonts = fonts_or_skip()
    from exam_agent.export_pdf import _p, _styles
    st = _styles(fonts)["q"]
    text = ("10. 一项道路维修工程，甲队单独施工需要 12 天完成，乙队单独施工需要 24 天完成。"
            "现在先由甲队单独施工 3 天，然后甲、乙两队合作完成剩下的工程。求甲、乙两队合作施工了多少天？")
    for width in range(300, 480, 7):
        para = _p(text, st)
        para.wrap(width, 2000)
        for line in para.blPara.lines[1:]:
            first = "".join(line[1])[:1]
            assert first not in "？，。、；：", f"宽度 {width} 时出现以“{first}”开头的行"


def test_export_intent_does_not_create_a_version():
    store, paper = make_paper()
    llm = ScriptedLLM([json.dumps({"action": "export", "ops": [], "reply": "好的，导出学生版和答案版。"},
                                  ensure_ascii=False)])
    res = apply_edit(llm, store, paper.paper_id, "导出学生版和答案版", review=False)
    assert res.action == "export" and res.version == 1 and store.current_version(paper.paper_id) == 1
