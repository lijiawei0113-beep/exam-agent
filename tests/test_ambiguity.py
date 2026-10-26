"""“两次答错得一样”的相关错误：多答案检查、比较类选择题的工具验证、盲解换模型。"""
import json

from exam_agent.agents.solver import BlindSolution
from exam_agent.domain import ExamRequest, QType, Question, VStatus
from exam_agent.llm import RoutedLLM, ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.review import check_question
from exam_agent.store import PaperStore
from exam_agent.verify import route

from fixtures import GOOD_BLUEPRINT, Q1, Q2, REQUEST_REPLY, q

j = lambda o: json.dumps(o, ensure_ascii=False)
REQ = ExamRequest.model_validate(json.loads(REQUEST_REPLY)["request"])
TABLE = ("下面是三年级一班参加兴趣小组的人数统计表：\n| 组别 | 美术组 | 音乐组 | 体育组 |\n|---|---|---|---|\n"
         "| 男生 | 8 | 6 | 12 |\n| 女生 | 10 | 9 | 7 |\n参加人数最多的兴趣小组是（ ）")


def choice(stem, answer, options):
    return Question(slot_id="S1", qtype=QType.CHOICE, score=4, knowledge_point="k", difficulty="基础",
                    stem=stem, answer=answer, options=options, analysis="略")


def test_comparison_choice_catches_shared_wrong_answer():
    """d019：答案键 A，盲解也选 A（两个模型错得一样）；但盲解列出的各组算式是对的，程序算出最大的是 C。"""
    q = choice(TABLE, "A", ["美术组", "音乐组", "体育组", "无法确定"])
    b = BlindSolution(slot_id="S1", choice="A", option_exprs={"A": "8+10", "B": "6+9", "C": "12+7"}, pick="max")
    r = route(q, b)
    assert r.verification.status == VStatus.PROBLEM and "最大的是 C" in r.mismatch and "C=19" in r.mismatch
    ok = route(q.model_copy(update={"answer": "C"}), b)
    assert ok.verification.status == VStatus.TOOL
    tie = BlindSolution(slot_id="S1", choice="A", option_exprs={"A": "9+10", "C": "12+7"}, pick="max")
    assert route(q, tie).verification.status == VStatus.AGREED   # 并列最大：工具不下结论，退回到比对选项


def test_multiple_defensible_options_is_must_fix():
    """p_1a787ead 第 20 题第 5 小题：短文同时写了 hungry 和 happy，问 How does Ben feel?"""
    q = Question(slot_id="S1", qtype=QType.READING, score=15, knowledge_point="k", difficulty="中等",
                 stem="I'm Ben. Today I'm hungry. ... I'm happy.\n5. How does Ben feel? A. Hungry B. Funny C. Happy D. Cool",
                 answer="5. C", analysis="略")
    b = BlindSolution(slot_id="S1", final_answer="5. C",
                      ambiguous_items=["第5小题：A 和 C 都能成立，短文有 I'm hungry 和 I'm happy"])
    fs, v = check_question(q, b, REQ)
    assert [f.code for f in fs] == ["multiple_answers"] and v.status == VStatus.PROBLEM
    assert "只有一个正确答案" in fs[0].message


def test_blind_solver_can_use_a_different_model():
    blind_reply = j({"solutions": [{"slot_id": "S1", "choice": "B"}, {"slot_id": "S2", "equations": ["2x = 6"],
                                    "unknowns": ["x"], "target": "x"}, {"slot_id": "S3"}, {"slot_id": "S4"}]})
    main = ScriptedLLM([REQUEST_REPLY, GOOD_BLUEPRINT,
                        j({"questions": [Q1, Q2, q("S3", "解方程：x+1=2", "x = 1"), q("S4", "解方程：x+2=3", "x = 1")]}),
                        j({"issues": []})], model="writer-model")
    blind = ScriptedLLM([blind_reply], model="blind-model")
    llm = RoutedLLM(main, {"blind_solver": blind})
    res = create_paper(llm, PaperStore(":memory:"), "七年级一元一次方程小测，4 道题，满分 30 分")
    assert len(blind.calls) == 1 and "看不到标准答案" in blind.calls[0][0]["content"]   # 盲解走另一个模型
    assert main.replies == [] and blind.replies == []
    assert res.review.stop_reason == "passed"
    assert llm.describe() == "writer-model；blind_solver=blind-model"
