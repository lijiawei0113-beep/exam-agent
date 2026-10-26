"""英语：题型、验证路由、词汇范围核对、知识库构建、检索接入。"""
import json

from exam_agent import kbuild
from exam_agent.agents.requirement import RequestDraft, _validate
from exam_agent.domain import Blueprint, ExamRequest, QType, Question, VStatus
from exam_agent.grounding import Grounding, MultiGrounding
from exam_agent.invariants import check_blueprint
from exam_agent.llm import ScriptedLLM
from exam_agent.pipeline import create_paper
from exam_agent.retrieval.kb import KnowledgeBase, Scope, load_kb
from exam_agent.store import PaperStore
from exam_agent.tools.vocab import lemmas, out_of_vocabulary
from exam_agent.verify import reading_answers, route
from exam_agent.agents.solver import BlindSolution

j = lambda o: json.dumps(o, ensure_ascii=False)
EN_CLAUSES = load_kb("english").clauses


def en_req(**kw):
    base = dict(stage="小学", grade="三年级", subject="英语", term="上", topic="", question_count=4, total_score=40,
                required_types=["选择题", "填空题", "阅读理解", "书面表达"])
    base.update(kw)
    return ExamRequest(**base)


def mk(qtype, stem, answer, options=None):
    return Question(slot_id="S1", qtype=qtype, score=5, knowledge_point="k", difficulty="基础",
                    stem=stem, answer=answer, options=options or [], analysis="略")


# ---------- 学科与题型 ----------

def test_types_follow_subject():
    bp = Blueprint.model_validate({"slots": [
        {"slot_id": "S1", "position": 1, "qtype": "计算题", "score": 40, "knowledge_point": "x",
         "difficulty": "基础", "cognitive_level": "掌握", "intent": "x"}]})
    assert any(v.code == "type_subject" for v in check_blueprint(en_req(question_count=1, required_types=[]), bp))
    d = RequestDraft(request=en_req(required_types=["计算题"]))
    assert any("英语没有这些题型" in p for p in _validate(d, "三年级上册英语"))


def test_grade_one_two_is_practice_not_test():
    r = en_req(grade="二年级", subject="数学", term="全年")
    assert r.title == "二年级数学练习卷"
    assert en_req().title == "三年级上册英语测试卷"


# ---------- 验证路由 ----------

def test_writing_is_honestly_unverified():
    r = route(mk(QType.WRITING, "以 My family 为题写 5 句话", "This is my family. ..."))
    assert r.verification.status == VStatus.UNVERIFIED and "主观题" in r.verification.method


def test_reading_compared_item_by_item():
    assert reading_answers("1. B 2、C 3) a") == {1: "B", 2: "C", 3: "A"}
    q = mk(QType.READING, "短文……\n1. ...\n2. ...", "1. B 2. C")
    ok = route(q, BlindSolution(slot_id="S1", final_answer="1.B 2.C"))
    assert ok.verification.status == VStatus.AGREED
    bad = route(q, BlindSolution(slot_id="S1", final_answer="1.B 2.D"))
    assert bad.verification.status == VStatus.PROBLEM and bad.code == "reading_mismatch" and "第2小题" in bad.mismatch


def test_english_fill_agreement_and_no_false_accusation():
    q = mk(QType.FILL, "She ____ (like) apples.", "likes")
    assert route(q, BlindSolution(slot_id="S1", final_answer="Likes")).verification.status == VStatus.AGREED
    other = route(q, BlindSolution(slot_id="S1", final_answer="like"))
    assert other.verification.status == VStatus.UNVERIFIED and not other.mismatch   # 不一致只交给老师，不判错


# ---------- 词汇范围 ----------

def test_lemmas_and_out_of_vocabulary():
    assert "go" in lemmas("went") and "apple" in lemmas("apples") and "big" in lemmas("bigger")
    text = "Amy has a red apple. She likes elephants. Look at the Great Wall."
    assert out_of_vocabulary(text, {"red", "apple", "like", "look"}) == ["elephants"]


# ---------- 知识库构建（英语） ----------

def draft(units, **kw):
    return {"units": [{"unit": i, "name": n, "description": "d", "key_points": ["问候", "介绍自己"],
                       "boundary": "不涉及过去时", "vocabulary": ["hello", "hi", "name", "goodbye", "bye"],
                       "patterns": ["What's your name? My name's ..."], "signals": [], "standard_refs": ["E1-VOC"],
                       **kw} for i, n in enumerate(units, start=1)]}


def test_english_build_validation():
    units = ["Unit 1 Hello!"]
    kb = KnowledgeBase("english", [], EN_CLAUSES)
    bad = kbuild.BookDraft.model_validate(draft(units, vocabulary=["Hello", "你好"], signals=["你好"]))
    problems = kbuild.validate_book(bad, "三上", units, kb, "english")
    assert any("小写英文" in p for p in problems) and any("必须是英文" in p for p in problems)
    llm = ScriptedLLM([j(draft(units))])
    cards = kbuild.build_book(llm, "三上", kb, {"edition": "PEP", "books": {"三上": units}}, subject="english")
    assert cards[0].id == "E3A-01" and cards[0].vocabulary[0] == "hello" and cards[0].subject == "英语"
    assert "Go for it" in kbuild.SYSTEM_EN or "PEP" in kbuild.SYSTEM_EN


def test_english_signal_must_be_new_word():
    earlier = kbuild.skeleton_cards("三上", ["Unit 1 Hello!"], "e", "english")
    earlier[0].vocabulary = ["yesterday"]
    kb = KnowledgeBase("english", earlier, EN_CLAUSES)
    d = kbuild.BookDraft.model_validate(draft(["Unit 2 Last weekend"], signals=["yesterday"], standard_refs=["E2-VOC"]))
    assert any("已经出现过" in p for p in kbuild.validate_book(d, "六下", ["Unit 2 Last weekend"], kb, "english"))


# ---------- 检索接入 + 端到端 ----------

def english_kb() -> KnowledgeBase:
    c = kbuild.skeleton_cards("三上", ["Unit 1 Hello!", "Unit 2 Colours"], "PEP", "english")
    c[0] = c[0].model_copy(update={"vocabulary": ["hello", "name", "goodbye"], "key_points": ["问候"]})
    c[1] = c[1].model_copy(update={"vocabulary": ["red", "green", "blue", "colour", "look"], "key_points": ["颜色"]})
    later = kbuild.skeleton_cards("六下", ["Unit 2 Last weekend"], "PEP", "english")
    later[0] = later[0].model_copy(update={"vocabulary": ["yesterday"], "signals": ["yesterday", "last weekend"]})
    return KnowledgeBase("english", c + later, EN_CLAUSES)


def test_english_grounding_scope_vocab_and_signals():
    g = Grounding(english_kb())
    c = g.candidates(en_req())
    assert [p.id for p in c.points] == ["E3A-01", "E3A-02"]
    assert "red" in c.vocabulary and "yesterday" not in c.vocabulary
    assert c.signals == {}            # 英语不按单词判超纲（老师的要求：个别生词可以出现）
    mg = MultiGrounding({"英语": g})
    assert mg.candidates(en_req(grade="一年级")) is None
    assert "三年级开设" in mg.explain_missing(en_req(grade="一年级"))


def test_english_paper_end_to_end():
    """三年级上册：阅读短文用了一般过去时（课标二级）→ 命题时代码自查就发现并只重写这一题；个别生词（kite）不判错。"""
    request = {"request": en_req().model_dump(mode="json"), "assumptions": []}
    slot = lambda i, t, kp, sc: {"slot_id": f"S{i}", "position": i, "qtype": t, "score": sc, "knowledge_point": "x",
                                 "difficulty": "基础", "cognitive_level": "掌握", "intent": "x", "kp_id": kp}
    bp = {"slots": [slot(1, "选择题", "E3A-01", 5), slot(2, "填空题", "E3A-02", 5), slot(3, "阅读理解", "E3A-02", 15),
                    slot(4, "书面表达", "E3A-01", 15)], "rationale": ""}
    q = lambda s, stem, a, o=None: {"slot_id": s, "stem": stem, "options": o or [], "answer": a, "analysis": "略"}
    filler = " Look at my bag. It is blue. Look at my pen. It is red. I like my pen. This is my friend Amy." * 3
    past = ("阅读短文，回答问题。\nYesterday I went to the park. I saw a red kite. I liked it." + filler + "\n"
            "1. What colour was the kite? A. Red. B. Blue.\n2. Did the writer like it? A. Yes. B. No.")
    present = ("阅读短文，回答问题。\nLook at my kite. It is red and green. I like it." + filler + "\n"
               "1. What colour is the kite? A. Red and green. B. Blue.\n2. Does the writer like it? A. Yes. B. No.")
    qs = [q("S1", "—Hello! What's your name? —____", "B", ["Goodbye.", "My name's Amy.", "Red.", "Thank you."]),
          q("S2", "Look! The apple is ____ (红色的).", "red"),
          q("S3", past, "1. A 2. A"),
          q("S4", "用 3～5 句话介绍你喜欢的颜色。", "I like blue. ...")]
    blind = j({"solutions": [{"slot_id": "S1", "choice": "B"}, {"slot_id": "S2", "final_answer": "red"},
                             {"slot_id": "S3", "final_answer": "1. A 2. A"}, {"slot_id": "S4"}]})
    llm = ScriptedLLM([j(request), j(bp), j({"questions": [qs[0], qs[1], qs[3]]}),
                       j({"questions": [qs[2]]}),                                  # 阅读理解单独一次调用
                       j({"questions": [q("S3", present, "1. A 2. A")]}),         # 命题时代码自查发现过去时，只重写 S3
                       blind, j({"issues": []})])
    res = create_paper(llm, PaperStore(":memory:"), "出一份三年级上册英语试卷，4 道题，满分 40 分",
                       grounding=MultiGrounding({"英语": Grounding(english_kb())}))
    assert "只写这一篇阅读理解" in llm.calls[3][-1]["content"] and '"S3"' not in llm.calls[2][-1]["content"]
    retry = llm.calls[4][-1]["content"]
    assert "一般过去时" in retry and "[S3]" in retry                                 # 带着具体问题当场返修
    r = res.review
    assert not [f for f in r.initial if f.severity == "must_fix"] and r.stop_reason == "passed"
    st = {x.slot_id: x.vstatus for x in res.paper.questions}
    assert st == {"S1": VStatus.AGREED, "S2": VStatus.AGREED, "S3": VStatus.AGREED, "S4": VStatus.UNVERIFIED}
    assert "【3年级英语规则】" in llm.calls[2][-1]["content"]                     # 命题员事先拿到年级规则


# ---------- 年级适切性：语法分级 + 阅读难度 ----------

from exam_agent.tools.english import check_question as check_en, grade_rules_text, level_vocab, passage_of


def codes(issues):
    return [(i.severity, i.code) for i in issues]


def test_official_vocab_lists_loaded():
    assert len(level_vocab(2)) > 500 and len(level_vocab(3)) > 1600
    assert "panda" in level_vocab(2) and "ability" not in level_vocab(2) and "ability" in level_vocab(3)


def test_grammar_by_curriculum_level():
    # 三年级（一级）：过去时、将来时、比较级都超出课标级别
    assert codes(check_en("选择题", "He is taller than me.", [], "", 3, "三上")) == [("must_fix", "grammar_level")]
    assert codes(check_en("选择题", "I will go to the zoo.", [], "", 4, "四下")) == [("must_fix", "grammar_level")]
    # 五年级（二级）：过去时在课标范围内，但教材六下才学 → 只是建议
    assert codes(check_en("选择题", "I went to the park yesterday.", [], "", 5, "五上")) == [("suggest", "grammar_textbook")]
    # 六年级全年：过去时已学，不提示
    assert check_en("选择题", "I went to the park yesterday.", [], "", 6, "六下") == []
    # 七年级（三级）：现在完成时在课标内，但教材八下 → 建议
    assert codes(check_en("选择题", "I have finished my homework.", [], "", 7, "七下")) == [("suggest", "grammar_textbook")]
    # 不误判：morning 不是进行时，I have a dog 不是完成时，excited 不是过去时
    assert check_en("选择题", "It's morning. I have a dog. I am excited.", [], "", 3, "三上") == []


def test_reading_length_range_by_grade():
    """老师给定：3～4 年级 50～200 词，5～6 年级 100～300 词，七年级 150～400 词。单词难易不判定。"""
    words = lambda n: "Read the passage.\n" + " ".join(["This is my cat."] * (n // 4))
    assert check_en("阅读理解", words(120), [], "", 3, "三上") == []
    assert codes(check_en("阅读理解", words(40), [], "", 3, "三上")) == [("must_fix", "reading_length")]   # 太短
    assert codes(check_en("阅读理解", words(220), [], "", 4, "四下")) == [("must_fix", "reading_length")]  # 太长
    assert codes(check_en("阅读理解", words(80), [], "", 5, "五上")) == [("must_fix", "reading_length")]
    assert check_en("阅读理解", words(320), [], "", 7, "七上") == []
    assert "50～200 词" in grade_rules_text(3)


def test_reading_count_limited_by_grade():
    from exam_agent.invariants import check_blueprint
    slots = [{"slot_id": f"S{i}", "position": i, "qtype": "阅读理解", "score": 10, "knowledge_point": "x",
              "difficulty": "基础", "cognitive_level": "理解", "intent": "x"} for i in range(1, 4)]
    bp = Blueprint.model_validate({"slots": slots})
    assert any(v.code == "reading_count" for v in check_blueprint(en_req(question_count=3, total_score=30,
                                                                          required_types=[]), bp))


def test_true_false_reading_answers():
    assert reading_answers("1. T 2. F 3. √ 4. ×") == {1: "T", 2: "F", 3: "T", 4: "F"}


def test_english_build_trims_vocabulary_instead_of_retrying():
    """kb build 六下 实测：模型给了 41 个词，“应为 5～40 个”连续 3 次没改对。能由代码处理的就不让模型重写。"""
    units = ["Unit 4 Then and now"]
    kb = KnowledgeBase("english", [], EN_CLAUSES)
    many = draft(units, vocabulary=[f"word{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(45)],
                 signals=["then"], standard_refs=["E2-VOC"])
    llm = ScriptedLLM([j(many)])
    cards = kbuild.build_book(llm, "六下", kb, {"edition": "PEP", "books": {"六下": units}}, subject="english")
    assert len(llm.calls) == 1 and len(cards[0].vocabulary) == kbuild.MAX_VOCAB and cards[0].signals == []


# ---------- p_ba423ede 暴露的问题 ----------

def test_feasible_options_respect_reading_limit():
    """d025：规划连续 3 次给出 4 篇阅读而失败。程序算好的题型配比里先排除超过篇数上限的方案。"""
    from exam_agent.scoring import feasible_count_options
    opts = feasible_count_options(20, 100, [QType.CHOICE, QType.FILL, QType.READING], limit=50,
                                  max_counts={QType.READING: 2})
    assert opts and all(o[QType.READING] <= 2 for o in opts)


def test_disguised_reading_in_choice_is_sent_back():
    """d026：限制篇数后，模型把 6 篇短文放进了选择题。规划阶段按知识点 / 意图识别，题目阶段按题干词数识别。"""
    from exam_agent.agents.planner import plan_blueprint
    slot = lambda i, t, kp, sc: {"slot_id": f"S{i}", "position": i, "qtype": t, "score": sc, "knowledge_point": kp,
                                 "difficulty": "基础", "cognitive_level": "理解", "intent": "x"}
    rest = [slot(2, "选择题", "问候语", 5), slot(3, "填空题", "颜色单词", 5), slot(4, "填空题", "数字单词", 5)]
    bad = {"slots": [slot(1, "选择题", "阅读短文：动物", 5), *rest], "rationale": ""}
    good = {"slots": [slot(1, "选择题", "动物单词", 5), *rest], "rationale": ""}
    llm = ScriptedLLM([j(bad), j(good)])
    plan_blueprint(llm, en_req(question_count=4, total_score=20, required_types=["选择题", "填空题"]))
    assert "实际是阅读短文" in llm.calls[1][-1]["content"]


def test_instruction_line_is_not_part_of_passage():
    stem = ("Read the passage and match the information.\n\nHello! I'm Mike. I have a dog.\n"
            "1. What colour is the bird?\nA. Green and yellow\nB. Black")
    assert passage_of(stem) == "Hello! I'm Mike. I have a dog."


# ---------- p_49cad5b9 暴露的问题 ----------

def test_review_policy_is_enforced_in_code():
    """d029：教学审核因 want “不在本单元词汇表”判超纲；还有一条意见写着“不构成明显错误”却标 must_fix。"""
    from exam_agent.agents.reviewer import ReviewIssue
    from exam_agent.review import pedagogy_severity
    word = ReviewIssue(slot_id="S8", severity="must_fix", category="超纲", message="want 不在本单元词汇表中，属于超纲词汇")
    grammar = ReviewIssue(slot_id="S3", severity="must_fix", category="超纲", message="使用了一般过去时的语法，超出范围")
    cleared = ReviewIssue(slot_id="S9", severity="must_fix", category="蓝图符合度", message="经复核，本题与蓝图相符，不构成明显错误。")
    assert pedagogy_severity(word, en_req()) == "suggest"
    assert pedagogy_severity(grammar, en_req()) == "suggest"     # 英语超纲由代码按课标判定（d033）
    unit = ReviewIssue(slot_id="S8", severity="must_fix", category="超纲", message="使用了 has，超出 Unit 2 句型范围")
    assert pedagogy_severity(unit, en_req()) == "suggest"
    other = ReviewIssue(slot_id="S5", severity="must_fix", category="表述", message="题干有两个设问，学生会误解")
    assert pedagogy_severity(other, en_req()) == "must_fix"
    assert pedagogy_severity(cleared, en_req()) == "suggest"
    math = en_req(subject="数学", required_types=[])
    assert pedagogy_severity(word, math) == "must_fix"          # 只对英语放宽单词


def test_answer_leak_from_chinese_translation():
    q = mk(QType.CHOICE, "The apple is ____. (这个苹果是红色的。)", "A", ["red", "blue", "green", "yellow"])
    from exam_agent.review import check_question as review_check
    fs, v = review_check(q, BlindSolution(slot_id="S1", choice="A", leak="中文翻译直接说了“红色的”"), ExamRequest(
        **{**en_req().model_dump(), "required_types": []}))
    assert [f.code for f in fs] == ["answer_leak"] and v.status == VStatus.PROBLEM


def test_too_many_slots_on_one_point():
    """d030：20 题里 5 道都是“字母 X-X 书写”。"""
    from exam_agent.agents.planner import check_concentration
    slots = [{"slot_id": f"S{i}", "position": i, "qtype": "填空题", "score": 5, "difficulty": "基础",
              "cognitive_level": "掌握", "intent": "x",
              "knowledge_point": f"字母 {'ABCDE'[i - 1]}a-{'ABCDE'[i - 1]}d 书写" if i <= 5 else "甲乙丙丁戊己庚辛壬癸子丑寅卯辰"[i - 6] + "单元"}
             for i in range(1, 21)]
    assert "重复出题太多" in check_concentration(Blueprint.model_validate({"slots": slots}))[0]
    slots[4]["knowledge_point"] = "颜色单词"
    assert check_concentration(Blueprint.model_validate({"slots": slots})) == []


def test_reading_new_word_ratio_uses_official_lists():
    """老师定义：生词 = 课标二级 / 三级词汇表之外的词；例如 100 词里 40 个生词必须打回。"""
    from exam_agent.tools.english import NEW_WORD_MAX
    easy = "Look at my cat. It is black and white. I like my cat. " * 6                       # 全是二级词
    assert check_en("阅读理解", easy, [], "", 3, "三上") == []
    hard = ("Scientists discovered fascinating creatures in the jungle. " * 3 + "I like my cat. It is black. " * 5)
    found = check_en("阅读理解", hard, [], "", 3, "三上")
    assert [i.code for i in found] == ["reading_new_words"] and "scientists" in found[0].message
    some = easy + " Amy has a kite and a puppy."                                               # 个别生词：允许
    assert check_en("阅读理解", some, [], "", 3, "三上") == [] and NEW_WORD_MAX == 0.15
    # 三级词（ability）对小学是生词，对初中不是
    seventh = "He has the ability to swim. " * 40
    assert not any(i.code == "reading_new_words" for i in check_en("阅读理解", seventh, [], "", 7, "七上"))



def test_numbered_fill_answers_compare_by_words():
    q = mk(QType.FILL, "1. 面包：b____d 2. 牛奶：m____k", "1. bread 2. milk")
    assert route(q, BlindSolution(slot_id="S1", final_answer="bread; milk")).verification.status == VStatus.AGREED


def test_writing_slot_is_not_disguised_reading():
    """e2e 第二轮：书面表达的知识点写着“短文”，被误判为伪装的阅读，规划三次失败。"""
    from exam_agent.agents.planner import check_disguised_reading

    bp = Blueprint.model_validate({"slots": [
        {"slot_id": "S1", "position": 1, "qtype": "书面表达", "score": 15, "knowledge_point": "书面表达：写一篇短文介绍自己",
         "difficulty": "中等", "cognitive_level": "运用", "intent": "写短文"},
        {"slot_id": "S2", "position": 2, "qtype": "选择题", "score": 5, "knowledge_point": "阅读短文选答案",
         "difficulty": "基础", "cognitive_level": "理解", "intent": "x"}]})
    out = check_disguised_reading(en_req(), bp)
    assert out and "第2题" in out[0] and "第1题" not in out[0]
