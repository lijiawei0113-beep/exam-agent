"""生成修改 / 讨论评测用的两份试卷及其版本历史（papers.json）。运行：python evals/edit_eval/build_papers.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exam_agent.domain import (Blueprint, BlueprintSlot, Difficulty, ExamRequest, PaperSnapshot, QType, Question,  # noqa: E402
                               Verification, VStatus)

AG = Verification(status=VStatus.AGREED, method="盲解独立作答")
TOOL = Verification(status=VStatus.TOOL, method="题干直接提取算式 → 精确计算")
UNV = Verification(status=VStatus.UNVERIFIED, method="主观题：范文与评分要点由老师把握")


def paper(pid, title, req, items):
    slots, qs = [], []
    for i, (t, sc, kp, diff, stem, opts, ans, ana, v) in enumerate(items, 1):
        sid = f"S{i}"
        slots.append(BlueprintSlot(slot_id=sid, position=i, qtype=t, score=sc, knowledge_point=kp, difficulty=diff,
                                   cognitive_level="运用" if t == QType.APPLICATION else "掌握", intent=kp))
        qs.append(Question(slot_id=sid, qtype=t, score=sc, knowledge_point=kp, difficulty=diff, stem=stem,
                           options=opts, answer=ans, analysis=ana, verification=v))
    return PaperSnapshot(paper_id=pid, title=title, request=req, blueprint=Blueprint(slots=slots), questions=qs)


C, F, K, A, R, W = QType.CHOICE, QType.FILL, QType.CALC, QType.APPLICATION, QType.READING, QType.WRITING
MATH_REQ = ExamRequest(stage="小学", grade="三年级", subject="数学", term="上", topic="", question_count=10,
                       total_score=100, required_types=[C, F, K, A])
M = [
    (C, 5, "万以内的加法", "基础", "下面哪个算式的得数最接近 600？（ ）", ["298 + 305", "412 + 99", "189 + 300", "450 + 201"],
     "A", "298+305=603，412+99=511，189+300=489，450+201=651，603 最接近 600。", AG),
    (C, 5, "质量单位", "基础", "1 千克棉花和 1000 克铁比较，（ ）", ["棉花重", "铁重", "一样重", "无法比较"], "C",
     "1 千克 = 1000 克，两者一样重。", AG),
    (C, 5, "长方形的周长", "中等", "一个长方形的长是 8 厘米、宽是 5 厘米，它的周长是（ ）厘米。", ["13", "26", "40", "18"], "B",
     "(8+5)×2=26（厘米）。", AG),
    (F, 5, "时间单位换算", "基础", "3 时 = ____ 分", [], "180", "1 时 = 60 分，3×60=180。", AG),
    (F, 5, "长度单位比较", "基础", "在 ○ 里填上 >、< 或 =：5 千米 ○ 4900 米", [], ">", "5 千米 = 5000 米，5000 > 4900。", AG),
    (K, 15, "三位数加三位数", "基础", "计算：425 + 389 =", [], "814", "个位 5+9=14 写 4 进 1……", TOOL),
    (K, 15, "三位数减三位数", "中等", "计算：702 - 365 =", [], "337", "连续退位减法。", TOOL),
    (K, 15, "多位数乘一位数", "基础", "计算：48 × 6 =", [], "288", "40×6=240，8×6=48，240+48=288。", TOOL),
    (A, 15, "两位数减两位数解决问题", "中等",
     "学校组织三年级同学去秋游，一共有 4 个班，每个班有 38 名同学。老师还准备了一些水果，其中苹果有 96 个、梨有 58 个。"
     "苹果比梨多多少个？", [], "38 个", "96 - 58 = 38（个）。班级人数是多余条件。",
     Verification(status=VStatus.TOOL, method="盲解列算式 → 精确计算", detail="96-58 ⇒ 38")),
    (A, 15, "三位数加法解决问题", "中等",
     "小明家到学校的路程是 650 米。他每天上学走一趟、放学走一趟，中午不回家。小明一天上学、放学一共要走多少米？",
     [], "1300 米", "650 + 650 = 1300（米）。", Verification(status=VStatus.TOOL, method="盲解列算式 → 精确计算",
                                                             detail="650+650 ⇒ 1300")),
]
m1 = paper("p_evalmath", "三年级上册数学试卷", MATH_REQ, M)
m2 = m1.model_copy(deep=True)
m2.questions[2] = m2.questions[2].model_copy(update={
    "stem": "一个正方形的边长是 6 厘米，它的周长是（ ）厘米。", "options": ["12", "24", "36", "18"], "answer": "B",
    "analysis": "6×4=24（厘米）。", "knowledge_point": "正方形的周长", "difficulty": Difficulty.BASIC})
m2.blueprint.slots[2].knowledge_point, m2.blueprint.slots[2].difficulty = "正方形的周长", Difficulty.BASIC
m3 = m2.model_copy(deep=True)
m3.questions[5] = m3.questions[5].model_copy(update={"stem": "计算：356 + 278 =", "answer": "634",
                                                      "analysis": "个位 6+8=14 写 4 进 1……"})

EN_REQ = ExamRequest(stage="小学", grade="五年级", subject="英语", term="上", topic="", question_count=13,
                     total_score=100, required_types=[C, F, R, W])
READ1 = ("Read the passage and choose the correct answers.\n"
         "Hello, I am Wu Binbin. I am in Class 1, Grade 5. I have a new friend. His name is Zhang Peng. He is tall and "
         "strong. He is good at football. He often plays football with me after school. Zhang Peng is very kind. He helps "
         "me with my maths homework. On Saturday we often go to the park near our school. We fly kites there. Sometimes we "
         "read books under a big tree. My mother likes Zhang Peng too. She says he is polite and hard-working. I am happy "
         "to have a good friend like him.\n"
         "1. What is Zhang Peng good at? A. Basketball. B. Football. C. Drawing.\n"
         "2. What do they often do on Saturday? A. Watch TV. B. Cook dinner. C. Go to the park.\n"
         "3. What does Zhang Peng help Wu Binbin with? A. English. B. Maths. C. Music.")
READ2 = ("Read the passage and choose the correct answers.\n"
         "This is my robot. Its name is Robin. It is short but very clever. It can speak Chinese and English. Every "
         "morning Robin helps my grandpa in the kitchen. It can make a salad and some tea. Robin is strict, too. I must do "
         "my homework before I watch TV. In the afternoon Robin cleans the room. On Wednesdays it waters the flowers in "
         "the garden. My grandpa likes Robin very much. He says Robin is helpful. I love my robot. It is my good friend.\n"
         "1. What can Robin speak? A. Chinese and English. B. Only English. C. Only Chinese.\n"
         "2. When does Robin water the flowers? A. On Mondays. B. On Wednesdays. C. On Fridays.\n"
         "3. What does grandpa think of Robin? A. Funny. B. Helpful. C. Lazy.")
E = [
    (C, 5, "What would you like to eat?", "基础", "—What would you like to eat? —____",
     ["I like music.", "I'd like some noodles.", "It's Monday."], "B", "问想吃什么，用 I'd like... 回答。", AG),
    (C, 5, "描述人物性格", "基础", "My sister is very ____. She always helps others.", ["lazy", "kind", "short"], "B",
     "乐于助人说明她很善良。", AG),
    (C, 5, "What day is it today?", "基础", "—What day is it today? —____", ["It's sunny.", "It's Wednesday.", "It's my book."],
     "B", "问星期几。", AG),
    (C, 5, "课程表", "基础", "—What do you have on Thursdays? —I have ____.", ["English and music", "maths and science", "tired"],
     "A", "问星期四上什么课。", AG),
    (C, 5, "There is / There are", "基础", "There ____ two bikes behind the house.", ["is", "are", "am"], "B",
     "two bikes 是复数，用 are。", AG),
    (C, 5, "方位介词", "基础", "The park is ____ the school.", ["eat", "tall", "near"], "C", "near 表示在……附近。", AG),
    (F, 5, "I can...", "基础", "I can ____ (做饭) for my mother.", [], "cook", "做饭 cook。", AG),
    (F, 5, "描述人物性格", "基础", "My English teacher is very ____ (有趣的).", [], "funny", "有趣的 funny。", AG),
    (F, 5, "Is there...?", "基础", "—Is there a river in the park? —Yes, there ____.", [], "is", "Is there 的肯定回答。", AG),
    (F, 5, "日常活动", "基础", "We often ____ (打扫) the classroom on Fridays.", [], "clean", "打扫 clean。", AG),
    (R, 15, "人物介绍阅读", "中等", READ1, [], "1. B 2. C 3. B", "依据短文第 2、5、4 句作答。",
     Verification(status=VStatus.AGREED, method="盲解逐题独立作答", detail="3 个小题全部一致")),
    (R, 15, "机器人阅读", "中等", READ2, [], "1. A 2. B 3. B", "依据短文作答。",
     Verification(status=VStatus.AGREED, method="盲解逐题独立作答", detail="3 个小题全部一致")),
    (W, 20, "书面表达：我最喜欢的一天", "中等", "以 My Favourite Day 为题，写一篇不少于 5 句话的短文，介绍你最喜欢星期几以及那天做什么。",
     [], "范文：My favourite day is Saturday. I often play football in the morning. ...", "评分要点：内容完整、语句通顺。", UNV),
]
e1 = paper("p_evalen", "五年级上册英语试卷", EN_REQ, E)

# 留出集试卷：四年级上册数学，2 个版本（v2 改过第 5 题）
HOLD_REQ = ExamRequest(stage="小学", grade="四年级", subject="数学", term="上", topic="", question_count=10,
                       total_score=100, required_types=[C, F, K, A])
H = [
    (C, 5, "角的分类", "基础", "钟面上 3 时整，时针和分针所成的角是（ ）", ["锐角", "直角", "钝角", "平角"], "B",
     "3 时整两针相差 3 个大格，即 90°，是直角。", AG),
    (C, 5, "直线、射线和线段", "基础", "过两点可以画（ ）条直线。", ["1", "2", "3", "无数"], "A", "两点确定一条直线。", AG),
    (C, 5, "积的变化规律", "中等", "一个因数不变，另一个因数乘 10，积（ ）", ["不变", "乘 10", "除以 10", "乘 100"], "B",
     "一个因数不变，积随另一个因数同样变化。", AG),
    (F, 5, "公顷和平方千米", "基础", "1 公顷 = ____ 平方米", [], "10000", "1 公顷 = 10000 平方米。",
     Verification(status=VStatus.UNVERIFIED, method="没有可用的核对工具")),
    (F, 5, "大数的认识", "基础", "由 5 个万和 3 个一组成的数是 ____", [], "50003", "5 个万是 50000，再加 3。", AG),
    (K, 15, "三位数乘两位数", "基础", "计算：125 × 16 =", [], "2000", "125×8=1000，1000×2=2000。", TOOL),
    (K, 15, "除数是两位数的除法", "中等", "计算：840 ÷ 24 =", [], "35", "试商：24×35=840。", TOOL),
    (K, 15, "三位数乘两位数", "基础", "计算：306 × 25 =", [], "7650", "306×25=7650。", TOOL),
    (A, 15, "乘除法解决问题", "中等",
     "学校图书室新买来 18 套科普书，每套 12 本。图书管理员王老师把这些书平均放在 6 个书架上，每个书架放多少本？", [],
     "36 本", "先求总本数 18×12=216（本），再平均分 216÷6=36（本）。",
     Verification(status=VStatus.TOOL, method="盲解列算式 → 精确计算", detail="18*12/6 ⇒ 36")),
    (A, 15, "速度、时间和路程", "基础",
     "一辆货车从甲地开往乙地，每小时行 65 千米，上午 8 时出发，中午 12 时到达。甲、乙两地相距多少千米？", [],
     "260 千米", "经过时间 12-8=4（小时），路程 65×4=260（千米）。",
     Verification(status=VStatus.TOOL, method="盲解列算式 → 精确计算", detail="65*(12-8) ⇒ 260")),
]
h1 = paper("p_holdmath", "四年级上册数学试卷", HOLD_REQ, H)
h2 = h1.model_copy(deep=True)
h2.questions[4] = h2.questions[4].model_copy(update={"stem": "最大的六位数是 ____", "answer": "999999",
                                                      "analysis": "六个数位都是 9。"})

out = {"p_holdmath": [
    {"snapshot": h1.model_dump(mode="json"), "operation": "create", "summary": "新建"},
    {"snapshot": h2.model_dump(mode="json"), "operation": "edit", "summary": "第5题换一道 → 最大的六位数"}],
    "p_evalmath": [
    {"snapshot": m1.model_dump(mode="json"), "operation": "create", "summary": "新建"},
    {"snapshot": m2.model_dump(mode="json"), "operation": "edit", "summary": "第3题太难 → 换成正方形周长的基础题"},
    {"snapshot": m3.model_dump(mode="json"), "operation": "edit", "summary": "第6题换一道 → 计算 356 + 278"}],
    "p_evalen": [{"snapshot": e1.model_dump(mode="json"), "operation": "create", "summary": "新建"}]}
Path(__file__).with_name("papers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")
