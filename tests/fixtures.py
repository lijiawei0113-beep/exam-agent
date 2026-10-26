"""测试用的预设 LLM 回复。"""
import json

REQUEST_REPLY = json.dumps({
    "request": {"stage": "初中", "grade": "七年级", "subject": "数学", "topic": "一元一次方程",
                "question_count": 4, "total_score": 30, "required_types": ["选择题", "填空题", "计算题"],
                "difficulty_note": "", "constraints": [], "extra_requirements": []},
    "assumptions": ["按人教版推断为七年级上册"],
}, ensure_ascii=False)


def slot(i, qtype, score, kp="移项", diff="基础", level="掌握"):
    return {"slot_id": f"S{i}", "position": i, "qtype": qtype, "score": score,
            "knowledge_point": kp, "difficulty": diff, "cognitive_level": level, "intent": "考查" + kp}


GOOD_BLUEPRINT = json.dumps({"slots": [
    slot(1, "选择题", 5, "方程的概念", level="了解"),
    slot(2, "填空题", 5, "方程的解", level="理解"),
    slot(3, "计算题", 10, "移项"),
    slot(4, "计算题", 10, "去括号", diff="中等"),
], "rationale": "基础为主"}, ensure_ascii=False)

# 总分 28，不等于 30 —— 用来测试校验反馈循环
BAD_BLUEPRINT = json.dumps({"slots": [
    slot(1, "选择题", 4), slot(2, "填空题", 4), slot(3, "计算题", 10), slot(4, "计算题", 10),
], "rationale": ""}, ensure_ascii=False)


def q(slot_id, stem, answer, options=None):
    return {"slot_id": slot_id, "stem": stem, "options": options or [], "answer": answer, "analysis": "略"}


Q1 = q("S1", "下列是一元一次方程的是（ ）", "B", ["x+y=1", "2x-3=5", "x²=4", "1/x=2"])
Q2 = q("S2", "方程 2x = 6 的解是 x = ____", "3")
Q3 = q("S3", "解方程：3x + 5 = 2x - 1", "x = -6")
Q4 = q("S4", "解方程：2(x - 1) = x + 3", "x = 5")

ALL_QUESTIONS = json.dumps({"questions": [Q1, Q2, Q3, Q4]}, ensure_ascii=False)
# 选择题只有 3 个选项且漏掉 S4
PARTLY_BAD_QUESTIONS = json.dumps({"questions": [
    {**Q1, "options": ["x+y=1", "2x-3=5", "x²=4"]}, Q2, Q3,
]}, ensure_ascii=False)
