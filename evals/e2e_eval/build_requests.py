"""生成端到端评测的 100 条出卷请求（evals/e2e_requests.jsonl）。固定随机种子，结果可复现。
运行：python evals/e2e_eval/build_requests.py

组成：数学一至七年级 60 条、英语三至七年级 25 条、边界情况 15 条（不支持的学科 / 年级、凑不成的配分、
口语化说法、需要图的单元）。说法模仿老师的真实表达：年级写法、册别、单元、题数、满分、题型要求各有变化。
expect：paper 应当出卷成功；reject 应当被拒绝并说明原因。
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 开发集（e2e_requests.jsonl，种子 20261025）用来发现问题、修复；留出集（e2e_requests_holdout.jsonl，
# 另一个种子 + 另一组边界说法）只在最后跑一次报告结果，不据此修改代码。
# 留出集 2（e2e_requests_holdout2.jsonl）：留出集 1 暴露问题并修复后，用来再验证一次，同样只跑一次。
WHICH = sys.argv[1] if len(sys.argv) > 1 else "dev"
HOLDOUT = WHICH in ("holdout", "holdout2")
rng = random.Random({"dev": 20261025, "holdout": 20261111, "holdout2": 20261212}[WHICH])
CN = "一二三四五六七"
MATH_TOC = json.loads((ROOT / "kb/math/sources/toc.json").read_text(encoding="utf-8"))["books"]
EN_TOC = json.loads((ROOT / "kb/english/sources/toc.json").read_text(encoding="utf-8"))["books"]
MATH7 = ["有理数", "有理数的运算", "代数式", "整式的加减", "一元一次方程"]
NEED_PICTURE = ("图形", "观察物体", "钟表", "时间", "位置", "运动", "角", "统计", "数据", "广角", "准备课", "方向", "平行四边形",
                "三角形", "圆", "长方体", "正方体", "扇形", "条形", "折线", "比多少")


def grade_name(g: int) -> str:
    if g == 7:
        return rng.choice(["七年级", "7年级", "初一"])
    return rng.choice([f"{CN[g - 1]}年级", f"{g}年级", f"小学{CN[g - 1]}年级"])


def math_units(g: int, term: str) -> list[tuple[int, str]]:
    if g == 7:
        return list(enumerate(MATH7, 1))
    units = MATH_TOC[f"{CN[g - 1]}{term}"]
    return [(i, u) for i, u in enumerate(units, 1) if not any(w in u for w in NEED_PICTURE)]


def math_request(g: int) -> str:
    G = grade_name(g)
    term = rng.choice(["上", "下"]) if g < 7 else "上"
    T = rng.choice([f"{term}册", f"{term}册", ""])
    n = rng.choice([5, 8, 10, 12] if g <= 2 else [8, 10, 12, 15, 20, 20, 25])
    s = rng.choice([100, 100, 100, 50, 120])
    units = math_units(g, term)
    k, unit = rng.choice(units) if units else (1, "")
    chap = "章" if g == 7 else "单元"
    templates = [
        f"出一份{G}{T}数学试卷，{n}道题，满分{s}分",
        f"{G}{term}册数学单元测试，考“{unit}”，{n}题" if unit else f"{G}{T}数学测试，{n}题",
        f"给{G}学生出一套数学练习，{n}道，选择、填空、计算都要有",
        f"{G}{T}数学期中卷，{n}道题，总分{s}，要有应用题" if g >= 3 else f"{G}{T}数学练习，{n}道题",
        f"帮我出{G}{term}册第{k}{chap}的数学小测，{n}道题",
        f"{G}数学，{n}题，难度中等，计算题多一点",
    ]
    return rng.choice(templates)


def english_request(g: int) -> str:
    G = grade_name(g)
    term = rng.choice(["上", "下"])
    T = rng.choice([f"{term}册", ""])
    n = rng.choice([8, 10, 12] if g == 3 else [10, 12, 15, 20])
    s = rng.choice([100, 100, 60])
    k = rng.randint(1, len(EN_TOC[f"{CN[g - 1]}{term}"]))
    templates = [
        f"出一份{G}{T}英语试卷，{n}道题，满分{s}分",
        f"{G}{term}册英语 Unit {k} 测试，{n}题",
        f"{G}英语卷，{n}道题，要有阅读理解和书面表达",
        f"{G}{T}英语练习，选择和填空为主，{n}题",
    ]
    return rng.choice(templates)


rows = []
for g, cnt in zip(range(1, 8), [6, 8, 10, 9, 9, 9, 9]):
    rows += [{"group": "数学", "grade": g, "text": math_request(g), "expect": "paper"} for _ in range(cnt)]
for g in range(3, 8):
    rows += [{"group": "英语", "grade": g, "text": english_request(g), "expect": "paper"} for _ in range(5)]
EDGE = [
    ("出一份七年级语文试卷", "reject"), ("八年级物理测试，10题", "reject"), ("八年级数学一次函数单元测试，20题", "reject"),
    ("二年级英语练习，10道题", "reject"), ("高一数学集合测试卷", "reject"),
    ("三年级数学，3道选择题，满分100分", "reject"),
    ("七年级数学，5道题", "paper"), ("四年级数学试卷", "paper"), ("出张卷子，三年级的，数学，题少点，10道左右", "paper"),
    ("五年级英语，只要阅读理解，2篇", "paper"), ("六年级数学 30 道题 100 分", "paper"),
    ("七年级上册数学第四章整式的加减，15题，计算题多一点", "paper"), ("三年级上册英语 Unit 3，12 题，满分 60", "paper"),
    ("二年级上册数学认识时间练习，8道题", "paper"), ("一年级下册数学认识图形练习，6题", "paper"),
]
EDGE_HOLDOUT = [
    ("九年级化学单元测试", "reject"), ("出一份小学三年级音乐练习", "reject"), ("八年级英语期末卷，30题", "reject"),
    ("一年级英语字母练习", "reject"), ("初二数学勾股定理测试，15题", "reject"),
    ("五年级数学，2道选择题，满分100分", "reject"),
    ("初一数学，6道题", "paper"), ("六年级英语试卷", "paper"), ("来一套二年级的数学题，不用太难，十几道就行", "paper"),
    ("四年级英语，只出选择题，10道", "paper"), ("三年级数学 40 道题 100 分", "paper"),
    ("七年级上册数学第二章有理数的运算，12题，填空题多一点", "paper"), ("七年级下册英语 Unit 2，15 题，满分 80", "paper"),
    ("一年级上册数学认识钟表，6道题", "paper"), ("三年级下册数学位置与方向练习，8题", "paper"),
]
EDGE_HOLDOUT2 = [
    ("高二物理电场测试卷", "reject"), ("出一份四年级科学试卷，15题", "reject"), ("九年级数学二次函数，20题", "reject"),
    ("幼儿园大班数学练习", "reject"), ("一年级英语，认识颜色，8题", "reject"),
    ("六年级数学，4道填空题，总分100分", "reject"),
    ("初一英语，8道题", "paper"), ("二年级数学试卷", "paper"), ("给五年级出点数学题，二十道左右，别太难", "paper"),
    ("三年级英语，只要选择题，12道", "paper"), ("四年级数学 35 题 100 分", "paper"),
    ("七年级上册数学第三章代数式，10题，应用题多一点", "paper"), ("六年级下册英语 Unit 1，12 题，满分 70", "paper"),
    ("五年级上册数学可能性，8道题", "paper"), ("二年级下册数学图形的运动练习，6题", "paper"),
]
edge = {"dev": EDGE, "holdout": EDGE_HOLDOUT, "holdout2": EDGE_HOLDOUT2}[WHICH]
rows += [{"group": "边界", "grade": 0, "text": t, "expect": e} for t, e in edge]
assert len(rows) == 100
out = ROOT / "evals" / {"dev": "e2e_requests.jsonl", "holdout": "e2e_requests_holdout.jsonl",
                        "holdout2": "e2e_requests_holdout2.jsonl"}[WHICH]
with open(out, "w", encoding="utf-8") as f:
    for i, r in enumerate(rows, 1):
        prefix = {"dev": "r", "holdout": "h", "holdout2": "g"}[WHICH]
        f.write(json.dumps({"id": f"{prefix}{i:03d}", **r}, ensure_ascii=False) + "\n")
print(out, len(rows))
