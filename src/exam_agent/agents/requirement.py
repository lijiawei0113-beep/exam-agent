"""需求理解：把老师的自然语言转成结构化 ExamRequest，并显式列出系统做的假设。

出卷范围严格按老师原话确定，不按日期或经验推断：
  “三年级”→ 上下两册；“三年级上册”→ 上册；“三年级上册第二单元”→ 上册 + 该单元。
册别由代码对照老师原话核对（模型自作主张写成“上册”会被打回），年级写法和学段由代码统一。
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from ..domain import ExamRequest, allowed_types, type_list_text
from ..llm import LLMClient
from .base import Tracer, structured_call


class RequestDraft(BaseModel):
    request: ExamRequest
    assumptions: list[str] = Field(default_factory=list, description="老师没说、系统自行推断的内容")

    @model_validator(mode="before")
    @classmethod
    def _fill_missing(cls, data):
        """老师没说题数、满分时，模型常输出 null 或 0，校验直接失败、三次重试后整卷报错（e2e 评测 5 次）。
        这里先填占位值，真正的默认值由 apply_defaults 按老师原话决定。"""
        if isinstance(data, dict) and isinstance(data.get("request"), dict):
            r = dict(data["request"])
            for key, default in (("question_count", DEFAULT_COUNT), ("total_score", DEFAULT_TOTAL)):
                v = r.get(key)
                if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                    try:
                        v = int(str(v).strip().rstrip("分题道")) if v not in (None, "") else 0
                    except ValueError:
                        v = 0
                    r[key] = v if v > 0 else default
            data = {**data, "request": r}
        return data


SYSTEM = """你是中小学命题助手中的“需求分析员”。把老师的出卷要求整理成 json。
规则：
1. 老师明确说的内容必须如实保留，不得改动（题数、满分、题型等）。
2. 出卷范围严格按老师原话：
   - grade 写成“一年级”…“九年级”；老师说“初一”就是七年级。
   - term：老师说了“上册/上学期”填“上”，说了“下册/下学期”填“下”，没说册别一律填“全年”（表示上下两册），
     不要根据日期或经验推断册别。
   - topic：只在老师点名了单元或知识点时填写（如“一元一次方程”“第二单元”“两位数乘两位数”），否则留空。
   老师没说年级、只说了知识点时，可以按人教版推断年级，并写进 assumptions；教材版本默认人教版，也写进 assumptions。
3. required_types 只填老师明确提到的题型，取值只能是该学科的可用题型（""" + type_list_text() + """）。
4. 可被程序检验的持续性要求放进 constraints：如“答案都是整数”用 kind=integer_answer，
   scope_types 写它作用的题型（作用于全卷则为空列表）；其他要求用 kind=free_text。
5. 只输出 json，格式示例：
{"request": {"stage": "初中", "grade": "七年级", "subject": "数学", "term": "全年", "topic": "一元一次方程",
  "question_count": 10, "total_score": 100, "required_types": ["选择题", "填空题", "计算题"],
  "difficulty_note": "", "constraints": [], "extra_requirements": []},
 "assumptions": ["老师未指定年级，按人教版推断为七年级上册"]}"""


CN_GRADES = "一二三四五六七八九"
_ALIAS = {"初一": "七年级", "初二": "八年级", "初三": "九年级"}


def normalize_grade(grade: str) -> str:
    g = grade.strip()
    for k, v in _ALIAS.items():
        if k in g:
            return v
    m = re.search(r"([1-9])\s*年级", g)
    if m:
        return f"{CN_GRADES[int(m.group(1)) - 1]}年级"
    return g


def stage_of(grade: str) -> str | None:
    m = re.match(r"([一二三四五六七八九])年级", grade)
    if not m:
        return None
    return "小学" if CN_GRADES.index(m.group(1)) < 6 else "初中"


def term_in_text(text: str) -> str:
    up = re.search(r"上册|上学期|上半学期", text)
    down = re.search(r"下册|下学期|下半学期", text)
    if up and not down:
        return "上"
    if down and not up:
        return "下"
    return "全年"


def normalize(d: RequestDraft) -> RequestDraft:
    """确定性整理：年级写法、学段由代码决定，不依赖模型。"""
    r = d.request
    grade = normalize_grade(r.grade)
    stage = stage_of(grade) or r.stage
    return d.model_copy(update={"request": r.model_copy(update={"grade": grade, "stage": stage})})


def _validate(d: RequestDraft, teacher_text: str = "") -> list[str]:
    r = d.request
    if r.subject not in SUPPORTED:
        return []        # 不支持的学科不让模型反复修题型，解析完直接告诉老师
    problems = []
    bad_types = [t.value for t in r.required_types if t not in allowed_types(r.subject)]
    if bad_types:
        problems.append(f"{r.subject}没有这些题型：{bad_types}；可用题型：{'、'.join(t.value for t in allowed_types(r.subject))}")
    expected = term_in_text(teacher_text)
    if r.term != expected:
        problems.append(f"term 应为“{expected}”：老师原话里" +
                        ("没有提到上册或下册，表示上下两册都考，不要自行推断" if expected == "全年"
                         else f"说的是{expected}册"))
    if len(set(r.required_types)) > r.question_count:
        problems.append("要求的题型种类多于题目数量")
    if r.total_score < r.question_count:
        problems.append("满分小于题数，无法保证每题至少 1 分")
    return problems


SUPPORTED = {"数学": (1, 7), "英语": (3, 7)}   # 学科 → 支持的年级范围
DEFAULT_COUNT, DEFAULT_TOTAL = 20, 100
_NUM = r"[0-9一二三四五六七八九十百两]+"


def apply_defaults(d: RequestDraft, teacher_text: str) -> RequestDraft:
    """老师没说题数、满分时，由代码给出固定默认值（20 题、100 分），不让模型随意决定。
    （d037：“七年级语文试卷”被模型解析成 1 题、满分 1 分。）"""
    r, notes = d.request, []
    update = {}
    if not re.search(_NUM + r"\s*(道|题|个小题|篇)", teacher_text):
        update["question_count"] = DEFAULT_COUNT
        notes.append(f"老师未指定题数，按 {DEFAULT_COUNT} 题出卷")
    if not re.search(_NUM + r"\s*分|(满分|总分)\s*(是|为)?\s*" + _NUM, teacher_text):
        update["total_score"] = DEFAULT_TOTAL
        notes.append(f"老师未指定满分，按 {DEFAULT_TOTAL} 分出卷")
    if r.required_types and re.search(r"只(要|出|考|有|用|做)|仅|全(部)?(是|为|出)|都(出|是)(选择|填空|计算|应用|阅读|书面)",
                                      teacher_text):
        update["only_types"] = True
    if not update:
        return d
    kept = [a for a in d.assumptions if not re.search(r"题数|满分|总分|\d+\s*题", a)]
    return d.model_copy(update={"request": r.model_copy(update=update), "assumptions": kept + notes})


def check_supported(req) -> None:
    """系统目前只支持数学一至七年级、英语三至七年级；其他请求直接说明，不勉强出卷。"""
    from .base import UnsupportedRequest

    scope = "；".join(f"{k}{CN_GRADES[a - 1]}至{CN_GRADES[b - 1]}年级" for k, (a, b) in SUPPORTED.items())
    if req.subject not in SUPPORTED:
        raise UnsupportedRequest(f"目前支持{scope}，暂不支持{req.subject}")
    g = CN_GRADES.index(req.grade[:1]) + 1 if req.grade[:1] in CN_GRADES else None
    lo, hi = SUPPORTED[req.subject]
    if g is None or not lo <= g <= hi:
        extra = "（英语一般从三年级开设，且一二年级不进行纸笔考试）" if req.subject == "英语" and g and g < 3 else ""
        raise UnsupportedRequest(f"{req.subject}目前支持{CN_GRADES[lo - 1]}至{CN_GRADES[hi - 1]}年级，"
                                 f"暂不支持{req.grade}{extra}")


def _feasible(req, total: int, types) -> list:
    from ..scoring import feasible_count_options
    from .planner import max_counts

    return feasible_count_options(req.question_count, total, list(types), max_counts=max_counts(req))


def resolve_feasibility(d: RequestDraft, teacher_text: str) -> RequestDraft:
    """规划前先用代码检查“题数、满分、题型”能不能凑成一份卷，凑不成就在规划前处理好，并告诉老师怎么处理的。
    d040：“七年级数学试卷，5题”按默认满分 100、默认题型（选择、填空、计算），选择填空每题最多 6 分，
    程序算不出可行配比，规划员自作主张出了 7 题，三次被打回后整卷失败。
    规则（老师确定）：每题分值按 scoring.SCORE_RANGE（选择、填空 1～6 分，计算、应用 10～20 分），不放宽；
    老师说的题数、满分不改，没说满分就是 100。老师没指定题型、默认题型凑不成时，减少题型：
    先在默认题型里减（5 题 → 5 道计算题，每题 20 分），不行再用本学科其他题型。
    仍凑不成：没说满分就降到能凑成的整十分；说了满分就请老师调整。"""
    from itertools import combinations

    from ..domain import DEFAULT_TYPES, QType
    from ..scoring import SCORE_RANGE
    from .base import UnsupportedRequest

    r = d.request
    if r.subject not in SUPPORTED:
        return d
    from ..domain import plan_types

    default = plan_types(r)
    must = set(r.required_types)

    def pick(total: int):
        """能凑成 total 的题型组合：先用全部可用题型；不行就去掉一些，但老师要求的题型必须保留。
        老师说“只要…”时不能换题型。"""
        if _feasible(r, total, default):
            return default
        if r.only_types and r.required_types:
            return None
        pool = default + [t for t in allowed_types(r.subject) if t not in default]
        for group, sizes in ((default, range(len(default) - 1, 0, -1)), (pool, range(len(pool), 0, -1))):
            for k in sizes:
                for sub in combinations(group, k):
                    if must <= set(sub) and _feasible(r, total, sub):
                        return list(sub)
        return None

    total_given = bool(re.search(_NUM + r"\s*分|(满分|总分)\s*(是|为)?\s*" + _NUM, teacher_text))
    for total in ([r.total_score] if total_given else range(100, 0, -10)):
        types = pick(total)
        if types is None:
            continue
        if types is default and total == r.total_score:
            return d                                   # 绝大多数情况：默认题型就能凑成
        notes = [a for a in d.assumptions if total == r.total_score or "满分" not in a]
        if total != r.total_score:
            notes.append(f"老师未指定满分；{r.question_count} 道题凑不成 100 分，按 {total} 分出卷")
        if types is not default:
            notes.append(f"{r.question_count} 道题凑 {total} 分，题型安排为{'、'.join(QType(t).value for t in types)}")
        update = {"total_score": total}
        if types is not default:
            update["required_types"] = [QType(t) for t in types]
            update["only_types"] = True
        return d.model_copy(update={"request": r.model_copy(update=update), "assumptions": notes})
    ranges = "、".join(f"{QType(t).value} {SCORE_RANGE[QType(t)].start}～{SCORE_RANGE[QType(t)].stop - 1} 分"
                       for t in default)
    raise UnsupportedRequest(f"{r.question_count} 道题凑不成 {r.total_score} 分（每题分值：{ranges}），"
                             "请调整题数、满分或题型")


def parse_request(llm: LLMClient, teacher_text: str, tracer: Tracer | None = None) -> RequestDraft:
    draft = structured_call(
        llm, step="requirement", system=SYSTEM, user=f"老师的要求：{teacher_text}",
        schema=RequestDraft, validate=lambda d: _validate(apply_defaults(d, teacher_text), teacher_text),
        tracer=tracer,
    ).value
    draft = normalize(apply_defaults(draft, teacher_text))
    check_supported(draft.request)
    return resolve_feasibility(draft, teacher_text)
