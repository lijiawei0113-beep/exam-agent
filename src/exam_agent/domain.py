"""领域模型：试卷的“准确状态”全部由这些结构表达，并以版本快照形式存入数据库。

设计原则：
- ExamRequest  记录老师明确提出的要求（契约的来源），多轮对话会更新它。
- Blueprint    规划 Agent 产出的出卷蓝图（双向细目表），每个 slot 对应一道题。
- Question     命题 Agent 产出的题目。题型、分值、知识点从 slot 复制，不信任 LLM 自报。
- PaperSnapshot 某一版本的完整试卷状态，是唯一真相。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum

from pydantic import BaseModel, Field


class QType(str, Enum):
    CHOICE = "选择题"
    FILL = "填空题"
    CALC = "计算题"
    APPLICATION = "应用题"
    READING = "阅读理解"
    WRITING = "书面表达"


# 卷面上题型的标准排列顺序
TYPE_ORDER = [QType.CHOICE, QType.FILL, QType.CALC, QType.APPLICATION, QType.READING, QType.WRITING]

# 各学科可用的题型；没有列出的学科不限制
SUBJECT_TYPES: dict[str, list[QType]] = {
    "数学": [QType.CHOICE, QType.FILL, QType.CALC, QType.APPLICATION],
    "英语": [QType.CHOICE, QType.FILL, QType.READING, QType.WRITING],
}
DEFAULT_TYPES: dict[str, list[QType]] = {
    "数学": [QType.CHOICE, QType.FILL, QType.CALC],
    "英语": [QType.CHOICE, QType.FILL, QType.READING],
}
OBJECTIVE_TYPES = {QType.CHOICE, QType.FILL}


def allowed_types(subject: str) -> list[QType]:
    return SUBJECT_TYPES.get(subject, TYPE_ORDER)


def type_list_text() -> str:
    return "；".join(f"{k}：{'、'.join(t.value for t in v)}" for k, v in SUBJECT_TYPES.items())


class Difficulty(str, Enum):
    BASIC = "基础"
    MEDIUM = "中等"
    HARD = "提高"


class CognitiveLevel(str, Enum):
    """课标常用的认知层次，用于双向细目表。"""
    KNOW = "了解"
    UNDERSTAND = "理解"
    MASTER = "掌握"
    APPLY = "运用"


class ConstraintKind(str, Enum):
    INTEGER_ANSWER = "integer_answer"  # 可被工具校验：答案必须为整数
    FREE_TEXT = "free_text"            # 只能靠语义审核的要求


class PaperConstraint(BaseModel):
    """老师提出的、对整卷或某类题持续生效的约束。"""
    kind: ConstraintKind
    text: str = Field(description="老师原话或规范化描述")
    scope_types: list[QType] = Field(default_factory=list, description="为空表示作用于全卷")


TERMS = ("上", "下", "全年")


def plan_types(req: "ExamRequest") -> list[QType]:
    """规划时可用的题型：老师说“只要…”就只用这些；说“要有…”“…多一点”时，在默认题型之外必须包含它们。
    （e2e 评测：“20 道题，要有应用题”被当成“只出应用题”，按每题 10～20 分凑不成 100 分而被拒绝。）"""
    req_types = list(req.required_types)
    if req_types and req.only_types:
        return req_types
    base = DEFAULT_TYPES.get(req.subject, [QType.CHOICE, QType.FILL, QType.CALC])
    return [t for t in TYPE_ORDER if t in req_types or t in base]


class ExamRequest(BaseModel):
    stage: str = Field(description="学段，如 小学/初中/高中")
    grade: str = Field(description="年级，如 三年级、七年级")
    subject: str = Field(description="学科，如 数学")
    term: str = Field(default="全年", description="上 / 下 / 全年；老师没说册别就是全年（上下两册）")
    topic: str = Field(default="", description="老师指定的单元或知识点，如 一元一次方程；没指定为空")
    question_count: int = Field(gt=0, le=50)
    total_score: int = Field(gt=0, le=300)
    required_types: list[QType] = Field(default_factory=list)
    only_types: bool = Field(default=False, description="老师说“只要 / 只出某些题型”时为真；"
                             "否则 required_types 表示“必须包含”，其余题型可按默认补充")
    difficulty_note: str = Field(default="", description="整体难度要求，可为空")
    constraints: list[PaperConstraint] = Field(default_factory=list)
    extra_requirements: list[str] = Field(default_factory=list)

    @property
    def scope_text(self) -> str:
        """出卷范围的规范描述，如“三年级上册·两位数乘两位数”“三年级（上下册）”。"""
        book = {"上": "上册", "下": "下册"}.get(self.term, "（上下册）")
        return f"{self.stage}{self.grade}{book}" + (f"·{self.topic}" if self.topic else "")

    @property
    def title(self) -> str:
        """一二年级不进行纸笔考试（教育部 2021 年《关于加强义务教育学校考试管理的通知》），标为“练习卷”。"""
        book = {"上": "上册", "下": "下册"}.get(self.term, "")
        kind = "练习卷" if self.grade[:1] in "一二" else "测试卷"
        if self.topic:
            return f"{self.grade}{self.subject}《{self.topic}》单元{kind}"
        return f"{self.grade}{book}{self.subject}{kind}"


def integer_scope(req: "ExamRequest") -> set[QType] | None:
    """整数约束作用的题型；空集合 = 全卷（选择题除外）；None = 没有整数约束。"""
    scopes = [set(c.scope_types) for c in req.constraints if c.kind == ConstraintKind.INTEGER_ANSWER]
    if not scopes:
        return None
    return set() if any(not s for s in scopes) else set().union(*scopes)


def integer_required(req: "ExamRequest", qtype: QType) -> bool:
    scope = integer_scope(req)
    return scope is not None and qtype not in (QType.CHOICE, QType.READING, QType.WRITING) \
        and (not scope or qtype in scope)


class BlueprintSlot(BaseModel):
    slot_id: str
    position: int = Field(ge=1)
    qtype: QType
    score: int = Field(gt=0)
    knowledge_point: str
    difficulty: Difficulty
    cognitive_level: CognitiveLevel
    intent: str = Field(description="这道题想考查什么，一句话")
    kp_id: str = Field(default="", description="知识库卡片 id；从候选列表中选择")
    scope_note: str = Field(default="", description="该知识点的范围说明（主要内容与不考什么），由代码从知识库写入")


class Blueprint(BaseModel):
    slots: list[BlueprintSlot]
    rationale: str = ""
    scope: str = Field(default="", description="出卷范围及其确定方式，由代码写入")
    signals: dict[str, str] = Field(default_factory=dict, description="超纲信号词 → 来源，由代码写入")
    vocabulary: list[str] = Field(default_factory=list, description="英语：截至范围末尾已学词汇，由代码写入")

    def resequence(self) -> None:
        """按“题型标准顺序 → 原题号”重新编号。slot_id 永远不变，变的只是卷面题号。
        新增的 slot（position 很大）会排到同类题型的末尾。"""
        order = {t: i for i, t in enumerate(TYPE_ORDER)}
        self.slots.sort(key=lambda s: (order[s.qtype], s.position, s.slot_id))
        for i, s in enumerate(self.slots, start=1):
            s.position = i

    def next_slot_id(self) -> str:
        nums = [int(s.slot_id[1:]) for s in self.slots if s.slot_id[1:].isdigit()]
        return f"S{max(nums, default=0) + 1}"

    def slot_at(self, position: int) -> BlueprintSlot:
        for s in self.slots:
            if s.position == position:
                return s
        raise KeyError(f"第 {position} 题不存在")

    def slot(self, slot_id: str) -> BlueprintSlot:
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        raise KeyError(slot_id)


def _hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def new_qid() -> str:
    return "q_" + uuid.uuid4().hex[:8]


class VStatus(str, Enum):
    """每道题“被验证到什么程度”。无法自动验证是正常结果，不能包装成“审核通过”。"""
    TOOL = "工具验证"            # 工具独立算出答案，与给出的答案一致
    AGREED = "独立复核一致"      # 没有工具可用，但看不到答案的盲解 Agent 独立作答结果一致
    UNVERIFIED = "待老师确认"    # 没有任何手段能核对答案
    PROBLEM = "存在未解决问题"   # 审核发现的问题返修后仍未解决
    TEACHER = "老师已确认"       # 老师人工确认（可覆盖以上任何状态，原状态记录在 detail）
    NOT_REVIEWED = "未审核"      # 关闭了审核（--no-review）或旧版本数据


class Verification(BaseModel):
    status: VStatus
    method: str = Field(default="", description="怎么验证的，如“题干直接提取算式 → 精确计算”")
    detail: str = ""


class Question(BaseModel):
    qid: str = Field(default_factory=new_qid)
    slot_id: str
    qtype: QType
    score: int
    knowledge_point: str
    difficulty: Difficulty
    stem: str
    options: list[str] = Field(default_factory=list)
    answer: str
    analysis: str
    verification: Verification | None = None   # 不属于题目内容，不计入内容指纹

    @property
    def vstatus(self) -> VStatus:
        return self.verification.status if self.verification else VStatus.NOT_REVIEWED

    def fingerprint(self) -> str:
        """完整指纹（含分值等元数据）。"""
        return _hash(self.model_dump(mode="json"))

    def content_fingerprint(self) -> str:
        """内容指纹：只覆盖学生和老师看到的题目内容。
        用于证明“其他题别动”——为保持总分而调整分值不算改动题目内容。"""
        return _hash(self.model_dump(mode="json", include={
            "qid", "slot_id", "qtype", "knowledge_point", "difficulty",
            "stem", "options", "answer", "analysis"}))


class PaperSnapshot(BaseModel):
    paper_id: str
    title: str
    request: ExamRequest
    blueprint: Blueprint
    questions: list[Question]

    def ordered_questions(self) -> list[Question]:
        pos = {s.slot_id: s.position for s in self.blueprint.slots}
        return sorted(self.questions, key=lambda q: pos.get(q.slot_id, 10**6))

    def question_by_position(self, position: int) -> Question:
        for s in self.blueprint.slots:
            if s.position == position:
                for q in self.questions:
                    if q.slot_id == s.slot_id:
                        return q
        raise KeyError(f"第 {position} 题不存在")

    @property
    def total_score(self) -> int:
        return sum(q.score for q in self.questions)
