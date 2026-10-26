"""英语题目的年级适切性检查。

规则（老师定的）：
  1. 语法严格：按课标分级硬性检查（超出课标级别 → 必须修改）；教材通常更晚才学的 → 只提示。
  2. 阅读短文长度在年级区间内：3～4 年级 50～200 词，5～6 年级 100～300 词，七年级 150～400 词（超出区间 → 必须修改）。
  3. 生词：不在课标词汇表中的词（小学对照二级词汇表，初中对照三级词汇表）。阅读短文生词比例超过上限
     （默认 15%，例如 100 词里有 40 个生词）→ 必须修改。人名、数词、月份星期、作答说明不算生词。
  4. 在这些范围内，短文偏难还是偏易由老师判断（例如对话中说“第 20 题太难了，换一篇”）。

依据
  - 语法分级：《义务教育英语课程标准（2022年版）》表 11 语法知识内容要求（一级：一般现在时、现在进行时；
    二级：一般过去时、一般将来时、比较人和物）与附录 4 语法项目表（三级）。
  - 教材顺序（只用于提示）：人教版教材顺序，待核对。
  - 生词：课标附录 3 二级词汇表（505 词，小学）、三级词汇表（1600 词，初中），见 kb/english/standard/。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..retrieval.kb import KB_ROOT, book_base
from .vocab import BASIC, IRREGULAR, NAMES, lemmas

STD = KB_ROOT / "english" / "standard"
GRADE_LEVEL = {3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}
LEVEL_NAME = {1: "一级（3～4年级）", 2: "二级（5～6年级）", 3: "三级（7～9年级）"}


@dataclass(frozen=True)
class Limits:
    passage_min: int          # 阅读短文词数下限
    passage_max: int          # 阅读短文词数上限
    reading_items: int        # 一份卷最多几篇阅读
    target: int = 0           # 给命题员的目标词数（区间中部偏下，写短了也仍在区间内）
    paragraphs: int = 0       # 用模型能数准的单位下指令：几段、每段几句、每句约几词
    sentences: int = 5
    sentence_words: int = 7

    @property
    def shape(self) -> str:
        return (f"写 {self.paragraphs} 段，每段 {self.sentences} 句，每句约 {self.sentence_words} 个词"
                f"（合计约 {self.target} 词）")


# 老师给定的短文长度区间（3～4 年级及以下 50～200，5～6 年级 100～300，七年级 150～400）
# d038：只给区间时模型贴着下限写、而且数不准词数（三年级只写出 30～40 词），所以另给目标值和段落句数
LIMITS = {1: Limits(50, 200, 2, target=105, paragraphs=3, sentences=5, sentence_words=7),
          2: Limits(100, 300, 3, target=180, paragraphs=4, sentences=5, sentence_words=9),
          3: Limits(150, 400, 4, target=250, paragraphs=5, sentences=5, sentence_words=10)}
NEW_WORD_MAX = 0.15   # 阅读短文生词比例上限（老师的例子：100 词里 40 个生词必须打回）；可按需要调整
_INSTRUCTION = re.compile(r"^(read|choose|answer|match|look|fill|write|listen)\b.*\b(passage|question|answer|"
                          r"information|correct|blank|sentence|following)\b", re.I)

PAST_IRREGULAR = {k for k, v in IRREGULAR.items() if v not in ("good", "bad") and k not in
                  ("children", "men", "women", "feet", "teeth", "mice", "geese", "people", "better", "best", "worse",
                   "worst", "read", "left", "given", "taken", "eaten", "seen", "gone", "done", "written")}
PARTICIPLES = {"been", "done", "gone", "seen", "eaten", "taken", "given", "written", "made", "had", "bought", "built",
               "taught", "told", "found", "heard", "lost", "kept", "sent", "spent", "brought", "caught", "thought"}
ED_ADJECTIVES = {"excited", "interested", "tired", "bored", "worried", "surprised", "pleased", "scared", "named",
                 "red", "bed", "need", "feed", "seed", "speed", "shed", "bread", "head", "read", "ready", "hundred"}
ING_NOUNS = {"morning", "evening", "thing", "something", "anything", "nothing", "everything", "king", "ring",
             "spring", "swing", "building", "ceiling", "wing", "sing", "bring", "string", "painting", "drawing",
             "meeting", "clothing", "wedding", "pudding", "ping", "during", "interesting", "boring", "amazing"}
_LAST = r"last\s+(night|week|weekend|year|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday|summer|winter)"


def _past_regular(words: list[str], vocab: set[str]) -> list[str]:
    hits = []
    for w in words:
        if w.endswith("ed") and w not in ED_ADJECTIVES and len(w) > 4:
            bases = {w[:-2], w[:-1], w[:-3] + "y" if w.endswith("ied") else "", w[:-3] if w[-3] == w[-4] else ""}
            if bases & vocab:
                hits.append(w)
    return hits


@dataclass(frozen=True)
class GrammarItem:
    key: str
    name: str
    level: int          # 课标级别（硬性）
    textbook: str       # 教材通常最早出现的册（软性，待核对）

    def find(self, text: str, words: list[str], vocab: set[str]) -> list[str]:
        t = text
        if self.key == "progressive":
            return [m.group(0) for m in re.finditer(r"\b(am|is|are|'m|'re|'s)\s+(\w+ing)\b", t)
                    if m.group(2) not in ING_NOUNS]
        if self.key == "past":
            out = [m.group(0) for m in re.finditer(r"\b(was|were|did|didn't|wasn't|weren't|yesterday|ago)\b|\b" + _LAST, t)]
            out += [w for w in words if w in PAST_IRREGULAR]
            return out + _past_regular(words, vocab)
        if self.key == "future":
            return [m.group(0) for m in re.finditer(r"\b(will|won't)\s+\w+|\w+'ll\b|\b(am|is|are|'m|'re|'s)\s+going\s+to\b", t)]
        if self.key == "comparative":
            return [m.group(0) for m in re.finditer(r"\b\w+\s+than\b", t)]
        if self.key == "superlative":
            out = [m.group(0) for m in re.finditer(r"\bthe\s+(best|worst|most\s+\w+)\b", t)]
            out += [m.group(0) for m in re.finditer(r"\bthe\s+(\w+est)\b", t)
                    if {m.group(1)[:-3], m.group(1)[:-2], m.group(1)[:-4]} & vocab]
            return out
        if self.key == "perfect":
            return [m.group(0) for m in re.finditer(r"\b(have|has|haven't|hasn't|'ve)\s+(\w+ed|" +
                                                    "|".join(PARTICIPLES) + r")\b", t)
                    if m.group(2) not in ED_ADJECTIVES and m.group(2) != "had"]
        if self.key == "past_progressive":
            return [m.group(0) for m in re.finditer(r"\b(was|were)\s+(\w+ing)\b", t) if m.group(2) not in ING_NOUNS]
        if self.key == "passive":
            return [m.group(0) for m in re.finditer(r"\b(is|are|was|were|be|been)\s+(\w+ed|made|done|written|built|"
                                                    r"given|taken|eaten|seen|sold|taught|told)\s+(by|in|on|at)\b", t)
                    if m.group(2) not in ED_ADJECTIVES]
        return []


GRAMMAR = [
    GrammarItem("progressive", "现在进行时", 1, "五下"),
    GrammarItem("past", "一般过去时", 2, "六下"),
    GrammarItem("future", "一般将来时", 2, "六上"),
    GrammarItem("comparative", "比较级", 2, "六下"),
    GrammarItem("superlative", "最高级", 3, "八上"),
    GrammarItem("perfect", "现在完成时", 3, "八下"),
    GrammarItem("past_progressive", "过去进行时", 3, "八下"),
    GrammarItem("passive", "被动语态", 3, "九上"),
]

NUMBERS_ETC = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million
first second third fourth fifth sixth seventh eighth ninth tenth january february march april may june july august
september october november december monday tuesday wednesday thursday friday saturday sunday""".split())


@lru_cache(maxsize=4)
def level_vocab(level: int, root: Path = STD) -> frozenset[str]:
    """一、二级用二级词汇表（小学），三级用三级词汇表（初中）；数词、月份、星期各级都可用。"""
    name = "vocab_level3.txt" if level >= 3 else "vocab_level2.txt"
    path = root / name
    words = set(path.read_text(encoding="utf-8").split("\n")) if path.exists() else set()
    flat = {w for p in words for w in p.split()} | words
    return frozenset(w for w in flat if w) | NUMBERS_ETC | BASIC


def english_words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text)]


def passage_of(stem: str) -> str:
    """阅读理解题干：去掉作答说明（中文）和编号小题，只留短文。"""
    lines = []
    for line in stem.splitlines():
        s = line.strip()
        if not s or re.match(r"^[(（]?\d+\s*[.、)）]", s) or re.match(r"^[A-D]\s*[.、]", s):
            continue   # 编号小题“1.”“(1)”和选项行
        if _INSTRUCTION.match(s):   # “Read the passage and match the information.” 是作答说明，不是短文
            continue
        if re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]{3,}", s):
            continue
        lines.append(s)
    return " ".join(lines)


@dataclass
class Difficulty:
    words: int
    sentences: int
    sentence_avg: float
    unknown: list[str] = field(default_factory=list)

    @property
    def unknown_rate(self) -> float:
        return len(self.unknown) / self.words if self.words else 0.0


def measure(text: str, vocab: frozenset[str]) -> Difficulty:
    sents = [s for s in re.split(r"[.!?]+", text) if english_words(s)]
    words = english_words(text)
    caps = {m.group(0).lower() for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+", text)}  # 句中大写：专有名词
    unknown = [w for w in words if w not in NAMES and w not in caps and not (lemmas(w) & vocab) and len(w) > 1]
    return Difficulty(len(words), len(sents), round(len(words) / len(sents), 1) if sents else 0.0, unknown)


def scope_end_book(grade: str, term: str) -> str:
    return f"{grade[:1]}{'上' if term == '上' else '下'}"


@dataclass
class Issue:
    severity: str   # must_fix / suggest
    code: str
    message: str


def check_question(qtype: str, stem: str, options: list[str], answer: str, grade_no: int, end_book: str) -> list[Issue]:
    """返回该题的年级适切性问题。qtype 用中文题型名。"""
    level = GRADE_LEVEL.get(grade_no)
    if level is None:
        return []
    vocab = level_vocab(level)   # 小学：二级词汇表；初中：三级词汇表
    issues: list[Issue] = []
    # 语法：看题目本身（题干 + 选项）；书面表达看范文
    text = (answer if qtype == "书面表达" else stem + "\n" + "\n".join(options))
    low = text.lower()
    words = english_words(text)
    for item in GRAMMAR:
        hits = item.find(low, words, set(vocab))
        if not hits:
            continue
        ex = "、".join(dict.fromkeys(h.strip() for h in hits))[:60]
        if item.level > level:
            issues.append(Issue("must_fix" if qtype != "书面表达" else "suggest", "grammar_level",
                                f"使用了{item.name}（“{ex}”），课标列为{LEVEL_NAME[item.level]}内容，"
                                f"超出{grade_no}年级（{LEVEL_NAME[level]}）要求"))
        elif book_base(item.textbook) > book_base(end_book):
            issues.append(Issue("suggest", "grammar_textbook",
                                f"使用了{item.name}（“{ex}”），人教版教材通常在{item.textbook}才系统学习（教材顺序，待核对）"))
    if qtype == "阅读理解":
        lim = LIMITS[level]
        n = len(english_words(passage_of(stem)))
        if not lim.passage_min <= n <= lim.passage_max:
            issues.append(Issue("must_fix", "reading_length",
                                f"短文 {n} 词，不在{grade_no}年级的 {lim.passage_min}～{lim.passage_max} 词范围内"))
        d = measure(passage_of(stem), vocab)
        if d.words and d.unknown_rate > NEW_WORD_MAX:
            uniq = list(dict.fromkeys(d.unknown))
            issues.append(Issue("must_fix", "reading_new_words",
                                f"短文 {d.words} 词中有 {len(d.unknown)} 个生词（{d.unknown_rate:.0%}），超过上限 {NEW_WORD_MAX:.0%}；"
                                f"不在课标{'三' if level >= 3 else '二'}级词汇表中的词：{', '.join(uniq[:12])}"))
    return issues


def grade_rules_text(grade_no: int) -> str:
    """给命题员看的本年级规则。"""
    level = GRADE_LEVEL.get(grade_no)
    if level is None:
        return ""
    lim = LIMITS[level]
    allowed = "、".join(i.name for i in GRAMMAR if i.level <= level) or "一般现在时"
    banned = "、".join(i.name for i in GRAMMAR if i.level > level)
    return (f"【{grade_no}年级英语规则】语法只用课标{LEVEL_NAME[level]}范围：一般现在时、{allowed}"
            + (f"；不要使用{banned}" if banned else "") +
            f"。阅读短文必须在 {lim.passage_min}～{lim.passage_max} 词之间，按目标写：{lim.shape}，"
            f"段与段之间换行，不要写短；尽量使用课标{'三' if level >= 3 else '二'}级词汇表中的词，"
            f"生词不超过 {NEW_WORD_MAX:.0%}；一份卷阅读理解不超过 {lim.reading_items} 篇。")
