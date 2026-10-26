"""英语词汇范围核对：题目里的单词是否都在“截至出卷范围已学过的词汇”里。

词表来自知识库单元卡片（模型起草，未必完整），所以这里只给出“建议”级提示，不触发返修；
它的价值是把“可能超纲的词”挑出来给老师看，而不是替老师下结论。
"""
from __future__ import annotations

import re

# 功能词、数词、常见人名：不计入核对
BASIC = set("""
a an the and or but so because if then than as of in on at to for with from by into out up down off over under
about after before near behind between next i you he she it we they me him her us them my your his its our their
mine yours this that these those here there what who whom whose where when why how which be am is are was were
been being do does did done have has had not no yes can could will would shall should may might must let please
thank thanks hello hi goodbye bye ok okay oh mr mrs miss ms dear very too also only just all some any many much
more most lot lots every each other another one two three four five six seven eight nine ten eleven twelve
thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty
ninety hundred thousand first second third
read passage choose correct answer answers question questions match information true false sentence sentences
blank blanks fill write word words following according
""".split())   # 后两行是题目说明用语（Read the passage and choose…），不算短文生词
NAMES = set("""amy sarah mike john chen jie wu binbin yifan zhang peng zoom zip white jones oliver sam tom lily
lucy bob ben kate jim li ming wang tao""".split())
IRREGULAR = {
    "went": "go", "gone": "go", "did": "do", "done": "do", "had": "have", "saw": "see", "seen": "see", "ate": "eat",
    "eaten": "eat", "took": "take", "taken": "take", "made": "make", "got": "get", "came": "come", "gave": "give",
    "given": "give", "bought": "buy", "thought": "think", "said": "say", "told": "tell", "found": "find", "felt": "feel",
    "left": "leave", "met": "meet", "ran": "run", "sang": "sing", "sung": "sing", "swam": "swim", "wrote": "write",
    "written": "write", "rode": "ride", "drank": "drink", "slept": "sleep", "sat": "sit", "stood": "stand",
    "flew": "fly", "drew": "draw", "knew": "know", "grew": "grow", "threw": "throw", "began": "begin", "won": "win",
    "lost": "lose", "kept": "keep", "taught": "teach", "caught": "catch", "brought": "bring", "spent": "spend",
    "sent": "send", "built": "build", "heard": "hear", "learnt": "learn", "children": "child", "men": "man",
    "women": "woman", "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose", "people": "person",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
}
_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def lemmas(word: str) -> set[str]:
    """不依赖词典的简单词形还原：返回可能的原形集合（宁可多猜，避免误报）。"""
    w = word.lower().split("'")[0]
    out = {w, IRREGULAR.get(w, w)}
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ied", "y"), ("ed", ""), ("ed", "e"), ("ing", ""),
                     ("ing", "e"), ("ier", "y"), ("iest", "y"), ("er", ""), ("er", "e"), ("est", ""), ("est", "e"),
                     ("ly", "")):
        if w.endswith(suf) and len(w) > len(suf) + 1:
            base = w[: -len(suf)] + rep
            out.add(base)
            if len(base) > 2 and base[-1] == base[-2]:     # stopped → stop, bigger → big
                out.add(base[:-1])
    return out


def out_of_vocabulary(text: str, allowed: set[str]) -> list[str]:
    """返回不在词表里的单词（去重、保持出现顺序）。人名、功能词、句中大写的专有名词不计。"""
    known = allowed | BASIC
    phrases_words = {w for p in allowed for w in p.split()}
    known |= phrases_words
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        for i, m in enumerate(_WORD.finditer(sentence)):
            tok = m.group(0)
            low = tok.lower()
            if low in NAMES or (tok[0].isupper() and i > 0 and low != "i"):
                continue   # 句中大写词多为专有名词
            if lemmas(tok) & known:
                continue
            if low not in out:
                out.append(low)
    return out
