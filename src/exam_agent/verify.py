"""验证路由：按题目形式选择核对工具，给出每道题的验证状态。

路由顺序（越靠前越可靠）：
  1. 从题干直接提取（不经过模型）：“解方程：3x + 5 = 20”“计算：125 × 8 =”
     → 线性求解器 / 精确计算。模型可能列错式，代码直接提取就没有这一层风险。
  2. 盲解 Agent 列出的方程 → 线性求解器。
  3. 盲解 Agent 列出的算式 → 精确计算（小学计算题、应用题）。
  4. 都没有：选择题比对盲解选的选项；其他题比对盲解答案中的数值 → “独立复核一致”。
  5. 什么都比对不了 → “待老师确认”。
工具算出的值与答案不一致时报 wrong_answer；工具算不了（ToolError）就换下一条路由，绝不据此下结论。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from .domain import QType, Question, Verification, VStatus
from .tools.arith import evaluate, matches_answer, parse_remainder_answer, remainder_division
from .tools.linear import ToolError, extract_numbers, fmt, solve

_SOLVE = re.compile(r"^\s*解(?:下列)?方程[:：]\s*(?P<eq>[^，。；;\n]+?)\s*[。.]?\s*$")
_CALC = re.compile(r"^\s*(?:(?:直接写出得数|口算|计算|脱式计算|简便计算|用简便方法计算)[:：])?\s*"
                   r"(?P<expr>[\d\s.+\-−×÷*/()（）%％²³^]+?)\s*(?:=|＝)?\s*(?:_{2,}|（\s*）|\(\s*\))?\s*[。.]?\s*$")


@dataclass
class Formal:
    """题目的形式化结果：可交给工具计算的方程或算式。"""
    kind: str               # "equation" | "expression"
    text: list[str]
    unknowns: tuple[str, ...] = ()
    target: str = ""
    source: str = ""        # 题干直接提取 / 盲解列方程 / 盲解列算式


def from_stem(stem: str) -> Formal | None:
    """只处理“整道题就是一个算式或一个方程”的情况；多小题、带文字情境的一律不提取。"""
    m = _SOLVE.match(stem)
    if m:
        eq = m.group("eq")
        letters = sorted(set(re.findall(r"[a-zA-Z]", eq)))
        if eq.count("=") + eq.count("＝") == 1 and len(letters) == 1:
            return Formal("equation", [eq], (letters[0],), letters[0], "题干直接提取方程")
    m = _CALC.match(stem)
    if m and re.search(r"\d", m.group("expr")) and re.search(r"[+\-×÷*/]", m.group("expr")):
        return Formal("expression", [m.group("expr")], source="题干直接提取算式")
    return None


def _compute(f: Formal) -> Fraction:
    if f.kind == "equation":
        return solve(f.text, list(f.unknowns))[f.target]
    return evaluate(f.text[0])


@dataclass
class RouteResult:
    verification: Verification
    tool_value: Fraction | None = None
    mismatch: str = ""      # 非空表示与答案不一致
    code: str = "wrong_answer"
    source: str = "tool"


_ITEM = re.compile(r"(\d+)\s*[.、:：)）]?\s*([A-Da-dTFtf]|√|×)(?![A-Za-z])")


def reading_answers(text: str) -> dict[int, str]:
    """“1. B 2. C 3. A” → {1: "B", 2: "C", 3: "A"}；判断题“1. T 2. F”或“1. √ 2. ×”统一为 T/F。"""
    norm = {"√": "T", "×": "F"}
    return {int(n): norm.get(l, l.upper()) for n, l in _ITEM.findall(text)}


def _words(text: str) -> list[str]:
    """英语填空答案的比较单位。去掉小题编号“1.”“(2)”，“1. bread 2. milk”与“bread; milk”视为相同。"""
    text = re.sub(r"[(（]?\b\d+\s*[.)）、]", " ", text.lower())
    return [w for w in re.split(r"[\s;；,，、/]+", re.sub(r"[.!?。！？\"“”]", "", text)) if w]


def route(q: Question, blind=None) -> RouteResult:
    """blind：盲解 Agent 对这道题的作答（BlindSolution），可以为 None。"""
    if q.qtype == QType.WRITING:
        return RouteResult(Verification(status=VStatus.UNVERIFIED, method="主观题：范文与评分要点由老师把握"))
    if q.qtype == QType.READING:
        return _route_reading(q, blind)
    candidates: list[Formal] = []
    remainder = (parse_remainder_answer(q.answer) or len(extract_numbers(q.answer)) == 2) \
        and re.search(r"÷|/|平均|除", q.stem) and len(_SUBITEM.findall(q.stem)) < 2      # 商和余数两个空
    if is_multi_part(q.stem) and not remainder and extract_numbers(_strip_items(q.answer)):
        # 一道题有多个小题 / 多个空时，一个算式只能核对其中一个空，不能据此标“工具验证”（e2e 试跑：
        # 5 个小题的题只核对了一个就显示“工具验证”）。改为比对盲解对全部空的作答。
        return _route_multi(q, blind)
    direct = from_stem(q.stem) if q.qtype != QType.CHOICE else None
    if direct:
        candidates.append(direct)
    if blind is not None and q.qtype != QType.CHOICE:
        if blind.equations:
            target = blind.target or (blind.unknowns[0] if blind.unknowns else "")
            if target:
                candidates.append(Formal("equation", blind.equations, tuple(blind.unknowns), target, "盲解列方程"))
        if getattr(blind, "expression", ""):
            candidates.append(Formal("expression", [blind.expression], source="盲解列算式"))

    for f in candidates:
        # 答案写成“3……2”时，按有余数的除法比对商和余数；写成小数则走下面的精确比对
        stated_qr = parse_remainder_answer(q.answer)
        if f.kind == "expression" and stated_qr:
            try:
                qr = remainder_division(f.text[0])
            except ToolError:
                qr = None
            if qr:
                method = f"{f.source} → 有余数除法"
                if stated_qr == qr:
                    return RouteResult(Verification(status=VStatus.TOOL, method=method,
                                                    detail=f"{f.text[0]} = {qr[0]}……{qr[1]}"))
                return RouteResult(Verification(status=VStatus.PROBLEM, method=method),
                                   mismatch=f"按 {f.text[0]} 计算应为 {qr[0]}……{qr[1]}，但答案写的是“{q.answer}”")
        try:
            value = _compute(f)
        except (ToolError, KeyError):
            continue      # 这条路由算不了，换下一条
        method = f"{f.source} → {'线性方程求解器' if f.kind == 'equation' else '精确计算'}"
        stated = extract_numbers(q.answer)
        v = Verification(status=VStatus.TOOL, method=method, detail=f"{'；'.join(f.text)} ⇒ {fmt(value)}")
        if value in stated or (f.kind == "expression" and matches_answer(f.text[0], value, q.answer)):
            qr = remainder_division(f.text[0]) if f.kind == "expression" else None
            if qr and qr[1] and value not in stated:
                v.detail = f"{f.text[0]} = {qr[0]}……{qr[1]}"
            return RouteResult(v, value)
        if stated:
            v.status = VStatus.PROBLEM
            how = "求解" if f.kind == "equation" else "计算"
            return RouteResult(v, value, f"{f.source}，精确{how}得 {fmt(value)}（约 {float(value):.4g}），"
                                         f"但答案写的是“{q.answer}”")
        return RouteResult(Verification(status=VStatus.UNVERIFIED, method=method,
                                        detail=f"工具算得 {fmt(value)}，但答案里没有可比对的数值"), value)

    # 数值型选择题：盲解列出算式 → 精确计算 → 找到唯一对应的选项，与答案键比对
    if blind is not None and q.qtype == QType.CHOICE and getattr(blind, "expression", "") and q.options:
        try:
            value = evaluate(blind.expression)
        except ToolError:
            value = None
        if value is not None:
            hits = [i for i, o in enumerate(q.options) if value in extract_numbers(o)]
            if len(hits) == 1:
                letter = "ABCD"[hits[0]]
                method = "盲解列算式 → 精确计算 → 对应选项"
                detail = f"{blind.expression} ⇒ {fmt(value)}，对应选项 {letter}"
                if letter == q.answer:
                    return RouteResult(Verification(status=VStatus.TOOL, method=method, detail=detail), value)
                return RouteResult(Verification(status=VStatus.PROBLEM, method=method, detail=detail), value,
                                   f"按 {blind.expression} 精确计算得 {fmt(value)}，对应选项 {letter}，但答案键是 {q.answer}")

    # 比较类选择题：盲解为每个选项列算式，程序精确计算后按最大 / 最小选出答案（d019：统计表题两个模型都选错）
    if blind is not None and q.qtype == QType.CHOICE and getattr(blind, "option_exprs", None) \
            and getattr(blind, "pick", "") in ("max", "min"):
        try:
            vals = {k.strip().upper()[:1]: evaluate(v) for k, v in blind.option_exprs.items() if v.strip()}
        except ToolError:
            vals = {}
        if len(vals) >= 2:
            best = (max if blind.pick == "max" else min)(vals.values())
            winners = [k for k, v in vals.items() if v == best]
            method = "盲解为各选项列算式 → 精确计算 → 比较"
            detail = "，".join(f"{k}={fmt(v)}" for k, v in sorted(vals.items()))
            if len(winners) == 1:
                if winners[0] == q.answer:
                    return RouteResult(Verification(status=VStatus.TOOL, method=method, detail=detail), best)
                return RouteResult(Verification(status=VStatus.PROBLEM, method=method, detail=detail), best,
                                   f"按各选项算式精确计算：{detail}，{'最大' if blind.pick == 'max' else '最小'}的是 "
                                   f"{winners[0]}，但答案键是 {q.answer}")

    if blind is not None:
        if q.qtype == QType.CHOICE and blind.choice.strip():
            if blind.choice.strip().upper()[:1] == q.answer:
                return RouteResult(Verification(status=VStatus.AGREED, method="盲解独立选择",
                                                detail=f"盲解也选 {q.answer}"))
            return RouteResult(Verification(status=VStatus.UNVERIFIED, method="盲解独立选择"))  # 不一致由审核报告
        mine, theirs = set(extract_numbers(q.answer)), set(extract_numbers(blind.final_answer))
        if q.answer.isascii() and re.search(r"[A-Za-z]", q.answer):   # 英语填空：编号不是答案数值
            mine = set()
        if mine and mine == theirs:
            return RouteResult(Verification(status=VStatus.AGREED, method="盲解独立作答",
                                            detail=f"盲解答案：{blind.final_answer[:30]}"))
        if not mine and q.answer.isascii() and _words(q.answer) and _words(q.answer) == _words(blind.final_answer):
            return RouteResult(Verification(status=VStatus.AGREED, method="盲解独立作答",
                                            detail=f"盲解也填 {blind.final_answer[:30]}"))
        # 英语填空可能有多个正确答案：不一致时不下结论，交给老师
    return RouteResult(Verification(status=VStatus.UNVERIFIED, method="没有可用的核对工具"))


_SUBITEM = re.compile(r"[（(]\s*\d+\s*[)）]")


def _strip_items(text: str) -> str:
    """去掉小题编号“（1）”“1.”，避免把编号当成答案数值。"""
    return re.sub(r"[（(]\s*\d+\s*[)）]|(?<![\d.])\d+\s*[.、](?!\d)", " ", text)


def is_multi_part(stem: str) -> bool:
    """多个小题或多个空。“17 ÷ 5 = ____……____”（商和余数）是一个算式的两个空，不算。"""
    if len(_SUBITEM.findall(stem)) >= 2:
        return True
    blanks = re.findall(r"_{2,}", stem)
    return len(blanks) >= 2 and not (len(blanks) == 2 and re.search(r"_{2,}\s*…+\s*_{2,}", stem))


def _route_multi(q: Question, blind) -> RouteResult:
    """多小题 / 多空：比对盲解给出的全部数值（不计小题编号），一致才算“独立复核一致”。"""
    mine = extract_numbers(_strip_items(q.answer))
    theirs = extract_numbers(_strip_items(blind.final_answer)) if blind is not None else []
    method = f"多空题（{max(len(_SUBITEM.findall(q.stem)), len(re.findall(r'_{2,}', q.stem)))} 处作答）：盲解逐空作答"
    if mine and theirs and sorted(mine) == sorted(theirs):
        return RouteResult(Verification(status=VStatus.AGREED, method=method, detail=f"{len(mine)} 个数值全部一致"))
    return RouteResult(Verification(status=VStatus.UNVERIFIED, method=method + "，未能逐空核对"))


def _route_reading(q: Question, blind) -> RouteResult:
    mine = reading_answers(q.answer)
    if not mine:
        return RouteResult(Verification(status=VStatus.UNVERIFIED, method="阅读理解：答案格式无法逐题比对"))
    theirs = reading_answers(blind.final_answer) if blind is not None else {}
    if not theirs:
        return RouteResult(Verification(status=VStatus.UNVERIFIED, method="阅读理解：没有独立作答结果"))
    diff = [f"第{n}小题独立作答选 {theirs.get(n, '无')}，答案是 {a}" for n, a in sorted(mine.items()) if theirs.get(n) != a]
    if diff:
        return RouteResult(Verification(status=VStatus.PROBLEM, method="盲解逐题独立作答"),
                           mismatch="；".join(diff) + "，请核对短文和答案", code="reading_mismatch", source="blind")
    return RouteResult(Verification(status=VStatus.AGREED, method="盲解逐题独立作答",
                                    detail=f"{len(mine)} 个小题全部一致"))
