"""分值重配求解器：纯算术问题交给代码，不交给模型。

给定各题型题数和满分，在合理分值范围内找出“每类题每题分值”，使总分恰好等于满分，
并且相对原分值的改动尽量小（老师已指定的题型分值保持不变）。
"""
from __future__ import annotations

from itertools import product

from .domain import OBJECTIVE_TYPES, TYPE_ORDER, Blueprint, QType

# 各题型每题分值的合理范围
SCORE_RANGE: dict[QType, range] = {
    QType.CHOICE: range(1, 7),        # 老师的要求：选择、填空每题 1～6 分
    QType.FILL: range(1, 7),
    QType.CALC: range(10, 21),        # 计算、应用每题 10～20 分
    QType.APPLICATION: range(10, 21),
    QType.READING: range(4, 21),
    QType.WRITING: range(5, 26),
}
# 常见分值区间：区间外的分值允许，但会被尽量避免
TYPICAL_RANGE: dict[QType, range] = {
    QType.CHOICE: range(3, 6),
    QType.FILL: range(3, 6),
    QType.CALC: range(10, 16),
    QType.APPLICATION: range(10, 16),
    QType.READING: range(8, 16),
    QType.WRITING: range(10, 21),
}
DEFAULT_SCORE: dict[QType, int] = {QType.CHOICE: 4, QType.FILL: 4, QType.CALC: 10, QType.APPLICATION: 10,
                                   QType.READING: 10, QType.WRITING: 15}


def current_scores(bp: Blueprint) -> dict[QType, int]:
    """每类题型当前最常见的分值。"""
    seen: dict[QType, list[int]] = {}
    for s in bp.slots:
        seen.setdefault(s.qtype, []).append(s.score)
    return {t: max(set(v), key=v.count) for t, v in seen.items()}


def rebalance(
    counts: dict[QType, int],
    total: int,
    previous: dict[QType, int],
    fixed: dict[QType, int] | None = None,
    keep: dict[QType, int] | None = None,
) -> dict[QType, int] | None:
    """返回每类题型的每题分值；无解时返回 None。
    keep：已有试卷里正在使用的分值，即使不在当前的分值范围内也允许保留（d041：分值范围调整后，
    旧试卷的计算题每题 6 分不再合法，导致对旧卷的任何修改都无法重配分值）。

    目标（按优先级）：1) 总分正确；2) 尽量少用不常见的分值（如每题 19 分）；
    3) 改动的题数、偏差最小；4) 平局时让主观题分值不低于客观题。
    """
    fixed = fixed or {}
    types = [t for t, c in counts.items() if c > 0]
    prev = dict(previous)
    for t in types:
        if t not in prev:
            # 新出现的题型：应用题参考计算题，其余用默认值
            prev[t] = prev.get(QType.CALC, DEFAULT_SCORE[t]) if t == QType.APPLICATION else DEFAULT_SCORE[t]

    keep = keep or {}
    choices = [[fixed[t]] if t in fixed else
               sorted(set(SCORE_RANGE[t]) | ({keep[t]} if t in keep else set())) for t in types]
    best: tuple[tuple, dict[QType, int]] | None = None
    for combo in product(*choices):
        if sum(counts[t] * sc for t, sc in zip(types, combo)) != total:
            continue
        assign = dict(zip(types, combo))
        changed_questions = sum(counts[t] for t in types if assign[t] != prev[t])
        deviation = sum(counts[t] * abs(assign[t] - prev[t]) for t in types)
        objective = max((assign[t] for t in types if t in OBJECTIVE_TYPES), default=0)
        subjective = min((assign[t] for t in types if t not in OBJECTIVE_TYPES), default=99)
        inverted = 1 if objective > subjective else 0
        atypical = sum(counts[t] for t in types if assign[t] not in TYPICAL_RANGE[t])
        key = (atypical, changed_questions, deviation, inverted)
        if best is None or key < best[0]:
            best = (key, assign)
    return best[1] if best else None


def feasible_count_options(n: int, total: int, types: list[QType], limit: int = 6,
                           max_counts: dict[QType, int] | None = None) -> list[dict[QType, int]]:
    """枚举“每种题型至少 1 题、共 n 题、能凑成满分”的题数配比，按均衡程度排序。
    在调用规划模型之前算好，让模型从可行方案里选，而不是猜错后再返工。"""
    from itertools import product as _product

    types = [t for t in TYPE_ORDER if t in types]
    if not types:
        return []
    defaults = {t: DEFAULT_SCORE[t] for t in types}
    options = []
    for combo in _product(range(1, n + 1), repeat=len(types)):
        if sum(combo) != n:
            continue
        counts = dict(zip(types, combo))
        if max_counts and any(counts[t] > m for t, m in max_counts.items() if t in counts):
            continue   # 例如三年级英语卷阅读理解最多 2 篇（d025：不给上限时模型连续 3 次规划了 4 篇）
        assign = rebalance(counts, total, defaults)
        if assign is None:
            continue
        atypical = sum(counts[t] for t in types if assign[t] not in TYPICAL_RANGE[t])
        mean = n / len(types)
        options.append(((atypical, sum((c - mean) ** 2 for c in combo)), counts))
    options.sort(key=lambda x: x[0])
    return [c for _, c in options[:limit]]
