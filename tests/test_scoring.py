from exam_agent.domain import QType
from exam_agent.scoring import rebalance

C, F, K, A = QType.CHOICE, QType.FILL, QType.CALC, QType.APPLICATION


def total(counts, assign):
    return sum(counts[t] * assign[t] for t in counts)


def test_no_change_needed_keeps_scores():
    counts = {C: 2, F: 3, K: 4, A: 1}
    prev = {C: 5, F: 5, K: 15}
    # 一道计算题改成应用题，应用题沿用计算题的 15 分 → 无需改动任何分值
    assert rebalance(counts, 100, prev) == {C: 5, F: 5, K: 15, A: 15}


def test_add_question_rebalances_with_minimal_change():
    # 原来 4 选择(4分) + 4 填空(4分) + 2 计算(34分，不合理)；新增一道计算题后重新配分
    counts = {C: 4, F: 4, K: 3}
    prev = {C: 4, F: 4, K: 34}
    a = rebalance(counts, 100, prev)
    assert a is not None and total(counts, a) == 100


def test_teacher_fixed_score_is_respected():
    counts = {C: 5, F: 5, K: 3}
    a = rebalance(counts, 100, {C: 4, F: 4, K: 20}, fixed={C: 3})
    assert a[C] == 3 and total(counts, a) == 100


def test_infeasible_returns_none():
    assert rebalance({C: 2}, 100, {C: 4}) is None
