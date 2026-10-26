from fractions import Fraction as F

import pytest

from exam_agent.tools.linear import NotLinear, ToolError, extract_numbers, solve


def test_simple_equations():
    assert solve(["3x + 5 = 2x - 1"], ["x"]) == {"x": F(-6)}
    assert solve(["2(x - 3) - 3(x + 1) = 5(1 - x) + 2"], ["x"]) == {"x": F(4)}
    assert solve(["(x+1)/2 - (x-2)/3 = 1"], ["x"]) == {"x": F(-1)}


def test_exact_fractions_and_decimals():
    # 真实运行中模型把这题答成 1.5 小时，实际是 12/7 小时
    assert solve(["60t + 80t = 240"], ["t"]) == {"t": F(12, 7)}
    # 小数系数也精确：(0.3x - 0.4)/0.2 - (0.2x + 1)/0.5 = 1 → x = 50/11
    assert solve(["(0.3x - 0.4)/0.2 - (0.2x + 1)/0.5 = 1"], ["x"]) == {"x": F(50, 11)}


def test_full_width_and_chinese_symbols():
    assert solve(["２x＋３＝７".replace("２", "2").replace("３", "3").replace("７", "7")], ["x"]) == {"x": F(2)}
    assert solve(["6×x ÷ 3 = 4"], ["x"]) == {"x": F(2)}


def test_parameter_and_system():
    # 已知方程 2x + a = 3x - 1 的解是 x = 2，求 a
    assert solve(["2*2 + a = 3*2 - 1"], ["a"]) == {"a": F(1)}
    assert solve(["x + y = 10", "x - y = 2"], ["x", "y"]) == {"x": F(6), "y": F(4)}


def test_tool_admits_what_it_cannot_do():
    with pytest.raises(NotLinear):
        solve(["x^2 = 4"], ["x"])
    with pytest.raises(NotLinear):
        solve(["1/x = 2"], ["x"])
    with pytest.raises(ToolError):
        solve(["x + y = 3"], ["x", "y"])          # 条件不足
    with pytest.raises(ToolError):
        solve(["x + 1 = x + 2"], ["x"])           # 无解
    with pytest.raises(ToolError):
        solve(["x + k = 3"], ["x"])               # 未声明的字母


def test_extract_numbers_from_answers():
    assert extract_numbers("x = -6") == [F(-6)]
    assert extract_numbers("15支") == [F(15)]
    assert extract_numbers("1.5小时") == [F(3, 2)]
    assert extract_numbers("x = 22/5") == [F(22, 5)]
    assert extract_numbers("女生植树25棵") == [F(25)]
