"""算式的精确计算：小学计算题、应用题的验题工具。

复用线性方程求解器的表达式解析（不含未知数的表达式就是一个常数），全程 Fraction 精确运算：
0.1 + 0.2 得到 3/10，而不是 0.30000000000000004。
额外支持：百分数（25% → 1/4）、有余数的除法（17 ÷ 5 = 3……2）、中文括号与运算符。
"""
from __future__ import annotations

import re
from fractions import Fraction

from .linear import ToolError, extract_numbers, normalize, parse_linear

_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")
_REMAINDER = re.compile(r"(-?\d+)\s*(?:…+|\.{3,}|余)\s*(\d+)")
_INT_DIV = re.compile(r"^\(?(\d+)\)?/\(?(\d+)\)?$")


def evaluate(expr: str) -> Fraction:
    """计算不含未知数的算式。含字母、无法解析时抛出 ToolError，不猜。"""
    text = _PERCENT.sub(r"(\1/100)", expr.replace("％", "%"))
    text = text.split("=")[0]          # “125×8=” 只取等号左边
    lin = parse_linear(text)
    if not lin.is_const:
        raise ToolError(f"算式中含有未知数：{sorted(lin.coeffs)}")
    return lin.const


def remainder_division(expr: str) -> tuple[int, int] | None:
    """算式恰好是“整数 ÷ 整数”时返回 (商, 余数)，否则返回 None。"""
    m = _INT_DIV.match(normalize(expr.split("=")[0]))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        raise ToolError("除数为 0")
    return divmod(a, b)


def parse_remainder_answer(answer: str) -> tuple[int, int] | None:
    """识别“3……2”“3...2”“商3余2”这类有余数的答案。"""
    m = _REMAINDER.search(answer.replace("商", ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def matches_answer(expr: str, value: Fraction, answer: str) -> bool:
    """算式结果与答案是否一致。有余数的除法接受三种写法：“3……2”、分两空填写的“3，2”、小数“3.4”。
    （实测：“每人分到 ____ 个，还剩 ____ 个”答案写“3，2”，只按 17/5 比对会被误判为算错，见 d017。）"""
    nums = extract_numbers(answer)
    if value in nums:
        return True
    try:
        qr = remainder_division(expr)
    except ToolError:
        return False
    if not qr or qr[1] == 0:
        return False
    if parse_remainder_answer(answer) == qr:
        return True
    return any((nums[i], nums[i + 1]) == qr for i in range(len(nums) - 1))
