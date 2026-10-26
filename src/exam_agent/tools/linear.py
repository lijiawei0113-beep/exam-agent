"""线性方程（组）的精确求解器：验题工具。

为什么自己写而不是直接用 SymPy：
- 当前题目范围是一元一次方程和二元一次方程组，线性求解足够；
- 全程用 Fraction 精确运算，12/7 不会变成 1.7142857；
- 不认识的形式（未知数在分母、二次项）明确抛出 NotLinear，而不是给出可疑结果——
  工具“不知道”时必须说不知道，交给其他审核手段。
需要支持非线性题目时，可以在同一接口后面换成 SymPy。
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from fractions import Fraction


class ToolError(Exception):
    """工具无法给出可靠结论（解析失败、非线性、无解或无穷多解）。"""


class NotLinear(ToolError):
    pass


@dataclass
class Lin:
    coeffs: dict[str, Fraction] = field(default_factory=dict)
    const: Fraction = Fraction(0)

    @property
    def is_const(self) -> bool:
        return not any(self.coeffs.values())

    def scale(self, k: Fraction) -> "Lin":
        return Lin({v: c * k for v, c in self.coeffs.items()}, self.const * k)

    def add(self, other: "Lin", sign: int = 1) -> "Lin":
        coeffs = dict(self.coeffs)
        for v, c in other.coeffs.items():
            coeffs[v] = coeffs.get(v, Fraction(0)) + sign * c
        return Lin(coeffs, self.const + sign * other.const)


_REPLACE = {
    "（": "(", "）": ")", "［": "(", "］": ")", "[": "(", "]": ")", "＋": "+", "－": "-", "−": "-",
    "×": "*", "·": "*", "÷": "/", "＝": "=", "²": "**2", "³": "**3", "^": "**", " ": "", "　": "",
}


def normalize(expr: str) -> str:
    for a, b in _REPLACE.items():
        expr = expr.replace(a, b)
    # 隐式乘法：2x、3(x-1)、(x+1)(x-1)、x(x+1)、)2
    expr = re.sub(r"(?<=[\d)])(?=[A-Za-z(])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z])(?=\()", "*", expr)
    expr = re.sub(r"(?<=\))(?=\d)", "*", expr)
    return expr


def _to_lin(node: ast.AST) -> Lin:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Lin(const=Fraction(repr(node.value)))  # repr 保证 0.3 → 3/10 精确
    if isinstance(node, ast.Name):
        return Lin(coeffs={node.id: Fraction(1)})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _to_lin(node.operand)
        return v.scale(Fraction(-1)) if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _to_lin(node.left), _to_lin(node.right)
        if isinstance(node.op, ast.Add):
            return a.add(b)
        if isinstance(node.op, ast.Sub):
            return a.add(b, -1)
        if isinstance(node.op, ast.Mult):
            if a.is_const:
                return b.scale(a.const)
            if b.is_const:
                return a.scale(b.const)
            raise NotLinear("出现了未知数相乘")
        if isinstance(node.op, ast.Div):
            if not b.is_const:
                raise NotLinear("未知数出现在分母中")
            if b.const == 0:
                raise ToolError("除数为 0")
            return a.scale(1 / b.const)
        if isinstance(node.op, ast.Pow):
            if not b.is_const or b.const.denominator != 1:
                raise NotLinear("指数不是整数常数")
            e = int(b.const)
            if a.is_const:
                return Lin(const=a.const ** e)
            if e == 1:
                return a
            if e == 0:
                return Lin(const=Fraction(1))
            raise NotLinear("出现了未知数的高次项")
    raise ToolError(f"无法识别的表达式：{ast.dump(node)[:60]}")


def parse_linear(expr: str) -> Lin:
    try:
        tree = ast.parse(normalize(expr), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"无法解析表达式“{expr}”") from e
    return _to_lin(tree.body)


def solve(equations: list[str], unknowns: list[str]) -> dict[str, Fraction]:
    """求解线性方程组，返回每个未知数的精确值。无唯一解时抛出 ToolError。"""
    if not equations:
        raise ToolError("没有方程")
    rows: list[Lin] = []
    for eq in equations:
        parts = normalize(eq).split("=")
        if len(parts) != 2:
            raise ToolError(f"方程必须恰好有一个等号：“{eq}”")
        rows.append(parse_linear(parts[0]).add(parse_linear(parts[1]), -1))

    symbols = {v for r in rows for v, c in r.coeffs.items() if c != 0}
    unknowns = list(dict.fromkeys(unknowns)) or sorted(symbols)
    extra = symbols - set(unknowns)
    if extra:
        raise ToolError(f"方程中出现了未声明的字母：{sorted(extra)}")

    # 高斯消元：矩阵 [A | b]，其中 A·x = -const
    m = [[r.coeffs.get(v, Fraction(0)) for v in unknowns] + [-r.const] for r in rows]
    n = len(unknowns)
    pivot_row = 0
    pivots = []
    for col in range(n):
        sel = next((i for i in range(pivot_row, len(m)) if m[i][col] != 0), None)
        if sel is None:
            continue
        m[pivot_row], m[sel] = m[sel], m[pivot_row]
        p = m[pivot_row][col]
        m[pivot_row] = [x / p for x in m[pivot_row]]
        for i in range(len(m)):
            if i != pivot_row and m[i][col] != 0:
                k = m[i][col]
                m[i] = [a - k * b for a, b in zip(m[i], m[pivot_row])]
        pivots.append(col)
        pivot_row += 1
    for row in m[pivot_row:]:
        if all(x == 0 for x in row[:n]) and row[n] != 0:
            raise ToolError("方程（组）无解")
    if len(pivots) < n:
        raise ToolError("方程（组）有无穷多解，条件不足")
    return {unknowns[col]: m[i][n] for i, col in enumerate(pivots)}


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_numbers(text: str) -> list[Fraction]:
    """从答案文字中提取数值：'x = -6'、'15支'、'1.5小时'、'x = 22/5'。"""
    text = normalize(text).replace("*", " ")
    out = []
    for tok in _NUM.findall(text):
        try:
            out.append(Fraction(tok))
        except (ValueError, ZeroDivisionError):
            continue
    return out


def fmt(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
