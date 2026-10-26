"""中文分词：字二元组（bigram）。

为什么不用 jieba
- 词典分词遇到错别字（“去分毋”“移相”）和教学术语（“系数化为1”）容易切错，切错就完全匹配不上；
  二元组只要有一半字对了还能命中。
- 无依赖、结果确定，测试可复现。代价是索引稍大、会有一些无意义的二元组（由 IDF 自动压低权重）。
"""
from __future__ import annotations

import re
import unicodedata

_CJK = re.compile(r"[一-鿿]+")
_WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[√|≤≥<>=+\-×÷²]")


def normalize(text: str) -> str:
    # NFKC：全角字母数字转半角，“ｘ”“１”与“x”“1”一致
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    t = normalize(text)
    tokens: list[str] = []
    for run in _CJK.findall(t):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    tokens.extend(_WORD.findall(_CJK.sub(" ", t)))
    return tokens
