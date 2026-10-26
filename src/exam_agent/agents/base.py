"""所有 Agent 共用的“结构化调用 + 校验反馈”循环。

流程：调用 LLM → 解析成 Pydantic 模型 → 跑业务校验 →
有问题就把具体错误作为新一轮 user 消息反馈给 LLM → 直到通过或达到最大次数。
停止条件是显式的：成功 / 达到 max_attempts（抛出 AgentFailure，由上层决定怎么办）。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm import LLMClient
from ..store import PaperStore

T = TypeVar("T", bound=BaseModel)


class AgentFailure(Exception):
    def __init__(self, step: str, problems: list[str]):
        super().__init__(f"{step} 在多次尝试后仍未通过校验：{problems}")
        self.step = step
        self.problems = problems


class UnsupportedRequest(AgentFailure):
    """超出系统支持范围的请求（如语文、八年级数学）。直接告诉老师支持什么，不去勉强出卷。"""
    def __init__(self, message: str):
        Exception.__init__(self, message)
        self.step, self.problems = "requirement", [message]

    def __str__(self) -> str:
        return self.args[0]


@dataclass
class Tracer:
    store: PaperStore | None = None
    run_id: str = field(default_factory=lambda: "r_" + uuid.uuid4().hex[:8])
    paper_id: str | None = None
    totals: dict[str, int] = field(default_factory=lambda: {"llm_calls": 0, "prompt_tokens": 0,
                                                             "completion_tokens": 0})
    listener: Callable[[str, dict], None] | None = None   # 网页用：把每一步实时推送给浏览器

    def log(self, step: str, **payload) -> None:
        if self.store is not None:
            self.store.log_event(self.run_id, self.paper_id, step, payload)
        if self.listener is not None:
            try:
                self.listener(step, payload)
            except Exception:
                pass   # 推送失败（如浏览器断开）不影响出卷本身


@dataclass
class StructuredResult(Generic[T]):
    value: T
    attempts: int


def structured_call(
    llm: LLMClient,
    *,
    step: str,
    system: str,
    user: str,
    schema: type[T],
    validate: Callable[[T], list[str]] | None = None,
    tracer: Tracer | None = None,
    max_attempts: int = 3,
) -> StructuredResult[T]:
    llm = llm.for_step(step) if hasattr(llm, "for_step") else llm   # 某些步骤可以用另一个模型
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        res = llm.chat_json(messages)
        elapsed = round(time.perf_counter() - t0, 2)
        if tracer:
            tracer.totals["llm_calls"] += 1
            tracer.totals["prompt_tokens"] += res.prompt_tokens
            tracer.totals["completion_tokens"] += res.completion_tokens

        problems = []
        value: T | None = None
        try:
            # strict=False：允许字符串里出现原始换行/制表符。模型在长解析里常这样输出，
            # 内容本身没错，没必要为此整批重来（实测一次这样的重试浪费约 5000 token、10 秒）
            value = schema.model_validate(json.loads(res.content, strict=False))
        except json.JSONDecodeError as e:
            problems = [f"输出不是合法 JSON：{e}"]
        except ValidationError as e:
            problems = [f"字段 {'.'.join(map(str, err['loc']))}：{err['msg']}" for err in e.errors()[:10]]
        if value is not None and validate is not None:
            problems = validate(value)

        if tracer:
            tracer.log(step, attempt=attempt, model=llm.model, seconds=elapsed,
                       prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                       problems=problems)
        if not problems and value is not None:
            return StructuredResult(value=value, attempts=attempt)

        messages.append({"role": "assistant", "content": res.content})
        messages.append({"role": "user", "content": (
            "你上一次的输出存在以下问题：\n- " + "\n- ".join(problems) +
            "\n请修正这些问题，并重新输出完整的 json 对象（不要只输出改动部分）。"
        )})
    raise AgentFailure(step, problems)
