"""LLM 客户端抽象。

业务代码只依赖 LLMClient 协议，因此测试时可以换成 ScriptedLLM，不花钱、结果可复现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings


@dataclass
class LLMResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # 模型给出的 JSON 字符串，由调用方解析（解析失败也要能反馈给模型）


@dataclass
class ToolTurn:
    """一次带工具的调用结果：要么调用若干工具，要么直接回复文字。"""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def as_message(self) -> dict:
        """写回对话历史的 assistant 消息（OpenAI 格式）。"""
        m: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            m["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls]
        return m


class LLMClient(Protocol):
    model: str

    def chat_json(self, messages: list[dict]) -> LLMResult: ...

    # 可选：原生函数调用（ReAct 修改规划员使用）
    # def chat_tools(self, messages: list[dict], tools: list[dict]) -> ToolTurn: ...


class DeepSeekClient:
    """DeepSeek 使用 OpenAI 兼容接口。JSON 模式要求 prompt 中出现 json 字样并给出示例。"""

    def __init__(self, settings: Settings, timeout: float = 120.0):
        if not settings.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填写")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                              timeout=timeout, max_retries=2)
        self.model = settings.model
        self.thinking = settings.thinking

    def chat_json(self, messages: list[dict]) -> LLMResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=8000,
            extra_body={"thinking": {"type": self.thinking}},
        )
        usage = resp.usage
        cached = 0
        if usage is not None:
            cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        return LLMResult(
            content=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=cached,
        )

    def chat_tools(self, messages: list[dict], tools: list[dict], tool_choice: str | dict = "auto") -> ToolTurn:
        """DeepSeek 原生函数调用（OpenAI 兼容的 tools 参数）。"""
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, tool_choice=tool_choice, max_tokens=4000,
            extra_body={"thinking": {"type": self.thinking}},
        )
        msg, usage = resp.choices[0].message, resp.usage
        calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (msg.tool_calls or [])]
        return ToolTurn(content=msg.content or "", tool_calls=calls,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        cached_tokens=getattr(usage, "prompt_cache_hit_tokens", 0) or 0)


def tool_reply(*calls: tuple[str, dict]) -> str:
    """测试用：构造一条“调用这些工具”的预设回复，供 ScriptedLLM.chat_tools 使用。"""
    import json

    return json.dumps({"__tool_calls__": [{"name": n, "arguments": a} for n, a in calls]}, ensure_ascii=False)


@dataclass
class ScriptedLLM:
    """按顺序返回预设回复的假 LLM，并记录收到的所有消息，便于断言。"""
    replies: list[str]
    model: str = "scripted"
    calls: list[list[dict]] = field(default_factory=list)

    def chat_json(self, messages: list[dict]) -> LLMResult:
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError("ScriptedLLM 的预设回复已用完")
        return LLMResult(content=self.replies.pop(0), prompt_tokens=100, completion_tokens=50)

    def chat_tools(self, messages: list[dict], tools: list[dict], tool_choice: str | dict = "auto") -> ToolTurn:
        """预设回复的三种形式：tool_reply(...) 构造的工具调用；一个 EditPlan JSON（视为直接调用 submit_plan，
        兼容一次调用式规划的旧测试）；其他文字（视为直接回复老师）。"""
        import json

        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError("ScriptedLLM 的预设回复已用完")
        raw = self.replies.pop(0)
        self._n = getattr(self, "_n", 0)
        try:
            obj = json.loads(raw)
        except ValueError:
            obj = None
        calls = []
        if isinstance(obj, dict) and "__tool_calls__" in obj:
            calls = [(c["name"], json.dumps(c["arguments"], ensure_ascii=False)) for c in obj["__tool_calls__"]]
        elif isinstance(obj, dict) and "action" in obj:
            calls = [("submit_plan", raw)]
        elif obj is not None:
            calls = [("submit_plan", raw)]       # 其他 JSON（如格式错误的方案）也按提交处理，便于测反馈
        out = []
        for name, args in calls:
            self._n += 1
            out.append(ToolCall(f"call_{self._n}", name, args))
        return ToolTurn(content="" if calls else raw, tool_calls=out, prompt_tokens=100, completion_tokens=50)


@dataclass
class RoutedLLM:
    """按步骤分派到不同模型。用途：盲解 Agent 换一个与命题员不同的模型，降低“两次答错得一样”的相关错误（d019）。"""
    default: LLMClient
    routes: dict[str, LLMClient] = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.default.model

    def for_step(self, step: str) -> LLMClient:
        return self.routes.get(step, self.default)

    def chat_json(self, messages: list[dict]) -> LLMResult:
        return self.default.chat_json(messages)

    def chat_tools(self, messages: list[dict], tools: list[dict], tool_choice: str | dict = "auto") -> ToolTurn:
        return self.for_step("edit_agent").chat_tools(messages, tools, tool_choice)

    def describe(self) -> str:
        extra = "，".join(f"{k}={getattr(v, 'model', '?')}" + (f"（thinking {v.thinking}）" if hasattr(v, "thinking") else "")
                         for k, v in self.routes.items())
        return self.default.model + (f"；{extra}" if extra else "")


def make_llm(settings: Settings) -> LLMClient:
    """默认所有步骤用同一个模型；.env 设置了 EXAM_BLIND_MODEL / EXAM_BLIND_THINKING 时，盲解单独用另一个配置。"""
    from dataclasses import replace

    default = DeepSeekClient(settings)
    model = settings.blind_model or settings.model
    thinking = settings.blind_thinking or settings.thinking
    if (model, thinking) == (settings.model, settings.thinking):
        return default
    return RoutedLLM(default, {"blind_solver": DeepSeekClient(replace(settings, model=model, thinking=thinking))})
