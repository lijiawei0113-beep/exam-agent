"""修改 / 讨论评测：一次调用式规划（oneshot，只看摘要）对比 ReAct（按需查阅试卷、历史和知识库）。

数据集
  evals/edit_eval/papers.json   两份试卷（三年级上册数学，含 3 个版本；五年级上册英语）
  evals/edit_cases.jsonl        31 条老师的话，标注期望的动作、操作类型、题号（恢复类还标注版本），
                                问答类标注回答中必须出现的关键词（每组至少命中一个）
只评测“规划”这一步（理解老师的话 → 方案 / 回答），不调用命题员，因此便宜、可重复。
指标
  正确率     动作正确，且（修改类）操作与题号正确 /（问答类）关键词全部命中
  工具调用   ReAct 平均每条调用了几次只读工具
  调用 / token / 耗时  每条平均
按标签分组：explicit 题号明确 | content 按内容指代 | patch 局部改字 | history 历史版本 | qa 题目细节问答 |
           stats 全卷统计 | scope 是否超纲 | discuss 开放讨论 | clarify 应当追问 | followup 同意上一轮建议
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ..agents.base import AgentFailure, Tracer
from ..domain import PaperSnapshot
from ..config import PROJECT_ROOT
from .review_eval import RESULTS_DIR, git_commit

PAPERS = PROJECT_ROOT / "evals" / "edit_eval" / "papers.json"
DATASET = PROJECT_ROOT / "evals" / "edit_cases.jsonl"
PLANNERS = ["oneshot", "react"]
PLANNER_LABEL = {"oneshot": "一次调用（只看摘要）", "react": "ReAct（按需查阅）"}
TAG_LABEL = {"explicit": "题号明确", "content": "按内容指代", "patch": "局部改字", "history": "历史版本",
             "qa": "题目细节问答", "stats": "全卷统计", "scope": "是否超纲", "discuss": "开放讨论",
             "clarify": "应当追问", "followup": "同意上一轮建议"}


class ECase(BaseModel):
    id: str
    paper: str
    tag: str
    message: str
    action: str
    ops: list[str] = Field(default_factory=list)
    positions: list[int] = Field(default_factory=list)
    from_version: int | None = None
    keywords: list[list[str]] = Field(default_factory=list)
    turns: list[dict] = Field(default_factory=list)
    split: str = "dev"      # dev：开发集（看过错误、调过提示词）；holdout：留出集（只在最后评测，不据此调参）


def load_ecases(path: Path = DATASET) -> list[ECase]:
    with open(path, encoding="utf-8") as f:
        return [ECase(**json.loads(line)) for line in f if line.strip()]


def make_store(paper_id: str, turns: list[dict] | None = None):
    """建一个内存数据库，按顺序写入这份试卷的各个版本（和对话），模拟老师已经改过几次的状态。"""
    from ..store import PaperStore

    store = PaperStore(":memory:")
    versions = json.loads(PAPERS.read_text(encoding="utf-8"))[paper_id]
    for i, v in enumerate(versions):
        snap = PaperSnapshot.model_validate(v["snapshot"])
        if i == 0:
            store.create_paper(snap, v["summary"])
        else:
            store.commit_version(snap, expected_version=i, operation=v["operation"], summary=v["summary"])
    cur = store.current_version(paper_id)
    for t in turns or []:
        store.add_turn(paper_id, t["role"], t["content"], cur)
    return store


@dataclass
class EResult:
    case_id: str
    tag: str
    planner: str
    ok: bool
    action: str
    ops: list[dict] = field(default_factory=list)
    reply: str = ""
    reason: str = ""
    tool_calls: int = 0
    llm_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0


def judge(case: ECase, plan, same_content=None) -> tuple[bool, str]:
    """same_content(position, v1, v2)：两个版本中这道题内容是否相同（恢复到内容相同的版本也算对）。"""
    if plan.action != case.action:
        return False, f"动作应为 {case.action}，实际 {plan.action}"
    if case.action == "edit":
        if not plan.ops:
            return False, "没有操作"
        op = plan.ops[0]
        if op.op not in case.ops:
            return False, f"操作应为 {case.ops}，实际 {op.op}"
        if case.positions and sorted(op.positions) != sorted(case.positions):
            return False, f"题号应为 {case.positions}，实际 {op.positions}"
        if case.from_version is not None and op.from_version != case.from_version and not (
                same_content and op.from_version and same_content(case.positions[0], op.from_version, case.from_version)):
            return False, f"应恢复到 v{case.from_version}，实际 v{op.from_version}"
    for group in case.keywords:
        if not any(k.lower() in plan.reply.lower() for k in group):
            return False, f"回答中缺少 {group}"
    return True, ""


def run_one(llm, case: ECase, planner: str, grounding=None) -> EResult:
    from ..agents.edit_agent import EditWorkspace, run_edit_agent
    from ..agents.editor import plan_edit
    from ..editing import apply_ops

    store = make_store(case.paper, case.turns)
    paper, version = store.load(case.paper)
    load_version = lambda v: store.load(case.paper, v)[0]

    def validate(plan):
        if plan.action != "edit":
            return []
        return ["action 为 edit 时必须给出至少一个操作"] if not plan.ops else apply_ops(paper, plan, load_version).problems

    tracer = Tracer(store=store, paper_id=case.paper)
    t0 = time.time()
    try:
        if planner == "oneshot":
            plan = plan_edit(llm, paper, case.message, history=store.recent_turns(case.paper, 6), validate=validate,
                             tracer=tracer)
        else:
            ws = EditWorkspace(store=store, paper_id=case.paper, paper=paper, version=version, grounding=grounding)
            plan = run_edit_agent(llm, ws, case.message, validate=validate, tracer=tracer)
    except AgentFailure as e:
        return EResult(case.id, case.tag, planner, False, "failure", reason=str(e)[:200],
                       llm_calls=tracer.totals["llm_calls"], seconds=time.time() - t0)
    def same_content(pos, a, b):
        sid = paper.blueprint.slot_at(pos).slot_id
        try:
            qa = next(q for q in load_version(a).questions if q.slot_id == sid)
            qb = next(q for q in load_version(b).questions if q.slot_id == sid)
        except (KeyError, StopIteration):
            return False
        return qa.content_fingerprint() == qb.content_fingerprint()

    ok, reason = judge(case, plan, same_content)
    steps = [e for e in store.conn.execute("SELECT step, payload FROM trace_events WHERE step='edit_agent_step'")]
    tools = sum(1 for s in steps if json.loads(s["payload"]).get("tool") not in ("submit_plan", "reply"))
    return EResult(case.id, case.tag, planner, ok, plan.action,
                   [o.model_dump(mode="json", exclude_defaults=True) for o in plan.ops], plan.reply[:300], reason,
                   tools, tracer.totals["llm_calls"], tracer.totals["prompt_tokens"] + tracer.totals["completion_tokens"],
                   round(time.time() - t0, 2))


def summarize(results: list[EResult]) -> dict:
    out: dict = {}
    for p in PLANNERS:
        rs = [r for r in results if r.planner == p]
        if not rs:
            continue
        n = len(rs)
        tags = {}
        for t in TAG_LABEL:
            sub = [r for r in rs if r.tag == t]
            if sub:
                tags[t] = sum(r.ok for r in sub) / len(sub)
        out[p] = {"n": n, "accuracy": sum(r.ok for r in rs) / n, "by_tag": tags,
                  "tool_calls": sum(r.tool_calls for r in rs) / n, "llm_calls": sum(r.llm_calls for r in rs) / n,
                  "tokens": sum(r.tokens for r in rs) / n, "seconds": sum(r.seconds for r in rs) / n}
    return out


def run_edit_eval(llm, planners: list[str] | None = None, repeat: int = 1, grounding=None,
                  case_ids: list[str] | None = None, progress=None, split: str = "dev") -> tuple[dict, tuple[Path, Path]]:
    cases = [c for c in load_ecases() if (not case_ids or c.id in case_ids) and (split == "all" or c.split == split)]
    results: list[EResult] = []
    for p in planners or PLANNERS:
        for _ in range(repeat):
            for c in cases:
                r = run_one(llm, c, p, grounding)
                results.append(r)
                if progress:
                    progress(r)
    summary = summarize(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"stamp": stamp, "commit": git_commit(), "model": getattr(llm, "model", "?"), "repeat": repeat,
            "cases": len(cases), "split": split}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = RESULTS_DIR / f"edit_eval_{stamp}.json"
    jp.write_text(json.dumps({"meta": meta, "summary": summary, "results": [r.__dict__ for r in results]},
                             ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# 修改 / 讨论评测 {stamp}", "", f"代码版本 `{meta['commit']}`，模型 {meta['model']}，数据集 {split}，{len(cases)} 条 × {repeat} 次", "",
          "| 方式 | 正确率 | 工具调用/条 | 模型调用/条 | token/条 | 耗时秒/条 |", "|---|---|---|---|---|---|"]
    for p, s in summary.items():
        md.append(f"| {PLANNER_LABEL[p]} | {s['accuracy']:.0%} | {s['tool_calls']:.1f} | {s['llm_calls']:.1f} | "
                  f"{s['tokens']:.0f} | {s['seconds']:.1f} |")
    md += ["", "| 类别 | " + " | ".join(PLANNER_LABEL[p] for p in summary) + " |",
           "|---|" + "---|" * len(summary)]
    for t, label in TAG_LABEL.items():
        if any(t in s["by_tag"] for s in summary.values()):
            md.append(f"| {label} | " + " | ".join(f"{s['by_tag'].get(t, 0):.0%}" for s in summary.values()) + " |")
    md += ["", "## 错误明细", ""]
    md += [f"- {r.planner} {r.case_id}（{TAG_LABEL[r.tag]}）：{r.reason}；回复：{r.reply[:80]}" for r in results if not r.ok]
    mp = RESULTS_DIR / f"edit_eval_{stamp}.md"
    mp.write_text("\n".join(md), encoding="utf-8")
    return summary, (jp, mp)
