"""网页服务的业务层：不依赖 Web 框架，可以直接测试。

- 出卷 / 修改在后台线程中运行，每一步通过 Tracer 的 listener 实时产出进度事件（供 SSE 推送给浏览器）；
- 试卷序列化成前端需要的结构（分大题、验证状态、蓝图、历史）；
- 公开访问时的成本保护：同时进行的出卷数上限、每个 IP 每小时次数上限（环境变量可调）。
"""
from __future__ import annotations

import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from ..domain import PaperSnapshot, VStatus
from ..store import PaperStore

STEP_LABEL = {
    "input": "收到出卷要求",
    "edit_input": "收到修改要求",
    "requirement": "理解需求（年级、册别、题型、题数）",
    "grounding": "检索课程知识库，确定考查范围",
    "planner": "规划蓝图（双向细目表）",
    "edit_planner": "理解修改意图",
    "writer": "命题",
    "writer_expand": "扩写阅读短文（字数不足）",
    "writer_check": "命题自查（格式、工具验算、年级规则）",
    "blind_solver": "盲解 Agent 独立作答（看不到答案）",
    "tool_check": "工具核验答案",
    "pedagogy_reviewer": "教学审核",
    "review_findings": "汇总审核意见",
    "review_done": "审核结束",
    "edit_check": "检查未要求修改的题是否保持不变",
    "final_check": "整卷校验",
    "committed": "保存为新版本",
}

STATUS_KEY = {VStatus.TOOL: "tool", VStatus.AGREED: "agreed", VStatus.UNVERIFIED: "unverified",
              VStatus.PROBLEM: "problem", VStatus.TEACHER: "teacher", VStatus.NOT_REVIEWED: "none"}


def progress_event(step: str, payload: dict) -> dict | None:
    """把内部轨迹事件变成给老师看的一句话；不重要的事件返回 None。"""
    if step == "edit_agent_step":   # ReAct 每一步：查了什么
        detail = "；".join(map(str, payload.get("problems") or []))[:120]
        return {"type": "progress", "step": step, "label": payload.get("label", "查阅试卷"),
                "detail": ("方案被程序退回：" + detail) if detail else ""}
    label = STEP_LABEL.get(step)
    if label is None:
        return None
    detail = ""
    if "attempt" in payload and payload.get("attempt", 1) > 1:
        detail = f"第 {payload['attempt']} 次尝试"
    if payload.get("problems"):
        detail = "发现问题，带着具体意见重试：" + "；".join(map(str, payload["problems"]))[:120]
    if step == "grounding":
        detail = payload.get("how", "")
    if step == "review_findings":
        must = [f for f in payload.get("findings", []) if f.get("severity") == "must_fix"]
        detail = f"必须修改 {len(must)} 处" if must else "没有必须修改的问题"
    if step == "review_done":
        detail = {"passed": "全部通过", "max_rounds": "达到最大返修轮数", "no_progress": "返修没有进展，停止",
                  "repair_failed": "返修失败"}.get(payload.get("stop_reason", ""), "")
    if step == "committed":
        detail = f"v{payload.get('version')}"
    return {"type": "progress", "step": step, "label": label, "detail": detail}


def paper_view(paper: PaperSnapshot, version: int, store: PaperStore | None = None) -> dict[str, Any]:
    """前端需要的试卷结构。题号、大题标题与 PDF 完全一致（复用同一套编排函数）。"""
    from ..export_pdf import _section_title, _sections

    slots = {s.slot_id: s for s in paper.blueprint.slots}
    sections = []
    for i, (qtype, items) in enumerate(_sections(paper)):
        sections.append({"title": _section_title(i, qtype, items), "questions": [{
            "no": no, "slot_id": q.slot_id, "qtype": q.qtype.value, "score": q.score, "stem": q.stem,
            "options": q.options, "answer": q.answer, "analysis": q.analysis,
            "knowledge_point": q.knowledge_point, "difficulty": q.difficulty.value,
            "kp_id": slots[q.slot_id].kp_id if q.slot_id in slots else "",
            "status": q.vstatus.value, "status_key": STATUS_KEY[q.vstatus],
            "method": q.verification.method if q.verification else "",
            "detail": q.verification.detail if q.verification else "",
        } for no, q in items]})
    counts: dict[str, int] = {}
    for q in paper.questions:
        counts[q.vstatus.value] = counts.get(q.vstatus.value, 0) + 1
    r = paper.request
    view = {
        "paper_id": paper.paper_id, "title": paper.title, "version": version,
        "request": {"subject": r.subject, "scope": r.scope_text, "question_count": r.question_count,
                    "total_score": r.total_score, "constraints": [c.text for c in r.constraints]},
        "scope": paper.blueprint.scope, "status_counts": counts, "sections": sections,
        "blueprint": [{"position": s.position, "qtype": s.qtype.value, "score": s.score,
                       "knowledge_point": s.knowledge_point, "difficulty": s.difficulty.value,
                       "kp_id": s.kp_id} for s in sorted(paper.blueprint.slots, key=lambda s: s.position)],
    }
    if store is not None:
        view["history"] = store.history(paper.paper_id)
    return view


def review_view(report) -> dict | None:
    if report is None:
        return None
    return {"rounds": report.rounds, "stop_reason": report.stop_reason,
            "initial": [f.model_dump() for f in report.initial if f.severity == "must_fix"],
            "remaining": [f.model_dump() for f in report.remaining],
            "suggestions": [f.model_dump() for f in report.suggestions],
            "repaired": sorted(report.repaired)}


class RateLimited(Exception):
    pass


@dataclass
class Limiter:
    """公开访问时的成本保护。数值来自环境变量，设为 0 表示不限制。"""
    per_hour: int = int(os.getenv("EXAM_WEB_PER_HOUR", "30"))
    concurrency: int = int(os.getenv("EXAM_WEB_CONCURRENCY", "3"))
    _hits: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    _running: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self, client: str) -> None:
        now = time.time()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 3600:
                hits.popleft()
            if self.per_hour and len(hits) >= self.per_hour:
                raise RateLimited(f"这个网络地址一小时内已经使用 {len(hits)} 次，请稍后再试")
            if self.concurrency and self._running >= self.concurrency:
                raise RateLimited("当前使用的人较多，请过一会儿再试")
            hits.append(now)
            self._running += 1

    def release(self) -> None:
        with self._lock:
            self._running = max(0, self._running - 1)


class ExamService:
    def __init__(self, store: PaperStore, llm_factory: Callable[[], Any], grounding=None,
                 limiter: Limiter | None = None, review: bool = True, out_dir: Path | None = None):
        self.store = store
        self.llm_factory = llm_factory
        self.grounding = grounding
        self.limiter = limiter or Limiter()
        self.review = review
        self.out_dir = out_dir
        self._paper_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    # ---------- 后台运行 + 进度事件 ----------
    def _stream(self, work: Callable[[Callable[[str, dict], None]], dict], client: str) -> Iterator[dict]:
        """在后台线程运行 work，边运行边产出进度事件，最后产出 result 或 error 事件。"""
        try:
            self.limiter.acquire(client)
        except RateLimited as e:
            yield {"type": "error", "message": str(e)}
            return
        events: queue.Queue = queue.Queue()

        def listener(step: str, payload: dict) -> None:
            ev = progress_event(step, payload)
            if ev:
                events.put(ev)

        def run() -> None:
            try:
                events.put({"type": "result", **work(listener)})
            except Exception as e:  # 失败如实告诉老师，试卷保持原样
                from ..agents.base import AgentFailure, UnsupportedRequest

                if isinstance(e, UnsupportedRequest):
                    events.put({"type": "error", "message": f"{e}。"})
                else:
                    msg = str(e) if isinstance(e, AgentFailure) else f"{type(e).__name__}: {e}"
                    events.put({"type": "error", "message": f"这次没能完成：{msg}。试卷保持原样，可以换个说法再试。"})
            finally:
                self.limiter.release()
                events.put(None)

        threading.Thread(target=run, daemon=True).start()
        while True:
            ev = events.get()
            if ev is None:
                return
            yield ev

    def create(self, text: str, client: str = "local") -> Iterator[dict]:
        from ..pipeline import create_paper

        def work(listener):
            res = create_paper(self.llm_factory(), self.store, text, review=self.review,
                               grounding=self.grounding, listener=listener)
            return {"kind": "created", "assumptions": res.assumptions, "review": review_view(res.review),
                    "totals": res.totals, "paper": paper_view(res.paper, res.version, self.store)}
        return self._stream(work, client)

    def edit(self, paper_id: str, text: str, client: str = "local") -> Iterator[dict]:
        from ..editing import apply_edit

        def work(listener):
            with self._paper_locks[paper_id]:   # 同一份试卷的修改排队执行
                res = apply_edit(self.llm_factory(), self.store, paper_id, text, review=self.review,
                                 listener=listener, grounding=self.grounding)
            return {"kind": res.action, "reply": res.reply, "warnings": res.warnings,
                    "changes": [c.model_dump() for c in res.changes], "review": review_view(res.review),
                    "totals": res.totals, "paper": paper_view(res.paper, res.version, self.store)}
        return self._stream(work, client)

    # ---------- 不需要模型的操作 ----------
    def get(self, paper_id: str, version: int | None = None) -> dict:
        paper, v = self.store.load(paper_id, version)
        return paper_view(paper, v, self.store)

    def list(self) -> list[dict]:
        return self.store.list_papers()

    def confirm(self, paper_id: str, positions: list[int], note: str = "") -> dict:
        from ..editing import confirm

        with self._paper_locks[paper_id]:
            paper, v, _ = confirm(self.store, paper_id, positions, note)
        return paper_view(paper, v, self.store)

    def undo(self, paper_id: str) -> dict:
        from ..editing import undo

        with self._paper_locks[paper_id]:
            _, _, nv = undo(self.store, paper_id)
        return self.get(paper_id)

    def delete(self, paper_id: str) -> dict:
        with self._paper_locks[paper_id]:   # 正在修改的试卷，等修改结束再删
            self.store.delete_paper(paper_id)
        self._remove_pdfs(paper_id)
        return {"deleted": paper_id}

    def clear_all(self) -> dict:
        ids = [r["paper_id"] for r in self.store.list_papers()]
        n = self.store.clear_all()
        for pid in ids:
            self._remove_pdfs(pid)
        return {"deleted_count": n}

    def _remove_pdfs(self, paper_id: str) -> None:
        from ..export_pdf import OUTPUT_DIR

        out = self.out_dir or OUTPUT_DIR
        if out.exists():
            for f in out.glob(f"{paper_id}_v*.pdf"):
                f.unlink(missing_ok=True)

    def export(self, paper_id: str, version: int | None = None) -> tuple[Path, Path]:
        from ..export_pdf import OUTPUT_DIR, export_paper

        paper, v = self.store.load(paper_id, version)
        return export_paper(paper, v, self.out_dir or OUTPUT_DIR)
