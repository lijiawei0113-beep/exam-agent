"""新建试卷流程：需求理解 → 规划蓝图 → 命题（出错的题单独返修）→ 整卷校验 → 入库。"""
from __future__ import annotations

from dataclasses import dataclass

from .agents.base import AgentFailure, Tracer
from .agents.planner import plan_blueprint
from .agents.requirement import parse_request
from .agents.writer import fill_slots
from .domain import VStatus, PaperSnapshot
from .invariants import check_paper
from .llm import LLMClient
from .review import ReviewReport, review_and_repair, stamp_status
from .store import PaperStore, new_paper_id


@dataclass
class CreateResult:
    paper: PaperSnapshot
    version: int
    assumptions: list[str]
    run_id: str
    totals: dict[str, int]
    review: ReviewReport | None = None


def create_paper(llm: LLMClient, store: PaperStore, teacher_text: str, review: bool = True,
                 grounding=None, listener=None) -> CreateResult:
    """grounding：课程知识库（grounding.Grounding）；为 None 时不使用知识库，范围由模型判断。"""
    paper_id = new_paper_id()
    tracer = Tracer(store=store, paper_id=paper_id, listener=listener)
    tracer.log("input", text=teacher_text)

    draft = parse_request(llm, teacher_text, tracer)
    req = draft.request
    cands = grounding.candidates(req, teacher_text) if grounding is not None else None
    assumptions = list(draft.assumptions)
    if grounding is not None:
        assumptions.append(cands.how if cands else grounding.explain_missing(req))
    bp = plan_blueprint(llm, req, tracer, cands)
    questions = fill_slots(llm, req, bp.slots, tracer=tracer)
    flagged = {sid: q for sid, q in questions.items() if q.vstatus == VStatus.PROBLEM}   # 命题多轮仍未通过的题
    report = None
    if review:
        everything = set(questions)
        questions, report = review_and_repair(llm, req, bp, questions, verify=everything,
                                              pedagogy=everything, tracer=tracer)
    questions = stamp_status(questions, report, unreviewed=set(questions))
    for sid, q in flagged.items():   # 审核没有改动这道题：命题自查发现的问题仍然存在，不能被审核状态覆盖
        if questions[sid].content_fingerprint() == q.content_fingerprint():
            questions[sid] = questions[sid].model_copy(update={"verification": q.verification})
    still = sorted(s.position for s in bp.slots if questions[s.slot_id].vstatus == VStatus.PROBLEM)
    if still:
        assumptions.append(f"第 {'、'.join(map(str, still))} 题经多轮返修仍未通过检查，已标为“存在未解决问题”，"
                           "请检查，或直接说“第 N 题重出”")

    snap = PaperSnapshot(
        paper_id=paper_id,
        title=req.title,
        request=req, blueprint=bp, questions=[questions[s.slot_id] for s in bp.slots],
    )
    violations = check_paper(snap)
    tracer.log("final_check", violations=[str(v) for v in violations])
    if violations:
        raise AgentFailure("final_check", [str(v) for v in violations])

    version = store.create_paper(snap, summary=f"根据需求新建：{teacher_text[:40]}")
    store.add_turn(paper_id, "user", teacher_text, None)
    store.add_turn(paper_id, "assistant", f"已生成《{snap.title}》，共 {len(snap.questions)} 题", version)
    tracer.log("committed", version=version, totals=tracer.totals)
    return CreateResult(paper=snap, version=version, assumptions=assumptions,
                        run_id=tracer.run_id, totals=dict(tracer.totals), review=report)
