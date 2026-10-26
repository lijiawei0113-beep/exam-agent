"""对话式局部修改。

一次修改的完整流程：
  1. 从数据库读取当前版本（唯一真相），不依赖对话记忆。
  2. 修改规划员把老师的话翻译成 EditPlan。
  3. apply_ops 以纯函数方式“试运行”这些操作：定位题号、改蓝图、重新编号、重配分值、校验。
     试运行有问题 → 把具体问题反馈给规划员重新规划（这就是规划阶段的反馈循环）。
  4. 只把需要重出的 slot 交给命题员，其他题根本不进入命题上下文。
  5. 组装新版本 → 整卷校验 + 内容指纹校验（证明其他题没被动过）→ 乐观并发提交。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agents.base import AgentFailure, Tracer
from .agents.editor import EditPlan, plan_edit
from .agents.writer import fill_slots
from .diffing import SlotChange, diff_snapshots
from .domain import (Verification, VStatus, Blueprint, BlueprintSlot, CognitiveLevel, ConstraintKind, Difficulty, ExamRequest,
                     PaperSnapshot, QType, Question)
from .invariants import check_blueprint, check_paper, check_preserved
from .llm import LLMClient
from .scoring import DEFAULT_SCORE, current_scores, rebalance
from .review import ReviewReport, review_and_repair, stamp_status
from .store import PaperStore

VERIFIABLE = {ConstraintKind.INTEGER_ANSWER}


@dataclass
class AppliedOps:
    request: ExamRequest
    blueprint: Blueprint
    regenerate: set[str] = field(default_factory=set)   # 需要重新出题的 slot
    removed: set[str] = field(default_factory=set)
    notes: dict[str, str] = field(default_factory=dict)  # 给命题员的逐题要求
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)    # 需要如实告诉老师的事
    new_constraints: list = field(default_factory=list)   # 本次新增的约束（需要回溯核验现有题）
    restored: dict[str, Question] = field(default_factory=dict)   # 从旧版本原样恢复的题
    patched: dict[str, Question] = field(default_factory=dict)    # 局部改了文字的题（需要重新核验）


def _strip_letter(text: str) -> str:
    import re

    return re.sub(r"^\s*[A-D]\s*[.、．:：)）]\s*", "", text)


def _apply_patches(q: Question, patches, tag: str, problems: list[str]) -> Question | None:
    """逐处替换文字。原文必须恰好出现一次，否则报告给规划员（避免改错地方）。"""
    data = {"stem": q.stem, "answer": q.answer, "analysis": q.analysis, "options": list(q.options)}
    for p in patches:
        if p.field == "options":   # 模型常照着 get_question 的显示带上“D. ”前缀（e1：e10、m08 因此失败）
            p = p.model_copy(update={"old": _strip_letter(p.old), "new": _strip_letter(p.new)})
        if not p.old:
            problems.append(f"{tag} 替换的原文不能为空")
            return None
        if p.field == "options":
            hits = [i for i, o in enumerate(data["options"]) if p.old in o]
            count = sum(o.count(p.old) for o in data["options"])
            if count != 1:
                problems.append(f"{tag} 选项中“{p.old}”出现 {count} 次，必须恰好 1 次；请先用 get_question 查看原文")
                return None
            data["options"][hits[0]] = data["options"][hits[0]].replace(p.old, p.new)
        else:
            count = data[p.field].count(p.old)
            if count != 1:
                problems.append(f"{tag} {p.field} 中“{p.old[:30]}”出现 {count} 次，必须恰好 1 次；请先查看原文，"
                                "给出能唯一定位的一段")
                return None
            data[p.field] = data[p.field].replace(p.old, p.new)
    if q.qtype == QType.CHOICE:
        data["answer"] = data["answer"].strip().upper()
        if data["answer"] not in "ABCD"[:len(data["options"])] or len(data["answer"]) != 1:
            problems.append(f"{tag} 选择题答案必须是一个选项字母")
            return None
    return q.model_copy(update={**data, "verification": None})


def _add_slot(bp, req, out, qtype, kp: str, difficulty=None, instruction: str = "") -> None:
    sid = bp.next_slot_id()
    bp.slots.append(BlueprintSlot(
        slot_id=sid, position=10**6, qtype=qtype, score=DEFAULT_SCORE[qtype],
        knowledge_point=kp, difficulty=difficulty or Difficulty.MEDIUM,
        cognitive_level=CognitiveLevel.APPLY if qtype == QType.APPLICATION else CognitiveLevel.MASTER,
        intent=(instruction or kp)[:40],
    ))
    req.question_count += 1
    out.regenerate.add(sid)
    out.notes[sid] = f"教师要求：{instruction or '新增一道' + qtype.value}"


def _resize(bp, req, out, op, tag: str) -> None:
    """“扩大到 20 题，原题保留”：新增几道、每种题型各几道由代码算，不让模型数（d042：模型逐条写 add，
    要补 15 道却写了 20 道，试卷变成 25 题）。只支持扩大；缩小要由老师或模型指定删哪几道。"""
    from .agents.planner import max_counts
    from .domain import DEFAULT_TYPES, allowed_types
    from .scoring import feasible_count_options

    n = op.count or 0
    have: dict[QType, int] = {}
    for s in bp.slots:
        have[s.qtype] = have.get(s.qtype, 0) + 1
    if n <= len(bp.slots):
        out.problems.append(f"{tag} resize 只用于扩大试卷：现在已有 {len(bp.slots)} 题，目标 {n} 题；"
                            "要减少题目请用 remove 指定删除哪几道")
        return
    wanted = [op.new_qtype] if op.new_qtype else []
    types = list(dict.fromkeys([*have, *wanted, *DEFAULT_TYPES.get(req.subject, [])]))
    types = [t for t in types if t in allowed_types(req.subject)]
    options = [o for o in feasible_count_options(n, req.total_score, types, limit=500, max_counts=max_counts(req))
               if all(o.get(t, 0) >= c for t, c in have.items())]
    if not options:
        out.problems.append(f"{tag} 在保留现有 {len(bp.slots)} 题的前提下，{n} 道题凑不成满分 {req.total_score}")
        return
    plan = options[0]
    for t in types:
        for _ in range(plan.get(t, 0) - have.get(t, 0)):
            _add_slot(bp, req, out, t, op.knowledge_point or "在出卷范围内选择，与已有题目考查不同的内容",
                      op.difficulty, op.instruction or f"扩充试卷，新增一道{t.value}")


def apply_ops(paper: PaperSnapshot, plan: EditPlan, load_version=None) -> AppliedOps:
    """确定性地应用修改操作。不调用模型、不写数据库，可以反复试运行。
    load_version：按版本号读取旧快照（restore 用），由调用方从数据库提供。"""
    req = paper.request.model_copy(deep=True)
    bp = paper.blueprint.model_copy(deep=True)
    out = AppliedOps(request=req, blueprint=bp)
    old_stem = {q.slot_id: q.stem for q in paper.questions}
    old_q = {q.slot_id: q for q in paper.questions}
    n = len(bp.slots)

    # 第一步：所有题号都按“修改前”的卷面解析成稳定的 slot_id，之后的操作只认 slot_id
    resolved: list[list[str]] = []
    for i, op in enumerate(plan.ops, start=1):
        ids = []
        for p in op.positions:
            if not 1 <= p <= n:
                out.problems.append(f"操作{i}（{op.op}）引用了不存在的第 {p} 题，当前共 {n} 题")
            else:
                ids.append(bp.slot_at(p).slot_id)
        resolved.append(ids)
    if out.problems:
        return out

    fixed_scores: dict[QType, int] = {}
    for i, (op, ids) in enumerate(zip(plan.ops, resolved), start=1):
        tag = f"操作{i}（{op.op}）"
        if op.op == "rewrite":
            if not ids:
                out.problems.append(f"{tag} 需要指定要重出的题号")
            for sid in ids:
                s = bp.slot(sid)
                if op.difficulty:
                    s.difficulty = op.difficulty
                if op.knowledge_point:
                    s.knowledge_point = op.knowledge_point
                if op.instruction:
                    s.intent = op.instruction[:40]
                out.regenerate.add(sid)
                out.notes[sid] = f"教师要求：{op.instruction or '重出这道题'}。原题为“{old_stem[sid][:60]}”，新题必须与原题明显不同。"
        elif op.op == "retype":
            if len(ids) != 1 or op.new_qtype is None:
                out.problems.append(f"{tag} 需要恰好 1 个题号和 new_qtype")
                continue
            s = bp.slot(ids[0])
            if s.qtype == op.new_qtype:
                out.problems.append(f"{tag} 第 {s.position} 题已经是{s.qtype.value}，不需要改题型")
                continue
            s.qtype = op.new_qtype
            if op.difficulty:
                s.difficulty = op.difficulty
            if op.knowledge_point:
                s.knowledge_point = op.knowledge_point
            if op.new_qtype == QType.APPLICATION:
                s.cognitive_level = CognitiveLevel.APPLY
            s.intent = (op.instruction or f"改为{op.new_qtype.value}")[:40]
            out.regenerate.add(s.slot_id)
            out.notes[s.slot_id] = f"教师要求：{op.instruction or '改为' + op.new_qtype.value}。原题为“{old_stem[s.slot_id][:60]}”，现在改为{op.new_qtype.value}。"
        elif op.op == "add":
            if op.new_qtype is None or not op.knowledge_point:
                out.problems.append(f"{tag} 需要给出 new_qtype 和 knowledge_point")
                continue
            for _ in range(max(1, op.count or 1)):
                _add_slot(bp, req, out, op.new_qtype, op.knowledge_point, op.difficulty, op.instruction)
        elif op.op == "resize":
            _resize(bp, req, out, op, tag)
        elif op.op == "remove":
            if not ids:
                out.problems.append(f"{tag} 需要指定要删除的题号")
            bp.slots = [s for s in bp.slots if s.slot_id not in ids]
            req.question_count -= len(ids)
            out.removed |= set(ids)
            out.regenerate -= set(ids)
        elif op.op == "rescore":
            if not op.scores:
                out.problems.append(f"{tag} 需要给出 scores")
            fixed_scores.update(op.scores)
        elif op.op == "set_total":
            if not op.total or op.total <= 0:
                out.problems.append(f"{tag} 需要给出正整数 total")
            else:
                req.total_score = op.total
        elif op.op == "add_constraint":
            if op.constraint is None and op.instruction:
                # 模型常只写了 instruction 没给 constraint 对象（e2e 评测）：由代码按原话构造
                from .domain import PaperConstraint

                text = op.instruction
                scope = [t for t in QType if t.value in text or t.value[:2] in text]
                kind = ConstraintKind.INTEGER_ANSWER if "整数" in text else ConstraintKind.FREE_TEXT
                op = op.model_copy(update={"constraint": PaperConstraint(kind=kind, text=text, scope_types=scope)})
            if op.constraint is None:
                out.problems.append(f"{tag} 需要给出 constraint，如 {{\"kind\": \"integer_answer\", "
                                    f"\"text\": \"答案为整数\", \"scope_types\": [\"计算题\"]}}")
                continue
            req.constraints.append(op.constraint)
            out.new_constraints.append(op.constraint)
        elif op.op == "restore":
            if len(ids) != 1 or op.from_version is None:
                out.problems.append(f"{tag} 需要恰好 1 个题号和 from_version")
                continue
            if load_version is None:
                out.problems.append(f"{tag} 当前无法读取历史版本")
                continue
            try:
                old_paper = load_version(op.from_version)
            except KeyError:
                out.problems.append(f"{tag} 不存在第 {op.from_version} 版")
                continue
            sid = ids[0]
            prev = next((q for q in old_paper.questions if q.slot_id == sid), None)
            if prev is None:
                out.problems.append(f"{tag} 第 {op.from_version} 版里还没有这道题（它是后来新增的），无法恢复")
                continue
            if prev.content_fingerprint() == old_q[sid].content_fingerprint():
                out.problems.append(f"{tag} 这道题在第 {op.from_version} 版与当前内容相同，不需要恢复；"
                                    "请用 get_history 查看它是在哪一版被改动的")
                continue
            prev_slot = old_paper.blueprint.slot(sid)
            s = bp.slot(sid)
            for f in ("qtype", "knowledge_point", "difficulty", "cognitive_level", "intent", "kp_id", "scope_note"):
                setattr(s, f, getattr(prev_slot, f))
            out.restored[sid] = prev
            out.regenerate.discard(sid)
        elif op.op == "patch":
            if len(ids) != 1 or not op.patches:
                out.problems.append(f"{tag} 需要恰好 1 个题号和至少一处 patches")
                continue
            base = out.patched.get(ids[0], old_q[ids[0]])
            new = _apply_patches(base, op.patches, tag, out.problems)
            if new is not None:
                out.patched[ids[0]] = new

    if out.problems:
        return out

    # 第二步：重新编号（题型分组），再由求解器重配分值
    bp.resequence()
    counts: dict[QType, int] = {}
    for s in bp.slots:
        counts[s.qtype] = counts.get(s.qtype, 0) + 1
    unknown = [t.value for t in fixed_scores if t not in counts]
    if unknown:
        out.problems.append(f"试卷中没有这些题型，无法设置分值：{unknown}")
        return out
    now = current_scores(paper.blueprint)
    assign = rebalance(counts, req.total_score, now, fixed_scores, keep=now)
    if assign is None:
        from .scoring import SCORE_RANGE

        dist = "、".join(f"{t.value} {c} 题" for t, c in counts.items())
        ranges = "、".join(f"{t.value} {SCORE_RANGE[t].start}～{SCORE_RANGE[t].stop - 1} 分" for t in counts)
        out.problems.append(f"修改后为 {len(bp.slots)} 题（{dist}），按每题分值范围（{ranges}）凑不成满分 "
                            f"{req.total_score}。（修改前卷面总分为 {paper.total_score}，这是程序计算的结果，"
                            "请原样转述，不要自行计算或解释分值）请换一种修改方式，或请老师调整题数、题型或满分")
        return out
    for s in bp.slots:
        s.score = assign[s.qtype]

    out.problems += [str(v) for v in check_blueprint(req, bp)]
    return out


@dataclass
class EditResult:
    action: str
    reply: str
    paper: PaperSnapshot
    version: int
    changes: list[SlotChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    totals: dict[str, int] = field(default_factory=dict)
    review: ReviewReport | None = None


def _constraint_scope(paper_slots, constraints) -> set[str]:
    """新增的可验证约束作用到哪些现有题上。"""
    out: set[str] = set()
    for c in constraints:
        if c.kind not in VERIFIABLE:
            continue
        for s in paper_slots:
            if (not c.scope_types or s.qtype in c.scope_types) and s.qtype != QType.CHOICE:
                out.add(s.slot_id)
    return out


def asks_for_confirmation(reply: str) -> bool:
    """d042：回复写着“请回复‘照此执行’”，实际却已提交 edit 并执行了。"""
    import re

    text = reply.strip()
    return bool(re.search(r"请(您)?回复|照此执行|确认后(我|再)|是否(需要|要)?(我)?(按|照|这样)", text)) or \
        text.endswith(("？", "?"))


def claims_executed(reply: str) -> bool:
    """d046：回复写着“已把您的要求记为整卷持续约束”，实际只提交了回答，什么都没执行。"""
    import re

    return bool(re.search(r"(已|已经)(为您|帮您|把|将)?[^。，；\n]{0,12}?(记录|记下|记为|记入|修改|改好|改为|改成|替换|换成|"
                          r"更换|恢复|删除|删掉|添加|加入|新增|重出)", reply))


def execution_summary(paper: PaperSnapshot) -> str:
    """修改执行后由程序生成的事实摘要，附在模型回复后面：题数、满分、各题型分值都以程序结果为准，
    避免模型在回复里写错数字（d041、d042）。"""
    groups: dict[QType, list[int]] = {}
    for s in paper.blueprint.slots:
        groups.setdefault(s.qtype, []).append(s.score)
    parts = "；".join(f"{t.value} {len(v)} 题×{v[0]} 分" if len(set(v)) == 1 else f"{t.value} {len(v)} 题共 {sum(v)} 分"
                     for t, v in groups.items())
    return f"（程序执行结果：现在共 {len(paper.blueprint.slots)} 题，满分 {paper.total_score}；{parts}）"


def edit_planner_mode() -> str:
    """修改规划方式：react（默认，按需查阅试卷与历史后再定方案）或 oneshot（一次调用，只看摘要；对比实验用）。"""
    import os

    return os.getenv("EXAM_EDIT_PLANNER", "react").strip().lower() or "react"


def apply_edit(llm: LLMClient, store: PaperStore, paper_id: str, message: str,
               review: bool = True, listener=None, grounding=None, planner: str | None = None) -> EditResult:
    paper, version = store.load(paper_id)
    tracer = Tracer(store=store, paper_id=paper_id, listener=listener)
    tracer.log("edit_input", text=message, base_version=version)
    load_version = lambda v: store.load(paper_id, v)[0]

    def validate(plan: EditPlan) -> list[str]:
        if plan.action != "edit":
            if claims_executed(plan.reply):
                return ["reply 说已经记录 / 修改了，但 action 不是 edit，程序不会执行任何修改。"
                        "老师要求的修改（包括持续约束）要提交 action=edit 和 ops；只是回答就不要说“已记录”“已修改”"]
            return []
        if not plan.ops:
            return ["action 为 edit 时必须给出至少一个操作"]
        if asks_for_confirmation(plan.reply):
            return ["reply 在征求老师同意（如“请回复照此执行”），但 action=edit 会立即执行修改。"
                    "如果要先征求意见，用 action=answer 且不给 ops；如果老师已经要求修改，reply 直接说明将执行的修改"]
        return apply_ops(paper, plan, load_version).problems

    planner = planner or edit_planner_mode()
    if planner == "oneshot" or not hasattr(llm, "chat_tools"):
        history = store.recent_turns(paper_id, limit=6)
        plan = plan_edit(llm, paper, message, history=history, validate=validate, tracer=tracer)
    else:
        from .agents.edit_agent import EditWorkspace, run_edit_agent

        ws = EditWorkspace(store=store, paper_id=paper_id, paper=paper, version=version, grounding=grounding)
        plan = run_edit_agent(llm, ws, message, validate=validate, tracer=tracer)
    tracer.log("edit_plan", action=plan.action, ops=[o.model_dump(mode="json", exclude_defaults=True)
                                                     for o in plan.ops])
    store.add_turn(paper_id, "user", message, version)

    if plan.action != "edit":
        store.add_turn(paper_id, "assistant", plan.reply, version)
        return EditResult(action=plan.action, reply=plan.reply, paper=paper, version=version,
                          totals=dict(tracer.totals))

    applied = apply_ops(paper, plan, load_version)  # validate 已保证无 problems
    bp, req = applied.blueprint, applied.request
    old_q = {q.slot_id: q for q in paper.questions}

    to_write = [s for s in bp.slots if s.slot_id in applied.regenerate]
    others = [q for sid, q in old_q.items() if sid not in applied.regenerate and sid not in applied.removed]
    new_q = fill_slots(llm, req, to_write, existing=others, notes=applied.notes, tracer=tracer) if to_write else {}

    current: dict[str, Question] = {}
    restored, patched = applied.restored, applied.patched
    for s in bp.slots:
        if s.slot_id in new_q:
            current[s.slot_id] = new_q[s.slot_id]
        elif s.slot_id in restored:   # 从旧版本原样恢复：内容不经模型，只同步分值和题型
            current[s.slot_id] = restored[s.slot_id].model_copy(update={"score": s.score, "qtype": s.qtype})
        elif s.slot_id in patched:    # 局部改字：由代码替换文字，之后必须重新核验
            current[s.slot_id] = patched[s.slot_id].model_copy(update={"score": s.score})
        else:  # 未改动的题：内容原样保留，只同步可能被重配的分值
            current[s.slot_id] = old_q[s.slot_id].model_copy(update={"score": s.score})

    # 审核：新写、改字的题做核验 + 教学审核；恢复的题和新约束范围内的现有题只做核验
    report = None
    touched = set(new_q) | set(patched)
    allowed = applied.regenerate | applied.removed | set(restored) | set(patched)
    scope = _constraint_scope(bp.slots, applied.new_constraints)
    if review and (touched or scope or restored):
        current, report = review_and_repair(llm, req, bp, current, verify=touched | scope | set(restored),
                                            pedagogy=touched, tracer=tracer, keep_stem=set(patched))
        allowed |= report.repaired
    elif scope:
        applied.warnings.append("本次未开启审核，现有题目是否满足新约束未经核验")
    if patched and not review:
        applied.warnings.append("本次未开启审核，局部修改后的题目未经核验")
    current = stamp_status(current, report, unreviewed=touched)

    new_paper = PaperSnapshot(paper_id=paper.paper_id, title=paper.title, request=req,
                              blueprint=bp, questions=[current[s.slot_id] for s in bp.slots])

    violations = check_paper(new_paper) + check_preserved(paper, new_paper, allowed_slots=allowed)
    tracer.log("edit_check", violations=[str(v) for v in violations],
               regenerated=sorted(applied.regenerate), removed=sorted(applied.removed),
               repaired=sorted(report.repaired) if report else [])
    if violations:
        raise AgentFailure("edit_check", [str(v) for v in violations])

    changes = diff_snapshots(paper, new_paper)
    new_version = store.commit_version(new_paper, expected_version=version, operation="edit",
                                       summary=f"{message[:30]} → {plan.reply[:40]}")
    reply = plan.reply.rstrip() + "\n" + execution_summary(new_paper)
    store.add_turn(paper_id, "assistant", reply, new_version)
    tracer.log("committed", version=new_version, totals=tracer.totals)
    return EditResult(action="edit", reply=reply, paper=new_paper, version=new_version,
                      changes=changes, warnings=applied.warnings, totals=dict(tracer.totals),
                      review=report)


def undo(store: PaperStore, paper_id: str) -> tuple[PaperSnapshot, int, int]:
    """撤销 = 把上一个版本的内容作为一个新版本提交。历史永不改写，撤销本身也可追溯。"""
    v = store.current_version(paper_id)
    history = {h["version"]: h for h in store.history(paper_id)}
    target = history[v]["parent_version"]
    if target is None:
        raise ValueError("已经回到最初内容，无法继续撤销")
    previous, _ = store.load(paper_id, target)
    new_v = store.commit_version(previous, expected_version=v, operation="revert",
                                 summary=f"撤销 v{v}，恢复为 v{target} 的内容",
                                 parent_version=history[target]["parent_version"])
    store.add_turn(paper_id, "assistant", f"已撤销上一次修改，恢复为 v{target} 的内容", new_v)
    return previous, target, new_v


def confirm(store: PaperStore, paper_id: str, positions: list[int], note: str = "") -> tuple[PaperSnapshot, int, list[int]]:
    """老师人工确认若干道题：状态改为“老师已确认”，原状态写进 detail。题目内容不变，记为新版本。"""
    paper, v = store.load(paper_id)
    targets = {paper.blueprint.slot_at(p).slot_id: p for p in positions}  # 题号不存在会抛 KeyError
    questions = []
    for q in paper.questions:
        if q.slot_id in targets:
            before = q.verification or Verification(status=VStatus.NOT_REVIEWED)
            q = q.model_copy(update={"verification": Verification(
                status=VStatus.TEACHER, method="老师人工确认",
                detail=f"确认前：{before.status.value}" + (f"（{before.detail}）" if before.detail else "")
                       + (f"；备注：{note}" if note else ""))})
        questions.append(q)
    new = paper.model_copy(update={"questions": questions})
    nums = sorted(targets.values())
    new_v = store.commit_version(new, expected_version=v, operation="confirm",
                                 summary=f"老师确认第 {'、'.join(map(str, nums))} 题")
    store.add_turn(paper_id, "assistant", f"已标记第 {'、'.join(map(str, nums))} 题为老师已确认", new_v)
    return new, new_v, nums
