"""终端展示（阶段一）。PDF 导出在后续阶段用 Typst 实现。"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .domain import PaperSnapshot, QType, VStatus

LETTERS = "ABCD"


def show_blueprint(console: Console, paper: PaperSnapshot) -> None:
    grounded = any(s.kp_id for s in paper.blueprint.slots)
    t = Table(title="出卷蓝图（双向细目表）", show_lines=False)
    for col in ("题号", "题型", "分值", "知识点", "难度", "认知层次", "考查意图") + (("依据",) if grounded else ()):
        t.add_column(col)
    for s in sorted(paper.blueprint.slots, key=lambda s: s.position):
        t.add_row(str(s.position), s.qtype.value, str(s.score), s.knowledge_point,
                  s.difficulty.value, s.cognitive_level.value, s.intent, *((s.kp_id or "-",) if grounded else ()))
    console.print(t)
    if paper.blueprint.scope:
        units = {s.kp_id for s in paper.blueprint.slots if s.kp_id}
        console.print(f"考查范围：{paper.blueprint.scope}；本卷用到 {len(units)} 张知识库卡片")
    # 统计由代码计算，不直接相信模型在 rationale 里的自述（实测出现过自述与表格不一致）
    by_type: dict[str, list[int]] = {}
    by_diff: dict[str, int] = {}
    for s in paper.blueprint.slots:
        by_type.setdefault(s.qtype.value, []).append(s.score)
        by_diff[s.difficulty.value] = by_diff.get(s.difficulty.value, 0) + 1
    type_txt = "，".join(f"{k} {len(v)} 题共 {sum(v)} 分" for k, v in by_type.items())
    diff_txt = "，".join(f"{k} {by_diff.get(k, 0)} 题" for k in ("基础", "中等", "提高"))
    console.print(f"合计 {sum(s.score for s in paper.blueprint.slots)} 分：{type_txt}；难度：{diff_txt}")
    if paper.blueprint.rationale:
        console.print(f"[dim]规划说明（模型自述）：{paper.blueprint.rationale}[/]")


KIND_LABEL = {"request": "要求", "content": "重出", "added": "新增", "removed": "删除", "score": "分值", "position": "题号"}


def show_changes(console: Console, changes) -> None:
    if not changes:
        console.print("[dim]题目内容和分值没有变化[/]")
        return
    t = Table(title="本次改动", show_header=True)
    for col in ("类型", "slot", "题号", "说明"):
        t.add_column(col)
    for c in changes:
        if c.old_position is None and c.new_position is None:
            pos = "-"
        elif c.old_position != c.new_position:
            pos = f"{c.old_position or '-'}→{c.new_position or '-'}"
        else:
            pos = str(c.new_position)
        t.add_row(KIND_LABEL.get(c.kind, c.kind), c.slot_id, pos, c.detail)
    console.print(t)


def show_paper(console: Console, paper: PaperSnapshot, with_answers: bool = False) -> None:
    console.rule(f"[bold]{paper.title}")
    current_type: QType | None = None
    for i, q in enumerate(paper.ordered_questions(), start=1):
        if q.qtype != current_type:
            current_type = q.qtype
            console.print(f"\n[bold cyan]{q.qtype.value}[/]")
        console.print(f"{i}. （{q.score}分）{q.stem}")
        for j, opt in enumerate(q.options):
            console.print(f"    {LETTERS[j]}. {opt}")
        if with_answers:
            console.print(f"    [green]答案：{q.answer}[/]")
            console.print(f"    [dim]解析：{q.analysis}[/]")
            console.print(f"    {STATUS_STYLE[q.vstatus]}【{q.vstatus.value}】[/]"
                          + (f"[dim] {q.verification.method}[/]" if q.verification and q.verification.method else ""))
    console.print(f"\n满分 {paper.total_score} 分")


STATUS_STYLE = {VStatus.TOOL: "[green]", VStatus.AGREED: "[cyan]", VStatus.TEACHER: "[green]",
                VStatus.UNVERIFIED: "[yellow]", VStatus.PROBLEM: "[red]", VStatus.NOT_REVIEWED: "[dim]"}
STATUS_ORDER = [VStatus.TOOL, VStatus.AGREED, VStatus.TEACHER, VStatus.UNVERIFIED, VStatus.PROBLEM, VStatus.NOT_REVIEWED]


def status_counts(paper: PaperSnapshot) -> dict[VStatus, int]:
    counts = {s: 0 for s in STATUS_ORDER}
    for q in paper.questions:
        counts[q.vstatus] += 1
    return {k: v for k, v in counts.items() if v}


def status_summary(paper: PaperSnapshot) -> str:
    return "，".join(f"{k.value} {v} 题" for k, v in status_counts(paper).items())


def show_verification(console: Console, paper: PaperSnapshot) -> None:
    """每道题被验证到什么程度。“待老师确认”是正常结果，不是失败。"""
    t = Table(title="验证情况")
    for col in ("题号", "题型", "状态", "怎么验证的", "说明"):
        t.add_column(col)
    for i, q in enumerate(paper.ordered_questions(), start=1):
        v = q.verification
        t.add_row(str(i), q.qtype.value, f"{STATUS_STYLE[q.vstatus]}{q.vstatus.value}[/]",
                  v.method if v else "-", (v.detail if v else "")[:60] or "-")
    console.print(t)
    console.print(f"合计：{status_summary(paper)}")
    if any(q.vstatus == VStatus.AGREED for q in paper.questions):
        console.print("[dim]“独立复核一致”只表示两次模型作答结果相同，同一模型可能犯同样的错，不等于已证实[/]")
    if any(q.vstatus in (VStatus.UNVERIFIED, VStatus.PROBLEM) for q in paper.questions):
        console.print("[dim]核对后可以输入 /confirm 题号（如 /confirm 3 7）标记为老师已确认[/]")


SOURCE_LABEL = {"tool": "求解器", "blind": "盲解", "pedagogy": "教学审核", "kb": "知识库"}
STOP_LABEL = {"passed": "全部通过", "max_rounds": "达到最大返修轮数", "no_progress": "返修没有进展，停止",
              "repair_failed": "返修失败"}


def show_review(console: Console, report, paper) -> None:
    """如实展示审核过程：发现了什么、修了什么、还剩什么。"""
    if report is None:
        return
    pos = {s.slot_id: s.position for s in paper.blueprint.slots}
    p = lambda sid: f"第{pos[sid]}题" if sid in pos else sid
    initial_must = [f for f in report.initial if f.severity == "must_fix"]
    console.print(f"\n[bold]审核[/]：本次审核 {len(report.status)} 题，其中工具验证 {len(report.tool_verified)} 题；"
                  f"发现 {report.must_fix_initial} 道题有问题，返修 {report.rounds} 轮，"
                  f"结果：{STOP_LABEL.get(report.stop_reason, report.stop_reason)}")
    for f in initial_must:
        # 按“题 + 问题类型”判断：同一道题的旧问题修好了、又出现别的问题，旧问题仍算已修复
        still = any(r.slot_id == f.slot_id and r.code == f.code for r in report.remaining)
        mark = "[green]已修复[/]" if f.slot_id in report.repaired and not still else "[red]未修复[/]"
        console.print(f"  · {mark} {p(f.slot_id)}【{SOURCE_LABEL[f.source]}】{f.message}")
    for f in report.remaining:
        if f not in initial_must:
            console.print(f"  · [red]仍存在[/] {p(f.slot_id)}【{SOURCE_LABEL[f.source]}】{f.message}")
    for f in report.suggestions:
        console.print(f"  · [dim]建议 {p(f.slot_id)}【{SOURCE_LABEL[f.source]}】{f.message}[/]")
