"""命令行入口。

用法（在项目根目录）：
  python -m exam_agent chat                    # 对话模式：先出卷，再连续修改
  python -m exam_agent chat <paper_id>         # 对已有试卷继续对话修改
  python -m exam_agent new "出一份一元一次方程试卷，10 道题，选择、填空、计算都有，满分 100 分。"
  python -m exam_agent list
  python -m exam_agent show <paper_id> [--version N] [--answers] [--blueprint]
  python -m exam_agent history <paper_id>
  python -m exam_agent diff <paper_id> <旧版本> <新版本>
  python -m exam_agent export <paper_id> [--version N]   # 导出学生版和答案版 PDF
  python -m exam_agent trace <paper_id>
  python -m exam_agent verify <paper_id>                 # 每道题的验证情况
  python -m exam_agent delete <paper_id> | --all         # 删除一份 / 清空全部试卷
  python -m exam_agent confirm <paper_id> 3 7            # 老师确认第 3、7 题
  python -m exam_agent eval-review [--repeats 3] [--configs none,self,blind_tool,full]   # 审核对比实验
  python -m exam_agent web [--port 8000]                 # 网页版（浏览器访问）
  python -m exam_agent kb check                          # 知识库完整性校验
  python -m exam_agent kb init                           # 按教材目录生成单元骨架卡片
  python -m exam_agent kb build [--books 三上,三下]       # 模型起草单元卡片（代码校验）
  python -m exam_agent kb build --subject english        # 英语知识库（各 kb 命令都可加 --subject english）
  python -m exam_agent kb sample [--n 15]                # 随机抽检，生成核对表
  python -m exam_agent kb audit-report <核对表.csv>       # 统计抽检准确率
  python -m exam_agent kb show [--book 三上] [--unit 5]   # 查看知识点 / 单元卡片
  python -m exam_agent kb search "把未知数挪到一边" [--mode hybrid] [--book 七上]
  python -m exam_agent eval-retrieval [--modes bm25,vector,hybrid,hybrid_rerank]   # 检索评测
  python -m exam_agent eval-edit [--planners oneshot,react] [--repeats 1]           # 修改/讨论评测
  python -m exam_agent eval-e2e [--limit 10] [--workers 4] [--resume 目录]          # 端到端：批量出卷 + 自动修改
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from .agents.base import AgentFailure
from .config import Settings, load_settings
from .diffing import diff_snapshots
from .render import (LETTERS, show_blueprint, show_changes, show_paper, show_review, show_verification,
                     status_summary)
from .store import PaperStore, VersionConflict

CHAT_HELP = """可以直接用自然语言出卷、修改，也可以和助手讨论试卷，例如：
  第 3 题太难，换一道基础题，其他题别动
  把讲买苹果的那道题改简单点 / 第 5 题的 15 改成 18
  第 7 题答案为什么是 B？/ 这份卷难度合适吗？有没有超纲的题？
  还是把第 3 题改回原来的
  导出学生版和答案版
命令：/show 查看试卷  /answers 含答案  /blueprint 蓝图  /verify 验证情况  /confirm 3 7 老师确认题目
      /history 版本历史  /export 导出 PDF  /undo 撤销上一次修改  /trace 运行轨迹  /help 帮助  /quit 退出"""


def _print_created(console: Console, res) -> None:
    if res.assumptions:
        console.print("[yellow]系统做了以下假设：[/]")
        for a in res.assumptions:
            console.print(f"  · {a}")
    show_blueprint(console, res.paper)
    show_paper(console, res.paper)
    show_review(console, res.review, res.paper)
    show_verification(console, res.paper)
    console.print(f"\n[bold]paper_id = {res.paper.paper_id}[/]  版本 v{res.version}  "
                  f"LLM 调用 {res.totals['llm_calls']} 次，"
                  f"tokens {res.totals['prompt_tokens']}+{res.totals['completion_tokens']}")


def _print_changed_questions(console: Console, paper, changes) -> None:
    ids = {c.slot_id for c in changes if c.kind in ("content", "added")}
    pos = {s.slot_id: s.position for s in paper.blueprint.slots}
    for q in paper.questions:
        if q.slot_id in ids:
            console.print(f"\n[bold]第{pos[q.slot_id]}题（{q.qtype.value}，{q.score}分，{q.difficulty.value}）[/] {q.stem}")
            for j, opt in enumerate(q.options):
                console.print(f"    {LETTERS[j]}. {opt}")
            console.print(f"    [green]答案：{q.answer}[/]")
            console.print(f"    [dim]解析：{q.analysis}[/]")


def _show_history(console: Console, store: PaperStore, paper_id: str) -> None:
    tbl = Table("版本", "父版本", "操作", "说明", "时间")
    for r in store.history(paper_id):
        tbl.add_row(f"v{r['version']}", str(r["parent_version"] or "-"), r["operation"],
                    r["summary"], r["created_at"])
    console.print(tbl)


def _show_trace(console: Console, store: PaperStore, paper_id: str) -> None:
    tbl = Table("run", "步骤", "详情")
    for e in store.events(paper_id=paper_id):
        pl = e["payload"]
        detail = ", ".join(f"{k}={v}" for k, v in pl.items() if k != "text")[:160]
        tbl.add_row(e["run_id"], e["step"], detail)
    console.print(tbl)


def _export(console: Console, store: PaperStore, paper_id: str, version: int | None = None) -> None:
    from .export_pdf import ExportError, export_paper

    paper, v = store.load(paper_id, version)
    try:
        with console.status("正在生成 PDF ..."):
            student, answers = export_paper(paper, v)
    except ExportError as e:
        console.print(f"[red]导出失败：{e}[/]")
        return
    console.print(f"已导出 v{v}：\n  学生版：{student}\n  答案版：{answers}")


def _grounding(console: Console, use_kb: bool):
    """加载课程知识库与检索器；失败时如实提示，退回到不使用知识库。"""
    if not use_kb:
        return None
    from .grounding import MultiGrounding

    try:
        with console.status("加载课程知识库 ..."):
            g = MultiGrounding.from_env()
        console.print(f"[dim]课程知识库：{g.size()}，检索方式 {g.mode}[/]")
        return g
    except Exception as e:
        console.print(f"[yellow]课程知识库加载失败（{e}），本次不使用知识库[/]")
        return None


def chat(console: Console, store: PaperStore, settings: Settings, paper_id: str | None,
         review: bool = True, use_kb: bool = True) -> int:
    from .editing import apply_edit, confirm, undo
    from .llm import make_llm
    from .pipeline import create_paper

    llm = make_llm(settings)
    console.print(f"[dim]模型：{llm.describe() if hasattr(llm, 'describe') else llm.model}[/]")
    grounding = _grounding(console, use_kb)   # 出卷用它定范围；修改 / 讨论时 lookup_scope 工具也用它
    console.print(CHAT_HELP)
    if paper_id:
        paper, v = store.load(paper_id)
        console.print(f"\n已打开《{paper.title}》v{v}")
    else:
        console.print("\n请先描述要出的试卷。")

    while True:
        try:
            text = console.input("\n[bold cyan]老师> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            break
        if text == "/help":
            console.print(CHAT_HELP)
            continue
        if text.startswith("/") and not paper_id:
            console.print("[yellow]还没有试卷，请先描述要出的试卷。[/]")
            continue

        try:
            if text in ("/show", "/answers"):
                paper, v = store.load(paper_id)
                show_paper(console, paper, with_answers=(text == "/answers"))
                console.print(f"[dim]版本 v{v}[/]")
            elif text == "/verify":
                show_verification(console, store.load(paper_id)[0])
            elif text.startswith("/confirm"):
                parts = text.replace("/confirm", "").replace("，", " ").replace(",", " ").replace("、", " ").split()
                nums = [int(x) for x in parts if x.isdigit()]
                if not nums or len(nums) != len(parts):
                    console.print("[yellow]用法：/confirm 题号，例如 /confirm 3 7[/]")
                    continue
                paper, nv, done = confirm(store, paper_id, nums)
                console.print(f"已标记第 {'、'.join(map(str, done))} 题为“老师已确认”，保存为 v{nv}")
                show_verification(console, paper)
            elif text == "/blueprint":
                show_blueprint(console, store.load(paper_id)[0])
            elif text == "/history":
                _show_history(console, store, paper_id)
            elif text == "/trace":
                _show_trace(console, store, paper_id)
            elif text == "/export":
                _export(console, store, paper_id)
            elif text == "/undo":
                current, cv = store.load(paper_id)
                restored, target, nv = undo(store, paper_id)
                console.print(f"已撤销：内容恢复为 v{target}，记为新版本 v{nv}")
                show_changes(console, diff_snapshots(current, restored))
            elif text.startswith("/"):
                console.print("[yellow]未知命令，输入 /help 查看[/]")
            elif paper_id is None:
                with console.status("正在出卷：理解需求 → 规划蓝图 → 命题 → 审核返修 ..."):
                    res = create_paper(llm, store, text, review=review, grounding=grounding)
                paper_id = res.paper.paper_id
                _print_created(console, res)
            else:
                def show_step(step, payload):
                    if step == "edit_agent_step" and payload.get("tool") not in ("submit_plan", "reply"):
                        console.print(f"[dim]  · {payload.get('label', '')}[/]")

                with console.status("正在查阅试卷、思考答复 ..."):
                    res = apply_edit(llm, store, paper_id, text, review=review, grounding=grounding,
                                     listener=show_step)
                console.print(f"[bold green]助手：[/]{res.reply}")
                for w in res.warnings:
                    console.print(f"[yellow]注意：{w}[/]")
                if res.action == "export":
                    _export(console, store, paper_id)
                if res.action == "edit":
                    show_changes(console, res.changes)
                    _print_changed_questions(console, res.paper, res.changes)
                    show_review(console, res.review, res.paper)
                    console.print(f"验证情况：{status_summary(res.paper)}（/verify 查看详情）")
                    console.print(f"[dim]已保存为 v{res.version}；本次 LLM 调用 {res.totals['llm_calls']} 次，"
                                  f"tokens {res.totals['prompt_tokens']}+{res.totals['completion_tokens']}[/]")
        except AgentFailure as e:
            console.print(f"[red]这次没能完成：{e}[/]\n[dim]试卷保持原样，可以换个说法再试。[/]")
        except VersionConflict as e:
            console.print(f"[red]版本冲突：{e}[/]")
        except (KeyError, ValueError) as e:
            console.print(f"[red]{e}[/]")
    console.print("再见。" + (f"试卷 id：{paper_id}" if paper_id else ""))
    return 0


def _make_retriever(console: Console, args, need_models: bool):
    from .retrieval import Retriever, load_kb, make_embedder, make_reranker

    kb = load_kb(getattr(args, "subject", "math"))
    emb = rer = None
    if need_models:
        with console.status("加载向量 / 重排模型（首次使用会下载，约几百 MB，请耐心等待）..."):
            try:
                emb = make_embedder(args.embedder)
                rer = make_reranker(args.reranker) if emb else None
            except RuntimeError as e:
                console.print(f"[yellow]{e}\n先只用 BM25 继续。[/]")
                emb = rer = None
    return kb, Retriever(kb.docs(), emb, rer)


def _kb_pipeline(console: Console, args) -> int:
    from pathlib import Path

    from . import kbuild
    from .retrieval import load_kb

    if args.action == "init":
        written = kbuild.init_skeletons(force=args.force, subject=args.subject)
        console.print(f"已按目录生成骨架卡片：{'、'.join(written) or '无（各册都已起草，未覆盖）'}")
        return 0
    if args.action == "build":
        from .agents.base import Tracer
        from .llm import make_llm

        settings = load_settings()
        toc = kbuild.load_toc(subject=args.subject)
        books = [b.strip() for b in args.books.split(",")] if args.books else list(toc["books"])
        unknown = [b for b in books if b not in toc["books"]]
        if unknown:
            console.print(f"[red]目录中没有这些册别：{unknown}[/]")
            return 1
        tracer = Tracer(store=PaperStore(settings.db_path), paper_id="kb_build")
        result = kbuild.build_books(make_llm(settings), books, progress=console.print, tracer=tracer,
                                    subject=args.subject)
        problems = load_kb(args.subject).check()
        for pr in problems:
            console.print(f"[red]· {pr}[/]")
        console.print(f"完成 {sum(v.startswith('完成') for v in result.values())}/{len(result)} 册；"
                      f"LLM 调用 {tracer.totals['llm_calls']} 次，tokens {tracer.totals['prompt_tokens']}+{tracer.totals['completion_tokens']}")
        console.print("下一步：python -m exam_agent kb sample 抽检")
        return 0 if not problems else 1
    if args.action == "sample":
        path = kbuild.sample_for_audit(load_kb(args.subject), args.n, args.seed)
        console.print(f"已生成核对表：{path}\n用 Excel 打开，在“结论”列填 对 / 错，错的写明原因，保存后运行：\n"
                      f"  python -m exam_agent kb audit-report \"{path}\"")
        return 0
    if not args.query:
        console.print("[red]用法：kb audit-report <核对表路径>[/]")
        return 1
    rep = kbuild.audit_report(Path(args.query), root=kbuild.SPECS[args.subject].root)
    acc = "未填写" if rep["accuracy"] is None else f"{rep['accuracy']:.0%}"
    console.print(f"抽取 {rep['sampled']} 张，已填写 {rep['filled']} 张，错误 {rep['wrong']} 张，准确率 {acc}")
    for e in rep["errors"]:
        console.print(f"  · {e['id']} {e['unit']}：{e['note']}")
    console.print("[dim]判为“对”的卡片已标记为人工核对过（checked）；结果追加到 kb/math/audit/audit_history.csv[/]")
    return 0


def _retrieval_cmd(console: Console, args) -> int:
    from .retrieval import MODE_LABEL

    if args.cmd == "eval-retrieval":
        from .experiments.retrieval_eval import run_retrieval_eval

        _, r = _make_retriever(console, args, need_models=True)
        modes = [m.strip() for m in args.modes.split(",")] if args.modes else None
        bad = [m for m in modes or [] if m not in r.available_modes()]
        if bad:
            console.print(f"[red]这些方式当前不可用：{bad}（可用：{r.available_modes()}）[/]")
            return 1
        summary, (json_path, md_path) = run_retrieval_eval(r, modes)
        tbl = Table("方法", "Hit@1", "Recall@3", "Recall@5", "MRR", "延迟 ms")
        for m, s in summary.items():
            tbl.add_row(MODE_LABEL[m], f"{s['hit@1']:.0%}", f"{s['recall@3']:.0%}", f"{s['recall@5']:.0%}",
                        f"{s['mrr']:.3f}", f"{s['ms']:.1f}")
        console.print(tbl)
        console.print(f"报告：{md_path}\n原始数据：{json_path}")
        return 0

    if args.action in ("init", "build", "sample", "audit-report"):
        return _kb_pipeline(console, args)
    kb, r = _make_retriever(console, args, need_models=args.action == "search")
    if args.action == "check":
        problems = kb.check()
        console.print(f"知识点 {len(kb.points)} 个，课标条目 {len(kb.clauses)} 条")
        for pr in problems:
            console.print(f"[red]· {pr}[/]")
        console.print("[green]校验通过[/]" if not problems else f"[red]{len(problems)} 个问题[/]")
        return 1 if problems else 0
    if args.action == "show":
        tbl = Table("id", "位置", "名称", "主要内容", "考查边界", "状态")
        for p in kb.points:
            if (args.book is None or p.book == args.book) and (args.unit is None or p.unit == args.unit):
                tbl.add_row(p.id, p.location, p.name, "、".join(p.key_points) or p.description[:30] or "-",
                            p.boundary or "-", p.status or "-")
        console.print(tbl)
        return 0
    if not args.query:
        console.print("[red]用法：kb search \"查询内容\"[/]")
        return 1
    mode = args.mode or r.available_modes()[-1]
    where = {"kind": "kp"}
    if args.book:
        where["book"] = args.book
        if args.unit:
            where["unit"] = args.unit
    hits = r.search(args.query, args.k, mode, where)
    tbl = Table("名次", "id", "名称", "分数", "各路名次")
    for i, h in enumerate(hits, 1):
        tbl.add_row(str(i), h.id, h.doc.title, f"{h.score:.3f}", " ".join(f"{k}:{v}" for k, v in h.ranks.items()))
    console.print(f"方式：{MODE_LABEL[mode]}  耗时 {r.last_timings} ms")
    console.print(tbl)
    return 0


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":  # 避免 Windows 终端中文乱码
        for stream in (sys.stdout, sys.stdin):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    p = argparse.ArgumentParser(prog="exam_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat", help="对话模式：出卷并连续修改")
    c.add_argument("paper_id", nargs="?")
    c.add_argument("--no-review", action="store_true", help="关闭审核（用于对比实验）")
    c.add_argument("--no-kb", action="store_true", help="不使用课程知识库（用于对比实验）")
    n = sub.add_parser("new", help="根据一句话需求新建试卷")
    n.add_argument("text")
    n.add_argument("--no-review", action="store_true", help="关闭审核（用于对比实验）")
    n.add_argument("--no-kb", action="store_true", help="不使用课程知识库（用于对比实验）")
    sub.add_parser("ping", help="检查 DeepSeek API 是否配置正确")
    sub.add_parser("list", help="列出已有试卷")
    s = sub.add_parser("show", help="查看试卷")
    s.add_argument("paper_id")
    s.add_argument("--version", type=int)
    s.add_argument("--answers", action="store_true")
    s.add_argument("--blueprint", action="store_true")
    h = sub.add_parser("history", help="查看版本历史")
    h.add_argument("paper_id")
    x = sub.add_parser("export", help="导出学生版和答案版 PDF 到 output/ 文件夹")
    x.add_argument("paper_id")
    x.add_argument("--version", type=int)
    d = sub.add_parser("diff", help="比较两个版本")
    d.add_argument("paper_id")
    d.add_argument("old", type=int)
    d.add_argument("new", type=int)
    vf = sub.add_parser("verify", help="查看每道题的验证情况")
    vf.add_argument("paper_id")
    cf = sub.add_parser("confirm", help="老师确认题目（如 confirm p_xxx 3 7）")
    cf.add_argument("paper_id")
    cf.add_argument("positions", type=int, nargs="+")
    cf.add_argument("--note", default="")
    t = sub.add_parser("trace", help="查看运行轨迹")
    t.add_argument("paper_id")
    e = sub.add_parser("eval-review", help="审核效果对比实验，结果保存在 evals/results/")
    e.add_argument("--repeats", type=int, default=3, help="每种配置重复次数（默认 3）")
    e.add_argument("--configs", default="none,self,blind_tool,full", help="逗号分隔的配置列表")
    e.add_argument("--batch-size", type=int, default=10)
    e.add_argument("--limit", type=int, help="只用前 N 道题（试跑用）")
    e2 = sub.add_parser("eval-e2e", help="端到端评测：批量出卷并自动修改，结果保存在 evals/results/e2e_<时间>/")
    e2.add_argument("--limit", type=int, help="只跑前 N 条（试跑用）")
    e2.add_argument("--workers", type=int, default=4, help="并发数（默认 4）")
    e2.add_argument("--resume", help="中断后续跑：传入上次的结果目录")
    e2.add_argument("--no-edits", action="store_true", help="只出卷，不做修改")
    e2.add_argument("--sample", type=int, default=60, help="导出人工抽检的题目数（0 不导出）")
    e2.add_argument("--holdout", action="store_true", help="用留出集（另一批 100 条请求，只在最后跑一次）")
    e2.add_argument("--dataset", choices=["dev", "holdout", "holdout2"],
                    help="指定请求集；holdout2 是留出集 1 之后新增的第二套留出集")
    ee = sub.add_parser("eval-edit", help="修改/讨论评测：一次调用 vs ReAct，结果保存在 evals/results/")
    ee.add_argument("--planners", default="oneshot,react")
    ee.add_argument("--repeats", type=int, default=1)
    ee.add_argument("--cases", help="逗号分隔的用例 id，默认全部")
    ee.add_argument("--split", default="dev", choices=["dev", "holdout", "all"],
                    help="dev 开发集（默认）/ holdout 留出集（只在最后评测）/ all")
    dl = sub.add_parser("delete", help="删除试卷：delete <paper_id>，或 delete --all 清空全部")
    dl.add_argument("paper_id", nargs="?")
    dl.add_argument("--all", action="store_true")
    w = sub.add_parser("web", help="启动网页服务（浏览器访问）")
    w.add_argument("--host", default="127.0.0.1", help="0.0.0.0 表示允许局域网 / 隧道访问")
    w.add_argument("--port", type=int, default=8000)
    kbp = sub.add_parser("kb", help="课程知识库：check / show / search")
    kbp.add_argument("action", choices=["check", "show", "search", "init", "build", "sample", "audit-report"])
    kbp.add_argument("query", nargs="?")
    kbp.add_argument("--subject", default="math", choices=["math", "english"], help="学科，默认 math")
    kbp.add_argument("--book", help="只看某一册，如 三上、七上")
    kbp.add_argument("--unit", type=int, help="配合 --book 只看某一单元")
    kbp.add_argument("--mode", default=None, help="bm25 / vector / hybrid / hybrid_rerank，默认用可用的最强方式")
    kbp.add_argument("--k", type=int, default=5)
    kbp.add_argument("--embedder", help="覆盖 .env 中的 EXAM_EMBEDDER：fastembed / api / hash / none")
    kbp.add_argument("--reranker", help="覆盖 .env 中的 EXAM_RERANKER：fastembed / none")
    kbp.add_argument("--books", help="build：逗号分隔的册别，如 三上,三下；默认目录中全部小学册别")
    kbp.add_argument("--force", action="store_true", help="init：覆盖已起草的卡片（慎用）")
    kbp.add_argument("--n", type=int, default=15, help="sample：抽取张数")
    kbp.add_argument("--seed", type=int)
    er = sub.add_parser("eval-retrieval", help="检索评测，结果保存在 evals/results/")
    er.add_argument("--modes", help="逗号分隔，默认全部可用方式")
    er.add_argument("--embedder", help="fastembed / api / hash / none")
    er.add_argument("--reranker", help="fastembed / none")
    args = p.parse_args(argv)

    if args.cmd in ("kb", "eval-retrieval"):
        return _retrieval_cmd(Console(), args)
    if args.cmd == "web":
        try:
            import uvicorn
            from .web.app import build_default_app
        except ImportError:
            Console().print("[red]缺少网页依赖：请运行 pip install -r requirements.txt（需要 fastapi、uvicorn）[/]")
            return 1
        Console().print(f"网页服务启动中：浏览器打开 http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
        uvicorn.run(build_default_app(), host=args.host, port=args.port)
        return 0

    console = Console()
    settings = load_settings()
    store = PaperStore(settings.db_path)

    if args.cmd == "chat":
        return chat(console, store, settings, args.paper_id, review=not args.no_review, use_kb=not args.no_kb)

    if args.cmd == "eval-review":
        from .experiments.review_eval import CONFIGS, run_experiment
        from .llm import make_llm

        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown = [c for c in configs if c not in CONFIGS]
        if unknown:
            console.print(f"[red]未知配置：{unknown}，可选：{CONFIGS}[/]")
            return 1
        summary, (json_path, md_path) = run_experiment(
            make_llm(settings), configs, args.repeats, args.batch_size, args.limit,
            progress=console.print)
        tbl = Table("配置", "检出率", "误报率", "调用次数/40题", "token/40题", "耗时秒/40题")
        for cfg, s in summary.items():
            tbl.add_row(cfg, f"{s['recall']['mean']:.0%}", f"{s['false_positive_rate']['mean']:.0%}",
                        f"{s['llm_calls']['mean']:.0f}", f"{s['tokens']['mean']:.0f}", f"{s['seconds']['mean']:.0f}")
        console.print(tbl)
        console.print(f"报告：{md_path}\n原始数据：{json_path}")
        return 0

    if args.cmd == "eval-e2e":
        from pathlib import Path

        from .experiments.e2e_eval import run_e2e
        from .llm import make_llm

        grounding = _grounding(console, True)
        done = [0]

        def show(row):
            done[0] += 1
            mark = "✓" if row["ok"] else "✗"
            edits = " ".join(("✓" if e["ok"] else "✗") + e["kind"] for e in row["edits"])
            console.print(f"[{'green' if row['ok'] else 'red'}]{done[0]:>3} {row['id']} {mark} {row['outcome']}[/] "
                          f"[dim]{row['seconds']:.0f}s {row['text'][:30]}[/] {edits}")

        s_, out = run_e2e(lambda: make_llm(settings), grounding, workers=args.workers, limit=args.limit,
                          resume=Path(args.resume) if args.resume else None, do_edits=not args.no_edits,
                          sample=args.sample, model=settings.model, progress=show, holdout=args.holdout, dataset=args.dataset)
        console.print(f"出卷符合预期 {s_['create_ok']}/{s_['requests']}；修改成功 {s_['edits']['ok']}/{s_['edits']['n']}")
        console.print(f"报告：{out / 'report.md'}\n抽检表：{out / 'audit_sample.csv'}\n"
                      f"试卷库：{out / 'papers.db'}（设置 EXAM_DB_PATH 指向它，即可在网页中查看这些试卷）")
        return 0

    if args.cmd == "eval-edit":
        from .experiments.edit_eval import PLANNER_LABEL, PLANNERS, TAG_LABEL, run_edit_eval
        from .llm import make_llm

        planners = [x.strip() for x in args.planners.split(",") if x.strip()]
        if any(x not in PLANNERS for x in planners):
            console.print(f"[red]可选：{PLANNERS}[/]")
            return 1
        grounding = _grounding(console, True)
        show = lambda r: console.print(f"[{'green' if r.ok else 'red'}]{r.planner} {r.case_id} "
                                       f"{'✓' if r.ok else '✗ ' + r.reason}[/] [dim]工具 {r.tool_calls} 次[/]")
        ids = [x.strip() for x in args.cases.split(",")] if args.cases else None
        summary, (json_path, md_path) = run_edit_eval(make_llm(settings), planners, args.repeats, grounding, ids,
                                                      progress=show, split=args.split)
        tbl = Table("方式", "正确率", *[TAG_LABEL[t] for t in TAG_LABEL], "工具/条", "token/条", "秒/条")
        for p_, s_ in summary.items():
            tbl.add_row(PLANNER_LABEL[p_], f"{s_['accuracy']:.0%}",
                        *[f"{s_['by_tag'][t]:.0%}" if t in s_["by_tag"] else "-" for t in TAG_LABEL],
                        f"{s_['tool_calls']:.1f}", f"{s_['tokens']:.0f}", f"{s_['seconds']:.1f}")
        console.print(tbl)
        console.print(f"报告：{md_path}\n原始数据：{json_path}")
        return 0

    if args.cmd == "ping":
        from .llm import make_llm

        res = make_llm(settings).chat_json([
            {"role": "system", "content": "只输出 json"},
            {"role": "user", "content": '回复 json：{"ok": true, "msg": "一句中文问候"}'},
        ])
        console.print(f"模型 {settings.model} 返回：{res.content}")
        console.print(f"tokens：输入 {res.prompt_tokens}，输出 {res.completion_tokens}")
        return 0

    if args.cmd == "new":
        from .llm import make_llm
        from .pipeline import create_paper

        grounding = _grounding(console, not args.no_kb)
        with console.status("正在出卷：理解需求 → 规划蓝图 → 命题 → 审核返修 ..."):
            try:
                res = create_paper(make_llm(settings), store, args.text, review=not args.no_review,
                                   grounding=grounding)
            except AgentFailure as e:
                console.print(f"[red]出卷失败：{e}[/]")
                return 1
        _print_created(console, res)
    elif args.cmd == "list":
        tbl = Table("paper_id", "标题", "当前版本", "创建时间")
        for r in store.list_papers():
            tbl.add_row(r["paper_id"], r["title"], f"v{r['current_version']}", r["created_at"])
        console.print(tbl)
    elif args.cmd == "show":
        paper, v = store.load(args.paper_id, args.version)
        if args.blueprint:
            show_blueprint(console, paper)
        show_paper(console, paper, with_answers=args.answers)
        console.print(f"[dim]版本 v{v}[/]")
    elif args.cmd == "history":
        _show_history(console, store, args.paper_id)
    elif args.cmd == "export":
        _export(console, store, args.paper_id, args.version)
    elif args.cmd == "diff":
        old, _ = store.load(args.paper_id, args.old)
        new, _ = store.load(args.paper_id, args.new)
        show_changes(console, diff_snapshots(old, new))
    elif args.cmd == "delete":
        if args.all:
            if input("确定清空全部试卷？输入 yes 确认：").strip().lower() != "yes":
                console.print("已取消")
                return 0
            console.print(f"已清空全部试卷（共 {store.clear_all()} 份）")
        elif args.paper_id:
            store.delete_paper(args.paper_id)
            console.print(f"已删除 {args.paper_id}")
        else:
            console.print("[red]用法：delete <paper_id> 或 delete --all[/]")
            return 1
    elif args.cmd == "verify":
        show_verification(console, store.load(args.paper_id)[0])
    elif args.cmd == "confirm":
        from .editing import confirm

        paper, nv, done = confirm(store, args.paper_id, args.positions, args.note)
        console.print(f"已标记第 {'、'.join(map(str, done))} 题为“老师已确认”，保存为 v{nv}")
        show_verification(console, paper)
    elif args.cmd == "trace":
        _show_trace(console, store, args.paper_id)
    return 0
