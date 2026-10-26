"""端到端评测：真实调用模型，批量出卷，并对每份卷自动做两次修改，统计成功率、验证状态分布、耗时和成本。

回答“这个系统稳不稳定、出的题对不对”：
- 出卷：evals/e2e_requests.jsonl 中 100 条老师的说法（数学 60、英语 25、边界 15）。
  expect=paper 的应当出卷成功；expect=reject 的（语文、八年级数学、凑不成的配分等）应当被拒绝并说明原因。
- 修改：每份成功的卷由程序自动选择两条修改指令（不需要人工），走完整流程（理解 → 命题 → 审核 → 校验 → 保存），
  并用代码检查结果：
    E1  “第 k 题太难，换一道基础题，其他题别动”  → 只有第 k 题内容变化，题数、总分不变
    E2  轮换以下一种：
        restore   “把第 k 题改回原来的”           → 第 k 题内容与 v1 完全相同
        resize    “扩大到 n+5 题，原来的题保留”    → 题数为 n+5，总分不变
        ask       “第 j 题答案为什么是这个？”      → 只回答，不产生新版本
        patch     “第 j 题题干里的 a 改成 a+1”     → 题干出现 a+1，题数、总分不变（数学）
        constraint “计算题答案都要是整数”         → 记录约束，题数、总分不变（数学）
  其他题“一个字没变”由 apply_edit 内的内容指纹校验保证（违反会抛错，计为失败）。
- 正确性的代理指标：每道题的验证状态分布（工具验证 / 独立复核一致 / 待老师确认 / 存在未解决问题）。
  另导出随机抽样的题目表格，供人工逐题核对答案是否正确、是否适合年级（可选）。

运行：python -m exam_agent eval-e2e [--limit N] [--workers 4] [--resume 目录] [--no-edits]
结果目录 evals/results/e2e_<时间>/：results.jsonl（逐条，可断点续跑）、papers.db（全部试卷，可用网页查看）、
report.md、summary.json、audit_sample.csv。
"""
from __future__ import annotations

import csv
import json
import random
import re
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import PROJECT_ROOT
from ..domain import QType
from .review_eval import RESULTS_DIR, git_commit

DATASET = PROJECT_ROOT / "evals" / "e2e_requests.jsonl"
HOLDOUT = PROJECT_ROOT / "evals" / "e2e_requests_holdout.jsonl"
DATASETS = {"dev": DATASET, "holdout": HOLDOUT, "holdout2": PROJECT_ROOT / "evals" / "e2e_requests_holdout2.jsonl"}
DATASET_LABEL = {"dev": "开发集", "holdout": "留出集", "holdout2": "留出集 2"}
STATUS_ORDER = ["工具验证", "独立复核一致", "老师已确认", "待老师确认", "存在未解决问题", "未审核"]
EDIT_KINDS = ["restore", "resize", "ask", "patch", "constraint"]


@dataclass
class EditOutcome:
    kind: str
    message: str
    ok: bool = False
    action: str = ""
    reason: str = ""
    seconds: float = 0.0
    llm_calls: int = 0
    tokens: int = 0


@dataclass
class RunOutcome:
    id: str
    group: str
    grade: int
    text: str
    expect: str
    outcome: str = ""            # paper / reject / failure / error
    ok: bool = False
    reason: str = ""
    failure_step: str = ""
    seconds: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    paper_id: str = ""
    subject: str = ""
    grade_text: str = ""
    questions: int = 0
    total: int = 0
    status: dict = field(default_factory=dict)
    flagged: int = 0
    assumptions: list = field(default_factory=list)
    review_rounds: int = 0
    stop_reason: str = ""
    edits: list = field(default_factory=list)


def load_requests(path: Path = DATASET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _numbers_once(stem: str) -> list[int]:
    """题干中只出现一次的数（不含小题编号“（7）”“7.”），供“局部改字”使用。
    （试跑：选中了小题编号（7）里的 7，模型合理地追问，这是评测指令设计的问题。）"""
    stem = re.sub(r"[（(]\s*\d+\s*[)）]|^\s*\d+\s*[.、]", " ", stem, flags=re.M)
    nums = re.findall(r"(?<![\d.])(\d{1,3})(?![\d.])", stem)
    return [int(n) for n in nums if nums.count(n) == 1 and 2 <= int(n) <= 999]


def patch_edit(paper, rid: str, exclude: int | None = None):
    """局部改字指令：在当前卷面（第一次修改之后）选一道题题干中只出现一次的数，避开刚被换掉的第 k 题。
    （e2e 第一轮：按出卷时的卷面选数，而第一次修改已经换掉了那道题，数字早已不在题干里。）"""
    rng = random.Random(rid + "patch")
    qs = {q.slot_id: q for q in paper.questions}
    cands = [(s.position, _numbers_once(qs[s.slot_id].stem)) for s in sorted(paper.blueprint.slots,
                                                                           key=lambda s: s.position)
             if s.qtype in (QType.CALC, QType.FILL, QType.APPLICATION) and s.position != exclude]
    cands = [(p, ns) for p, ns in cands if ns]
    if not cands:
        return None
    j, ns = rng.choice(cands)
    a = rng.choice(ns)
    return ("patch", f"第 {j} 题题干里的 {a} 改成 {a + 1}", {"j": j, "new": str(a + 1)})


def plan_edits(paper, rid: str) -> list[tuple[str, str, dict]]:
    """由程序为这份卷选择两条修改指令及检查所需的参数（同一 id 每次选择相同）。
    局部改字的具体数字要等第一次修改完成后再选（见 patch_edit），这里先占位。"""
    rng = random.Random(rid)
    slots = sorted(paper.blueprint.slots, key=lambda s: s.position)
    qs = {q.slot_id: q for q in paper.questions}
    n = len(slots)
    k = rng.choice(slots).position
    edits = [("rewrite", f"第 {k} 题太难，换一道基础题，其他题别动", {"k": k})]
    kinds = list(EDIT_KINDS)
    if paper.request.subject != "数学":
        kinds = [x for x in kinds if x not in ("patch", "constraint")]
    kind = kinds[int(rid[1:]) % len(kinds)]
    if kind == "patch":
        if patch_edit(paper, rid, exclude=k) is not None:
            edits.append(("patch", "", {"k": k, "lazy": True}))
            return edits
        kind = "ask"
    if kind == "restore":
        edits.append(("restore", f"把第 {k} 题改回原来的", {"k": k}))
    elif kind == "resize":
        edits.append(("resize", f"把试卷扩大到 {n + 5} 题，原来的题保留", {"n": n + 5}))
    elif kind == "ask":
        j = rng.choice(slots).position
        edits.append(("ask", f"第 {j} 题答案为什么是这个？", {"j": j}))
    elif kind == "constraint":
        edits.append(("constraint", "计算题答案都要是整数", {}))
    return edits


def check_edit(kind: str, params: dict, before, res, store, paper_id: str) -> tuple[bool, str]:
    after = res.paper
    if kind == "ask":
        if res.action != "answer":
            return False, f"应当只回答，实际 {res.action}"
        return True, ""
    if res.action != "edit":
        if kind == "resize":
            # 按分值规则确实扩不了（如 10 道计算题每题 10 分已占满 100 分）时，向老师说明原因才是对的
            from ..agents.editor import EditPlan
            from ..editing import apply_ops

            dry = apply_ops(before, EditPlan.model_validate({"action": "edit", "reply": "", "ops": [
                {"op": "resize", "count": params["n"]}]}))
            if dry.problems:
                return True, "按分值规则无法扩充，正确地向老师说明了原因"
        return False, f"应当修改，实际 {res.action}：{res.reply[:60]}"
    n_expect = params.get("n", len(before.questions))
    if len(after.questions) != n_expect:
        return False, f"题数应为 {n_expect}，实际 {len(after.questions)}"
    if after.total_score != before.request.total_score:
        return False, f"总分应为 {before.request.total_score}，实际 {after.total_score}"
    if kind == "rewrite":
        sid = before.blueprint.slot_at(params["k"]).slot_id
        old = next(q for q in before.questions if q.slot_id == sid)
        new = next((q for q in after.questions if q.slot_id == sid), None)
        if new is None or new.content_fingerprint() == old.content_fingerprint():
            return False, f"第 {params['k']} 题没有被替换"
    if kind == "restore":
        sid = before.blueprint.slot_at(params["k"]).slot_id
        v1, _ = store.load(paper_id, 1)
        orig = next(q for q in v1.questions if q.slot_id == sid)
        new = next((q for q in after.questions if q.slot_id == sid), None)
        if new is None or new.content_fingerprint() != orig.content_fingerprint():
            return False, f"第 {params['k']} 题没有恢复成 v1 的内容"
    if kind == "patch":
        sid = before.blueprint.slot_at(params["j"]).slot_id
        new = next((q for q in after.questions if q.slot_id == sid), None)
        if new is None or params["new"] not in new.stem:
            return False, f"第 {params['j']} 题题干里没有出现 {params['new']}"
    return True, ""


def _status_counts(paper) -> dict:
    return dict(Counter(q.vstatus.value for q in paper.questions))


def run_one(req: dict, llm, store, grounding, do_edits: bool = True) -> RunOutcome:
    from ..agents.base import AgentFailure, UnsupportedRequest
    from ..editing import apply_edit
    from ..pipeline import create_paper

    r = RunOutcome(id=req["id"], group=req["group"], grade=req["grade"], text=req["text"], expect=req["expect"])
    t0 = time.time()
    try:
        res = create_paper(llm, store, req["text"], review=True, grounding=grounding)
    except UnsupportedRequest as e:
        r.outcome, r.reason = "reject", str(e)
    except AgentFailure as e:
        r.outcome, r.failure_step, r.reason = "failure", e.step, str(e)[:300]
    except Exception as e:     # noqa: BLE001  评测要记录所有异常，不能中断整批
        r.outcome, r.reason = "error", f"{type(e).__name__}: {e}"[:300]
        r.failure_step = traceback.extract_tb(e.__traceback__)[-1].name if e.__traceback__ else ""
    else:
        p = res.paper
        r.outcome, r.paper_id = "paper", p.paper_id
        r.subject, r.grade_text = p.request.subject, p.request.grade
        r.questions, r.total = len(p.questions), p.total_score
        r.status, r.flagged = _status_counts(p), sum(q.vstatus.value == "存在未解决问题" for q in p.questions)
        r.assumptions = res.assumptions
        r.llm_calls, r.tokens = res.totals["llm_calls"], res.totals["prompt_tokens"] + res.totals["completion_tokens"]
        if res.review:
            r.review_rounds, r.stop_reason = res.review.rounds, res.review.stop_reason
    r.seconds = round(time.time() - t0, 1)
    r.ok = r.outcome == r.expect
    if r.outcome != "paper" or not do_edits:
        return r
    for kind, msg, params in plan_edits(res.paper, req["id"]):
        before, _ = store.load(r.paper_id)
        if params.get("lazy"):
            planned = patch_edit(before, req["id"], exclude=params["k"])
            if planned is None:
                continue
            kind, msg, params = planned
        e = EditOutcome(kind=kind, message=msg)
        t1 = time.time()
        try:
            out = apply_edit(llm, store, r.paper_id, msg, review=True, grounding=grounding)
            e.action = out.action
            e.ok, e.reason = check_edit(kind, params, before, out, store, r.paper_id)
            e.llm_calls = out.totals.get("llm_calls", 0)
            e.tokens = out.totals.get("prompt_tokens", 0) + out.totals.get("completion_tokens", 0)
        except AgentFailure as ex:
            e.reason = f"失败（{ex.step}）：{str(ex)[:200]}"
        except Exception as ex:  # noqa: BLE001
            e.reason = f"异常 {type(ex).__name__}: {str(ex)[:200]}"
        e.seconds = round(time.time() - t1, 1)
        r.edits.append(asdict(e))
    return r


def _pct(a: int, b: int) -> str:
    return f"{a / b:.0%}（{a}/{b}）" if b else "-"


def summarize(rows: list[dict]) -> dict:
    papers = [r for r in rows if r["outcome"] == "paper"]
    expect_paper = [r for r in rows if r["expect"] == "paper"]
    expect_reject = [r for r in rows if r["expect"] == "reject"]
    status = Counter()
    for r in papers:
        status.update(r["status"])
    nq = sum(status.values())
    edits = [e for r in rows for e in r["edits"]]
    by_kind = {}
    for k in ["rewrite", *EDIT_KINDS]:
        es = [e for e in edits if e["kind"] == k]
        if es:
            by_kind[k] = {"n": len(es), "ok": sum(e["ok"] for e in es)}
    by_group = {}
    for g in ("数学", "英语", "边界"):
        rs = [r for r in rows if r["group"] == g]
        if rs:
            by_group[g] = {"n": len(rs), "ok": sum(r["ok"] for r in rs)}
    mean = lambda xs: sum(xs) / len(xs) if xs else 0
    return {
        "requests": len(rows),
        "create_ok": sum(r["ok"] for r in rows),
        "paper_success": {"n": len(expect_paper), "ok": sum(r["outcome"] == "paper" for r in expect_paper)},
        "reject_correct": {"n": len(expect_reject), "ok": sum(r["outcome"] == "reject" for r in expect_reject)},
        "failures": dict(Counter(f"{r['outcome']}:{r['failure_step'] or '-'}" for r in rows
                                 if r["outcome"] in ("failure", "error"))),
        "by_group": by_group,
        "papers_with_flag": sum(1 for r in papers if r["flagged"]),
        "questions": nq,
        "status": {k: status.get(k, 0) for k in STATUS_ORDER if status.get(k)},
        "edits": {"n": len(edits), "ok": sum(e["ok"] for e in edits)},
        "edits_by_kind": by_kind,
        "create_seconds": mean([r["seconds"] for r in papers]),
        "create_tokens": mean([r["tokens"] for r in papers]),
        "create_llm_calls": mean([r["llm_calls"] for r in papers]),
        "edit_seconds": mean([e["seconds"] for e in edits]),
        "edit_tokens": mean([e["tokens"] for e in edits]),
        "total_tokens": sum(r["tokens"] for r in rows) + sum(e["tokens"] for e in edits),
    }


KIND_LABEL = {"rewrite": "换一道基础题（第 k 题）", "restore": "改回原来的", "resize": "扩大题数（原题保留）",
              "ask": "提问（不应改卷）", "patch": "局部改字", "constraint": "加持续约束"}


def report(meta: dict, s: dict, rows: list[dict]) -> str:
    md = [f"# 端到端评测 {meta['stamp']}", "",
          f"代码版本 `{meta['commit']}`，模型 {meta['model']}，{meta.get('dataset', '开发集')} {s['requests']} 条请求，"
          f"并发 {meta['workers']}", "",
          "## 出卷", "",
          "| 指标 | 结果 |", "|---|---|",
          f"| 总体（结果符合预期） | {_pct(s['create_ok'], s['requests'])} |",
          f"| 应当出卷的请求，出卷成功 | {_pct(s['paper_success']['ok'], s['paper_success']['n'])} |",
          f"| 应当拒绝的请求，正确拒绝 | {_pct(s['reject_correct']['ok'], s['reject_correct']['n'])} |",
          *[f"| {g} | {_pct(v['ok'], v['n'])} |" for g, v in s["by_group"].items()],
          f"| 含“存在未解决问题”题目的试卷 | {s['papers_with_flag']} 份 |",
          f"| 平均每份：耗时 / 模型调用 / token | {s['create_seconds']:.0f} 秒 / {s['create_llm_calls']:.1f} 次 / "
          f"{s['create_tokens']:.0f} |", "",
          "失败分布：" + ("；".join(f"{k} {v} 次" for k, v in s["failures"].items()) or "无"), "",
          "## 题目验证状态（全部成功试卷）", "",
          "| 状态 | 题数 | 占比 |", "|---|---|---|",
          *[f"| {k} | {v} | {v / s['questions']:.1%} |" for k, v in s["status"].items()], "",
          "## 修改（程序自动选择指令）", "",
          f"总体成功 {_pct(s['edits']['ok'], s['edits']['n'])}；平均每次 {s['edit_seconds']:.0f} 秒、{s['edit_tokens']:.0f} token", "",
          "| 修改类型 | 成功 |", "|---|---|",
          *[f"| {KIND_LABEL[k]} | {_pct(v['ok'], v['n'])} |" for k, v in s["edits_by_kind"].items()], "",
          f"全部 token 合计：{s['total_tokens']}", "", "## 未通过明细", ""]
    for r in rows:
        if not r["ok"]:
            md.append(f"- 出卷 {r['id']}（{r['text']}）：期望 {r['expect']}，实际 {r['outcome']}；{r['reason'][:150]}")
        for e in r["edits"]:
            if not e["ok"]:
                md.append(f"- 修改 {r['id']}「{e['message']}」：{e['reason'][:150]}")
    return "\n".join(md)


def export_audit_sample(rows: list[dict], store, path: Path, n: int = 60, seed: int = 7) -> int:
    """随机抽题导出表格，供人工核对“答案是否正确、是否适合年级”（抽的是出卷时的 v1）。"""
    pool = []
    for r in rows:
        if r["outcome"] != "paper":
            continue
        p, _ = store.load(r["paper_id"], 1)
        for s in sorted(p.blueprint.slots, key=lambda s: s.position):
            q = next(x for x in p.questions if x.slot_id == s.slot_id)
            pool.append((r, p, s, q))
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["请求", "年级", "学科", "题号", "题型", "验证状态", "题干", "选项", "答案", "解析",
                    "答案正确？(是/否)", "适合该年级？(是/否)", "备注"])
        for r, p, s, q in sample:
            w.writerow([r["id"], p.request.grade, p.request.subject, s.position, q.qtype.value, q.vstatus.value,
                        q.stem, " | ".join(q.options), q.answer, q.analysis, "", "", ""])
    return len(sample)


def run_e2e(llm_factory, grounding=None, workers: int = 4, limit: int | None = None, resume: Path | None = None,
            do_edits: bool = True, sample: int = 60, model: str = "?", progress=None,
            holdout: bool = False, dataset: str | None = None) -> tuple[dict, Path]:
    from ..store import PaperStore

    dataset = dataset or ("holdout" if holdout else "dev")
    tag = "" if dataset == "dev" else f"_{dataset}"
    out = resume or RESULTS_DIR / f"e2e{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    store = PaperStore(out / "papers.db")
    results_path = out / "results.jsonl"
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    reqs = (load_requests(DATASETS[dataset]) if dataset != "dev" else load_requests())[: limit or None]
    todo = [q for q in reqs if q["id"] not in done]
    if grounding is not None:   # 检索器加锁：多线程共用同一个向量模型
        lock = threading.Lock()
        for sub in getattr(grounding, "by_subject", {}).values():
            orig = sub.retriever.search

            def locked(*a, _orig=orig, **k):
                with lock:
                    return _orig(*a, **k)
            sub.retriever.search = locked
    local = threading.local()
    write_lock = threading.Lock()

    def work(req):
        if not hasattr(local, "llm"):
            local.llm = llm_factory()
        return run_one(req, local.llm, store, grounding, do_edits)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, q): q for q in todo}
        for fut in as_completed(futures):
            row = asdict(fut.result())
            with write_lock, open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[row["id"]] = row
            if progress:
                progress(row)
    rows = [done[q["id"]] for q in reqs if q["id"] in done]
    s = summarize(rows)
    meta = {"stamp": out.name.removeprefix("e2e_"), "commit": git_commit(), "model": model, "workers": workers,
            "dataset": DATASET_LABEL[dataset]}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": s}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (out / "report.md").write_text(report(meta, s, rows), encoding="utf-8")
    if sample:
        export_audit_sample(rows, store, out / "audit_sample.csv", sample)
    return s, out
