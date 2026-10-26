"""审核效果对比实验。

问题：审核到底有没有用？用哪种审核方式最划算？
方法：40 道带标注的题（16 道正确 + 24 道各含一种已知缺陷），用四种配置检查，比较：
  - 检出率：有缺陷的题被标记出来的比例（按缺陷类别分别统计）
  - 误报率：正确的题被错误标记的比例
  - 成本：模型调用次数、token、耗时

四种配置：
  none        只有格式校验（不调用模型）——基线
  self        单个模型带着答案和解析审题（最常见的“让模型自己检查”做法）
  blind_tool  盲解 Agent（看不到答案）+ 精确求解器核对
  full        blind_tool + 教学审核 Agent（只计 must_fix）

题目是固定的，不经过命题员，所以实验只衡量审核模块本身；
命题、检索、界面的改动不会使结果失效。改审核逻辑或换模型后重跑即可（回归评测）。
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ..agents.base import Tracer, structured_call
from ..agents.reviewer import review_pedagogy
from ..agents.solver import blind_solve
from ..config import PROJECT_ROOT
from ..domain import BlueprintSlot, CognitiveLevel, Difficulty, ExamRequest, QType, Question
from ..invariants import check_question_format
from ..llm import LLMClient
from ..review import check_one

DATASET = PROJECT_ROOT / "evals" / "review_cases.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
CONFIGS = ["none", "self", "blind_tool", "full"]
CONFIG_LABEL = {
    "none": "仅格式校验",
    "self": "单模型自查（看得到答案）",
    "blind_tool": "盲解 + 求解器",
    "full": "盲解 + 求解器 + 教学审核",
}
CATEGORY_LABEL = {
    "clean": "正确题",
    "wrong_answer": "答案算错",
    "modeling_error": "列式错误",
    "ambiguous": "题干缺条件/歧义",
    "choice_key": "选择题答案键错误",
    "out_of_scope": "超出本单元",
    "unrealistic": "情境不合常理",
}
REQ = ExamRequest(stage="初中", grade="七年级", subject="数学", topic="一元一次方程（七年级上册）",
                  question_count=10, total_score=100)


class Case(BaseModel):
    id: str
    category: str
    qtype: QType
    stem: str
    options: list[str] = Field(default_factory=list)
    answer: str
    analysis: str
    knowledge_point: str
    difficulty: Difficulty = Difficulty.BASIC
    truth: dict | None = None
    scene: str = ""
    note: str = ""

    @property
    def defective(self) -> bool:
        return self.category != "clean"

    def to_question(self) -> Question:
        return Question(qid=self.id, slot_id=self.id, qtype=self.qtype, score=5,
                        knowledge_point=self.knowledge_point, difficulty=self.difficulty,
                        stem=self.stem, options=self.options, answer=self.answer, analysis=self.analysis)

    def to_slot(self, position: int) -> BlueprintSlot:
        return BlueprintSlot(slot_id=self.id, position=position, qtype=self.qtype, score=5,
                             knowledge_point=self.knowledge_point, difficulty=self.difficulty,
                             cognitive_level=CognitiveLevel.MASTER, intent=self.knowledge_point)


def load_cases(path: Path = DATASET) -> list[Case]:
    with open(path, encoding="utf-8") as f:
        return [Case.model_validate_json(line) for line in f if line.strip()]


def group_key(c: Case) -> str:
    """有情境的按情境，否则按知识点。同组的题互相“重复”，不能放进同一批。"""
    return c.scene or c.knowledge_point


def make_batches(cases: list[Case], size: int, seed: int) -> list[list[Case]]:
    """模拟一份份试卷：打乱后按分组键排序，再轮流发牌，让同情境、同知识点的题落进不同批次。
    数据集里每道题应被独立判断；如果两道考同一知识点的题被拼进同一份“试卷”，
    教学审核指出“重复”其实是对的，却会被记成误报。
    （实验 1 中 4 次误报有 3 次是这个原因，见 evals/results 的记录。）"""
    rng = random.Random(seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=group_key)  # 稳定排序：同组的题挨在一起，轮流发牌后分散到不同批
    n = max(1, -(-len(cases) // size))
    batches: list[list[Case]] = [[] for _ in range(n)]
    for i, c in enumerate(shuffled):
        batches[i % n].append(c)
    return batches


# ---------------- 基线：单模型带着答案自查 ----------------

class SelfIssue(BaseModel):
    id: str
    problem: str


class SelfOutput(BaseModel):
    issues: list[SelfIssue]


SELF_SYSTEM = """你是一名中学数学教研员，请审核下面这批题目（含答案和解析）。
找出有问题的题：答案或解析错误、题干缺条件或有歧义、选择题答案错误、超出该年级该单元的范围、情境不合常理等。
没有问题的题不要列出。只输出 json，示例：{"issues": [{"id": "c01", "problem": "答案应为 x = 3"}]}"""


def self_check(llm: LLMClient, batch: list[Case], tracer: Tracer) -> dict[str, list[str]]:
    lines = [f"学段年级：{REQ.stage}{REQ.grade}，单元：{REQ.topic}"]
    for c in batch:
        lines.append(f"[{c.id}]（{c.qtype.value}）{c.stem}")
        for letter, opt in zip("ABCD", c.options):
            lines.append(f"    {letter}. {opt}")
        lines.append(f"    答案：{c.answer}；解析：{c.analysis}")
    ids = {c.id for c in batch}
    out = structured_call(llm, step="eval_self_check", system=SELF_SYSTEM, user="\n".join(lines),
                          schema=SelfOutput, tracer=tracer).value
    flagged: dict[str, list[str]] = defaultdict(list)
    for i in out.issues:
        if i.id in ids:
            flagged[i.id].append(i.problem)
    return flagged


# ---------------- 运行一个配置 ----------------

@dataclass
class RunResult:
    config: str
    repeat: int
    flagged: dict[str, list[str]] = field(default_factory=dict)  # case id → 理由
    blind: dict[str, dict] = field(default_factory=dict)         # case id → 盲解的方程、答案、发现的问题
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def run_config(llm: LLMClient, config: str, cases: list[Case], repeat: int, batch_size: int = 10) -> RunResult:
    tracer = Tracer(store=None)
    res = RunResult(config=config, repeat=repeat)
    t0 = time.perf_counter()
    for batch in make_batches(cases, batch_size, seed=repeat):
        try:
            if config == "none":
                for c in batch:
                    issues = check_question_format(c.qtype, c.stem, c.options, c.answer, c.id)
                    if issues:
                        res.flagged[c.id] = [str(v) for v in issues]
            elif config == "self":
                res.flagged.update(self_check(llm, batch, tracer))
            elif config in ("blind_tool", "full"):
                questions = {c.id: c.to_question() for c in batch}
                blind = blind_solve(llm, list(questions.values()), tracer)
                for cid, b in blind.items():
                    res.blind[cid] = b.model_dump()
                for c in batch:
                    findings, _ = check_one(questions[c.id], blind.get(c.id), REQ)
                    reasons = [f"[{f.source}] {f.message}" for f in findings if f.severity == "must_fix"]
                    if reasons:
                        res.flagged.setdefault(c.id, []).extend(reasons)
                if config == "full":
                    slots = {c.id: c.to_slot(i) for i, c in enumerate(batch, start=1)}
                    for issue in review_pedagogy(llm, REQ, slots, questions, set(questions), tracer):
                        if issue.severity == "must_fix":
                            res.flagged.setdefault(issue.slot_id, []).append(
                                f"[pedagogy:{issue.category}] {issue.message}")
            else:
                raise ValueError(f"未知配置 {config}")
        except Exception as e:  # 一批失败不拖垮整个实验，记录下来并计入报告
            res.errors.append(f"{type(e).__name__}: {str(e)[:200]}")
    res.seconds = round(time.perf_counter() - t0, 1)
    res.llm_calls = tracer.totals["llm_calls"]
    res.prompt_tokens = tracer.totals["prompt_tokens"]
    res.completion_tokens = tracer.totals["completion_tokens"]
    return res


# ---------------- 指标 ----------------

def metrics(cases: list[Case], run: RunResult) -> dict:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for c in cases:
        by_cat[c.category].append(c.id in run.flagged)
    defective = [c for c in cases if c.defective]
    clean = [c for c in cases if not c.defective]
    tp = sum(c.id in run.flagged for c in defective)
    fp = sum(c.id in run.flagged for c in clean)
    return {
        "recall": tp / len(defective) if defective else 0.0,
        "false_positive_rate": fp / len(clean) if clean else 0.0,
        "precision": tp / (tp + fp) if tp + fp else None,  # 一道都没标记时精确率无意义
        "detected": tp, "defective": len(defective), "false_positives": fp, "clean": len(clean),
        "per_category": {cat: sum(v) / len(v) for cat, v in by_cat.items()},
        "llm_calls": run.llm_calls, "tokens": run.prompt_tokens + run.completion_tokens,
        "seconds": run.seconds, "errors": len(run.errors),
    }


def summarize(per_run: list[dict]) -> dict:
    """多次重复的均值与范围（模型输出有随机性，单次结果不可靠）。"""
    def agg(key):
        vals = [m[key] for m in per_run if m[key] is not None]
        if not vals:
            return {"mean": None, "min": None, "max": None}
        return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
    cats = per_run[0]["per_category"].keys()
    return {
        **{k: agg(k) for k in ("recall", "false_positive_rate", "precision", "llm_calls", "tokens", "seconds")},
        "per_category": {c: sum(m["per_category"][c] for m in per_run) / len(per_run) for c in cats},
        "errors": sum(m["errors"] for m in per_run),
    }


# ---------------- 报告与保存 ----------------

def git_commit() -> str:
    """当前代码版本；有未提交的改动时加上 -dirty，避免两次实验记成同一个版本却跑的是不同代码。"""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                        timeout=5).stdout.strip()
        head = run("rev-parse", "--short", "HEAD") or "unknown"
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def rng_pct(s: dict) -> str:
    return pct(s["mean"]) if s["min"] == s["max"] else f"{pct(s['mean'])}（{pct(s['min'])}–{pct(s['max'])}）"


def render_report(meta: dict, summary: dict[str, dict], runs: dict[str, list[RunResult]], cases: list[Case]) -> str:
    n_def = sum(c.defective for c in cases)
    n_clean = len(cases) - n_def
    lines = [
        "# 审核效果对比实验报告", "",
        f"- 时间：{meta['time']}　模型：`{meta['model']}`　代码版本：`{meta['git_commit']}`",
        f"- 数据集：{len(cases)} 题（{n_def} 道含已知缺陷，{n_clean} 道正确），每批 {meta['batch_size']} 题，"
        f"每种配置重复 {meta['repeats']} 次，括号内为最小–最大值", "",
        "## 总体结果", "",
        "| 配置 | 检出率 | 误报率 | 精确率 | 每 40 题调用次数 | 每 40 题 token | 每 40 题耗时（秒） |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cfg, s in summary.items():
        lines.append(f"| {CONFIG_LABEL[cfg]} | {rng_pct(s['recall'])} | {rng_pct(s['false_positive_rate'])} | "
                     f"{pct(s['precision']['mean']) if s['precision']['mean'] is not None else '-'} | {s['llm_calls']['mean']:.0f} | {s['tokens']['mean']:.0f} | "
                     f"{s['seconds']['mean']:.0f} |")
    cats = [c for c in CATEGORY_LABEL if c != "clean"]
    counts = {cat: sum(c.category == cat for c in cases) for cat in cats}
    lines += ["", "## 各类缺陷的检出率", "",
              "| 缺陷类别 | 题数 | " + " | ".join(CONFIG_LABEL[c] for c in summary) + " |",
              "| --- | --- | " + " | ".join("---" for _ in summary) + " |"]
    for cat in cats:
        lines.append(f"| {CATEGORY_LABEL[cat]} | {counts[cat]} | " +
                     " | ".join(pct(summary[cfg]["per_category"].get(cat, 0)) for cfg in summary) + " |")

    lines += ["", "## 逐题结果（命中次数 / 重复次数）", "",
              "| 题目 | 类别 | 说明 | " + " | ".join(CONFIG_LABEL[c] for c in summary) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in summary) + " |"]
    for c in cases:
        hits = [f"{sum(c.id in r.flagged for r in runs[cfg])}/{len(runs[cfg])}" for cfg in summary]
        lines.append(f"| {c.id} | {CATEGORY_LABEL[c.category]} | {c.note or '-'} | " + " | ".join(hits) + " |")
    errs = [(cfg, e) for cfg, rs in runs.items() for r in rs for e in r.errors]
    if errs:
        lines += ["", "## 运行错误", ""] + [f"- {CONFIG_LABEL[cfg]}：{e}" for cfg, e in errs]
    lines += ["", "> 检出率 = 被标记的缺陷题 / 缺陷题总数；误报率 = 被标记的正确题 / 正确题总数；",
              "> 精确率 = 被标记的题中确实有缺陷的比例。完整原始数据见同名 JSON 文件。"]
    return "\n".join(lines) + "\n"


def save(meta: dict, summary: dict, runs: dict[str, list[RunResult]], cases: list[Case],
         out_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta["stamp"]
    raw = {
        "meta": meta, "summary": summary,
        "runs": {cfg: [r.__dict__ for r in rs] for cfg, rs in runs.items()},
        "cases": [c.model_dump(mode="json") for c in cases],
    }
    json_path = out_dir / f"review_eval_{stamp}.json"
    md_path = out_dir / f"review_eval_{stamp}.md"
    json_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_report(meta, summary, runs, cases), encoding="utf-8")

    hist = out_dir / "history.csv"
    new = not hist.exists()
    with open(hist, "a", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 打开不乱码
        w = csv.writer(f)
        if new:
            w.writerow(["时间", "代码版本", "模型", "配置", "重复次数", "检出率", "误报率", "精确率",
                        "调用次数/40题", "token/40题", "耗时秒/40题"])
        for cfg, s in summary.items():
            w.writerow([meta["time"], meta["git_commit"], meta["model"], cfg, meta["repeats"],
                        f"{s['recall']['mean']:.3f}", f"{s['false_positive_rate']['mean']:.3f}",
                        "" if s["precision"]["mean"] is None else f"{s['precision']['mean']:.3f}", f"{s['llm_calls']['mean']:.0f}",
                        f"{s['tokens']['mean']:.0f}", f"{s['seconds']['mean']:.0f}"])
    return json_path, md_path


def run_experiment(llm: LLMClient, configs: list[str], repeats: int, batch_size: int = 10,
                   limit: int | None = None, progress=print, out_dir: Path = RESULTS_DIR):
    cases = load_cases()
    if limit:
        cases = cases[:limit]
    runs: dict[str, list[RunResult]] = {}
    for cfg in configs:
        runs[cfg] = []
        for rep in range(repeats):
            progress(f"运行 {CONFIG_LABEL[cfg]}，第 {rep + 1}/{repeats} 次 ...")
            r = run_config(llm, cfg, cases, rep, batch_size)
            runs[cfg].append(r)
            m = metrics(cases, r)
            progress(f"  检出 {m['detected']}/{m['defective']}，误报 {m['false_positives']}/{m['clean']}，"
                     f"{m['llm_calls']} 次调用，{m['tokens']} token，{m['seconds']} 秒"
                     + (f"，{m['errors']} 批出错" if m["errors"] else ""))
    summary = {cfg: summarize([metrics(cases, r) for r in rs]) for cfg, rs in runs.items()}
    now = datetime.now()
    meta = {"time": now.strftime("%Y-%m-%d %H:%M"), "stamp": now.strftime("%Y%m%d_%H%M%S"),
            "model": llm.describe() if hasattr(llm, "describe") else llm.model, "git_commit": git_commit(), "repeats": repeats,
            "batch_size": batch_size, "configs": configs, "n_cases": len(cases)}
    paths = save(meta, summary, runs, cases, out_dir)
    return summary, paths
