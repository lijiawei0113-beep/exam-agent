"""检索评测：同一批查询，比较 BM25 / 向量 / RRF 混合 / 混合 + 重排。

数据集 evals/retrieval_cases.jsonl：每条是“老师可能怎么说” → 正确的知识点或课标条目 id。
按查询类型分组统计，看每种方法在哪类查询上强、在哪类上弱：
  term          教学术语原词（去分母、合并同类项）
  paraphrase    换说法 / 口语（“把带x的项挪到左边”）
  typo          错别字（“去分毋”）
  request       老师出卷原话，夹杂题数、题型等无关词
  out_of_scope  超纲内容，应当检索到后续年级的知识点（供超纲检查使用）
  standard      查课标条目
指标
  Hit@1     第一条就对的比例
  Recall@k  正确答案出现在前 k 条中的比例（一条查询有多个正确答案时按比例计）
  MRR       第一个正确答案名次的倒数的平均值（第 1 名得 1，第 2 名得 0.5……前 10 名以外得 0）
  延迟      每条查询的平均耗时（毫秒）
检索是确定性的，不需要重复多次；改分词、换模型、改知识库后重跑即可比较。
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from ..config import PROJECT_ROOT
from ..retrieval import MODE_LABEL, Retriever
from .review_eval import RESULTS_DIR, git_commit

DATASET = PROJECT_ROOT / "evals" / "retrieval_cases.jsonl"
TAG_LABEL = {"term": "术语原词", "paraphrase": "换说法/口语", "typo": "错别字", "request": "出卷原话",
             "out_of_scope": "超纲内容", "standard": "课标条目"}
DEPTH = 10


class RCase(BaseModel):
    id: str
    tag: str
    query: str
    gold: list[str]
    kind: str = "kp"


def load_rcases(path: Path = DATASET) -> list[RCase]:
    with open(path, encoding="utf-8") as f:
        return [RCase(**json.loads(line)) for line in f if line.strip()]


def run_mode(retriever: Retriever, mode: str, cases: list[RCase]) -> list[dict]:
    rows = []
    for c in cases:
        s = time.perf_counter()
        hits = retriever.search(c.query, DEPTH, mode, where={"kind": c.kind})
        ms = (time.perf_counter() - s) * 1000
        ids = [h.id for h in hits]
        first = next((r for r, i in enumerate(ids, 1) if i in c.gold), None)
        rows.append({"id": c.id, "tag": c.tag, "top": ids[:5], "first_gold_rank": first, "ms": round(ms, 2),
                     **{f"recall@{k}": len(set(ids[:k]) & set(c.gold)) / len(c.gold) for k in (1, 3, 5)}})
    return rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {"n": len(rows),
            "hit@1": sum(r["first_gold_rank"] == 1 for r in rows) / n,
            "recall@3": sum(r["recall@3"] for r in rows) / n,
            "recall@5": sum(r["recall@5"] for r in rows) / n,
            "mrr": sum(1 / r["first_gold_rank"] for r in rows if r["first_gold_rank"]) / n,
            "ms": sum(r["ms"] for r in rows) / n}


def summarize(results: dict[str, list[dict]]) -> dict:
    out = {}
    for mode, rows in results.items():
        out[mode] = aggregate(rows)
        out[mode]["per_tag"] = {t: aggregate([r for r in rows if r["tag"] == t])
                                for t in TAG_LABEL if any(r["tag"] == t for r in rows)}
    return out


def render_report(meta: dict, summary: dict, results: dict, cases: list[RCase]) -> str:
    pct = lambda x: f"{x:.0%}"
    modes = list(summary)
    lines = [f"# 检索评测 {meta['time']}", "",
             f"- 代码版本：{meta['git_commit']}；查询 {meta['n_cases']} 条；知识库 {meta['n_docs']} 条",
             f"- 向量模型：{meta['embedder']}" + ("" if meta["semantic"] or meta["embedder"] == "无"
                                               else "（无语义能力，仅流程测试，向量结果不能用于结论）"),
             f"- 重排模型：{meta['reranker']}", "",
             "## 总体", "", "| 方法 | Hit@1 | Recall@3 | Recall@5 | MRR | 平均延迟 ms |", "| --- | --- | --- | --- | --- | --- |"]
    for m in modes:
        s = summary[m]
        lines.append(f"| {MODE_LABEL[m]} | {pct(s['hit@1'])} | {pct(s['recall@3'])} | {pct(s['recall@5'])} | "
                     f"{s['mrr']:.3f} | {s['ms']:.1f} |")
    lines += ["", "## 按查询类型：Recall@5（Hit@1）", "",
              "| 类型 | 条数 | " + " | ".join(MODE_LABEL[m] for m in modes) + " |",
              "| --- | --- | " + " | ".join("---" for _ in modes) + " |"]
    for t, label in TAG_LABEL.items():
        if t not in summary[modes[0]]["per_tag"]:
            continue
        cells = [f"{pct(summary[m]['per_tag'][t]['recall@5'])}（{pct(summary[m]['per_tag'][t]['hit@1'])}）" for m in modes]
        lines.append(f"| {label} | {summary[modes[0]]['per_tag'][t]['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## 没有排在第 1 名的查询（正确答案名次，- 表示前 10 名都没有）", "",
              "| 查询 | 类型 | 正确答案 | " + " | ".join(MODE_LABEL[m] for m in modes) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in modes) + " |"]
    by_mode = {m: {r["id"]: r for r in results[m]} for m in modes}
    for c in cases:
        ranks = [by_mode[m][c.id]["first_gold_rank"] for m in modes]
        if all(r == 1 for r in ranks):
            continue
        lines.append(f"| {c.query} | {TAG_LABEL[c.tag]} | {'、'.join(c.gold)} | " +
                     " | ".join(str(r) if r else "-" for r in ranks) + " |")
    lines += ["", "> Recall@k：正确答案在前 k 条中的比例；Hit@1：第一条就正确的比例；",
              "> MRR：第一个正确答案名次倒数的平均。完整的逐条结果见同名 JSON 文件。"]
    return "\n".join(lines) + "\n"


def run_retrieval_eval(retriever: Retriever, modes: list[str] | None = None, out_dir: Path = RESULTS_DIR,
                       cases: list[RCase] | None = None):
    cases = cases if cases is not None else load_rcases()
    modes = modes or retriever.available_modes()
    for c in cases[:1]:  # 预热：首次调用会加载模型，不计入延迟
        for m in modes:
            retriever.search(c.query, 1, m)
    results = {m: run_mode(retriever, m, cases) for m in modes}
    summary = summarize(results)
    now = datetime.now()
    emb, rer = retriever.embedder, retriever.reranker
    meta = {"time": now.strftime("%Y-%m-%d %H:%M"), "stamp": now.strftime("%Y%m%d_%H%M%S"),
            "git_commit": git_commit(), "n_cases": len(cases), "n_docs": len(retriever.docs),
            "embedder": emb.name if emb else "无", "semantic": bool(emb and getattr(emb, "semantic", True)),
            "reranker": rer.name if rer else "无", "modes": modes}
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"retrieval_eval_{meta['stamp']}.json"
    md_path = out_dir / f"retrieval_eval_{meta['stamp']}.md"
    json_path.write_text(json.dumps({"meta": meta, "summary": summary, "results": results,
                                     "cases": [c.model_dump() for c in cases]}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    md_path.write_text(render_report(meta, summary, results, cases), encoding="utf-8")
    hist = out_dir / "retrieval_history.csv"
    new = not hist.exists()
    with open(hist, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["时间", "代码版本", "向量模型", "重排模型", "方法", "Hit@1", "Recall@3", "Recall@5", "MRR", "延迟ms"])
        for m, s in summary.items():
            w.writerow([meta["time"], meta["git_commit"], meta["embedder"], meta["reranker"], m,
                        f"{s['hit@1']:.3f}", f"{s['recall@3']:.3f}", f"{s['recall@5']:.3f}", f"{s['mrr']:.3f}", f"{s['ms']:.1f}"])
    return summary, (json_path, md_path)
