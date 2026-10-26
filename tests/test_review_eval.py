"""对比实验：先检查数据集本身是否标注正确，再检查指标计算和流程。"""
import json

from exam_agent.experiments.review_eval import (CONFIGS, RunResult, load_cases, make_batches, metrics,
                                                 run_config, run_experiment)
from exam_agent.llm import ScriptedLLM
from exam_agent.tools.linear import ToolError, extract_numbers, solve


def test_dataset_shape():
    cases = load_cases()
    assert len(cases) == 40 and len({c.id for c in cases}) == 40
    counts = {}
    for c in cases:
        counts[c.category] = counts.get(c.category, 0) + 1
    assert counts == {"clean": 16, "wrong_answer": 8, "modeling_error": 4, "ambiguous": 4,
                      "choice_key": 3, "out_of_scope": 3, "unrealistic": 2}


def test_dataset_labels_are_true():
    """标注本身要经得起求解器检验：正确题的答案真的对，算错题的答案真的错。"""
    for c in load_cases():
        if not c.truth:
            continue
        value = solve(c.truth["equations"], c.truth["unknowns"])[c.truth["target"]]
        stated = extract_numbers(c.answer)
        if c.category in ("clean", "unrealistic"):
            assert value in stated, f"{c.id} 标为计算正确，但正确值是 {value}"
        elif c.category in ("wrong_answer", "modeling_error"):
            assert value not in stated, f"{c.id} 标为有错，但答案其实是对的"


def test_dataset_choice_labels():
    by_id = {c.id: c for c in load_cases()}
    # 答案键错误的题：正确选项与答案键不同
    truth = {"k01": "C", "k02": "C", "k03": "B"}
    for cid, right in truth.items():
        assert by_id[cid].answer != right
    assert by_id["c02"].options[1] == "x = 3" and by_id["c02"].answer == "B"


def test_batches_never_put_same_scene_or_knowledge_point_together():
    """实验 1 的教训：c01 和 k02 题干几乎一样，被拼进同一批后，教学审核指出“重复”，被错记为误报。"""
    from exam_agent.experiments.review_eval import group_key
    cases = load_cases()
    for seed in range(5):
        batches = make_batches(cases, 10, seed=seed)
        assert sorted(c.id for b in batches for c in b) == sorted(c.id for c in cases)
        for b in batches:
            keys = [group_key(c) for c in b]
            assert len(keys) == len(set(keys)), f"seed={seed} 同一批中有重复的情境或知识点：{keys}"


def test_metrics():
    cases = load_cases()
    run = RunResult(config="x", repeat=0, flagged={"w01": ["错"], "c01": ["误报"]})
    m = metrics(cases, run)
    assert m["detected"] == 1 and m["false_positives"] == 1
    assert abs(m["recall"] - 1 / 24) < 1e-9 and abs(m["false_positive_rate"] - 1 / 16) < 1e-9
    assert m["per_category"]["wrong_answer"] == 1 / 8


def test_format_only_baseline_detects_nothing():
    cases = load_cases()
    m = metrics(cases, run_config(ScriptedLLM([]), "none", cases, repeat=0))
    assert m["detected"] == 0 and m["false_positives"] == 0 and m["llm_calls"] == 0


def test_blind_tool_config_on_small_subset(tmp_path):
    """3 道题一批：c01（正确选择题）、c02（正确选择题）、c03（正确选择题）——流程能跑通、能保存。"""
    reply = json.dumps({"solutions": [
        {"slot_id": "c01", "choice": "B"}, {"slot_id": "c02", "choice": "B"}, {"slot_id": "c03", "choice": "A"}]})
    summary, (json_path, md_path) = run_experiment(
        ScriptedLLM([reply]), ["blind_tool"], repeats=1, batch_size=10, limit=3,
        progress=lambda s: None, out_dir=tmp_path)
    # c03 盲解选了 A 而答案键是 B → 被标记（这里是误报，用于验证统计）
    assert summary["blind_tool"]["false_positive_rate"]["mean"] == 1 / 3
    assert json_path.exists() and md_path.exists() and (tmp_path / "history.csv").exists()
    assert "盲解 + 求解器" in md_path.read_text(encoding="utf-8")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["runs"]["blind_tool"][0]["blind"]["c03"]["choice"] == "A"   # 盲解输出被保存，便于分析


def test_all_configs_are_known():
    assert CONFIGS == ["none", "self", "blind_tool", "full"]
