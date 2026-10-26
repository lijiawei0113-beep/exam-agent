"""阶段六：网页服务。业务层（ExamService）离线测试；HTTP 接口在装了 fastapi + httpx 的环境里测试。"""
import json

from exam_agent.llm import ScriptedLLM
from exam_agent.store import PaperStore
from exam_agent.web.service import ExamService, Limiter, progress_event

from fixtures import GOOD_BLUEPRINT, Q1, Q2, REQUEST_REPLY, q

j = lambda o: json.dumps(o, ensure_ascii=False)
QS = j({"questions": [Q1, Q2, q("S3", "解方程：3x + 5 = 2x - 1", "x = -6"), q("S4", "解方程：2(x - 1) = x + 3", "x = 5")]})


def service(replies, **kw):
    llm = ScriptedLLM(replies)
    return ExamService(PaperStore(":memory:"), lambda: llm, review=False, limiter=Limiter(0, 0), **kw), llm


def test_create_streams_progress_then_result():
    svc, _ = service([REQUEST_REPLY, GOOD_BLUEPRINT, QS])
    events = list(svc.create("出一份一元一次方程试卷，4 道题，满分 30 分"))
    labels = [e["label"] for e in events if e["type"] == "progress"]
    assert labels[0] == "收到出卷要求" and "规划蓝图（双向细目表）" in labels and labels[-1] == "保存为新版本"
    res = events[-1]
    assert res["type"] == "result" and res["kind"] == "created"
    p = res["paper"]
    assert p["version"] == 1 and [s["title"][:4] for s in p["sections"]] == ["一、选择", "二、填空", "三、计算"]
    assert [q["no"] for s in p["sections"] for q in s["questions"]] == [1, 2, 3, 4]
    assert p["status_counts"] == {"未审核": 4} and p["history"][0]["operation"] == "create"
    json.dumps(res, ensure_ascii=False)   # 整个结果可以序列化成 SSE


def test_edit_confirm_undo_flow():
    edit_plan = j({"action": "edit", "ops": [{"op": "rewrite", "positions": [3], "difficulty": "基础",
                                              "instruction": "换一道基础题"}], "reply": "好的，只换第 3 题。"})
    svc, _ = service([REQUEST_REPLY, GOOD_BLUEPRINT, QS, edit_plan,
                      j({"questions": [q("S3", "解方程：x + 2 = 5", "x = 3")]})])
    pid = list(svc.create("七年级一元一次方程小测，4 道题，满分 30 分"))[-1]["paper"]["paper_id"]
    res = list(svc.edit(pid, "第 3 题太难，换一道基础题，其他题别动"))[-1]
    assert res["kind"] == "edit" and res["paper"]["version"] == 2
    assert [c["slot_id"] for c in res["changes"] if c["kind"] == "content"] == ["S3"]
    p = svc.confirm(pid, [3], "已核对")
    third = [q for s in p["sections"] for q in s["questions"] if q["no"] == 3][0]
    assert p["version"] == 3 and third["status_key"] == "teacher"
    p = svc.undo(pid)
    assert p["version"] == 4 and [h["operation"] for h in p["history"]][-1] == "revert"


def test_failure_is_reported_not_raised():
    svc, _ = service([REQUEST_REPLY, "不是 json", "还不是", "也不是"])
    events = list(svc.create("七年级一元一次方程小测，4 道题，满分 30 分"))
    assert events[-1]["type"] == "error" and "试卷保持原样" in events[-1]["message"]
    assert svc.limiter._running == 0                     # 失败后释放并发名额


def test_rate_limit_protects_api_cost():
    lim = Limiter(per_hour=2, concurrency=0)
    lim.acquire("1.2.3.4"); lim.release(); lim.acquire("1.2.3.4"); lim.release()
    svc = ExamService(PaperStore(":memory:"), lambda: None, limiter=lim)
    ev = list(svc.create("七年级一元一次方程小测，4 道题，满分 30 分", client="1.2.3.4"))
    assert ev == [{"type": "error", "message": ev[0]["message"]}] and "一小时内" in ev[0]["message"]
    one = Limiter(per_hour=0, concurrency=1)
    one.acquire("a")
    try:
        one.acquire("b")
        assert False
    except Exception as e:
        assert "较多" in str(e)


def test_progress_labels_are_human_readable():
    assert progress_event("planner_scores", {}) is None
    ev = progress_event("planner", {"attempt": 2, "problems": ["题目数量为 3，要求恰好 4 道"]})
    assert ev["label"].startswith("规划蓝图") and "题目数量为 3" in ev["detail"]
    assert progress_event("review_done", {"stop_reason": "passed"})["detail"] == "全部通过"


def test_http_api():
    """需要 fastapi 和 httpx（requirements.txt 已包含）；缺少时跳过。"""
    try:
        from fastapi.testclient import TestClient
        from exam_agent.web.app import create_app
    except ImportError:
        return
    svc, _ = service([REQUEST_REPLY, GOOD_BLUEPRINT, QS])
    client = TestClient(create_app(svc))
    assert client.get("/api/health").json()["ok"] is True
    assert "对话式试卷编制助手" in client.get("/").text
    with client.stream("POST", "/api/papers", json={"text": "出一份一元一次方程试卷，4 道题，满分 30 分"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    pid = events[-1]["paper"]["paper_id"]
    assert client.get(f"/api/papers/{pid}").json()["version"] == 1
    assert client.get("/api/papers").json()[0]["paper_id"] == pid
    assert client.get("/api/papers/p_none").status_code == 404
    assert client.post(f"/api/papers/{pid}/confirm", json={"positions": [99]}).status_code == 404
    assert client.post(f"/api/papers/{pid}/confirm", json={"positions": [1]}).json()["version"] == 2
    r = client.get(f"/api/papers/{pid}/pdf/student")
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    assert client.post("/api/papers", json={"text": ""}).status_code == 422
    assert client.delete(f"/api/papers/{pid}").json() == {"deleted": pid}
    assert client.delete(f"/api/papers/{pid}").status_code == 404
    assert client.delete("/api/papers").json() == {"deleted_count": 0}


def test_delete_and_clear(tmp_path):
    svc, _ = service([REQUEST_REPLY, GOOD_BLUEPRINT, QS, REQUEST_REPLY, GOOD_BLUEPRINT, QS], out_dir=tmp_path)
    a = list(svc.create("第一份，4 道题，满分 30 分"))[-1]["paper"]["paper_id"]
    b = list(svc.create("第二份，4 道题，满分 30 分"))[-1]["paper"]["paper_id"]
    svc.export(a)
    assert list(tmp_path.glob(f"{a}_v*.pdf"))
    assert svc.delete(a) == {"deleted": a}
    assert [r["paper_id"] for r in svc.list()] == [b] and not list(tmp_path.glob(f"{a}_v*.pdf"))
    assert svc.store.events(paper_id=a) == []              # 运行轨迹一并删除
    try:
        svc.get(a)
        assert False
    except KeyError:
        pass
    assert svc.clear_all() == {"deleted_count": 1} and svc.list() == []
