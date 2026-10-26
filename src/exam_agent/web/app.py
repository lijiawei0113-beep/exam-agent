"""网页服务（FastAPI）。

运行：python -m exam_agent web            # 浏览器打开 http://127.0.0.1:8000
接口一览：
  POST /api/papers                     出卷（SSE 流式返回进度，最后一条是结果）
  POST /api/papers/{id}/messages       对话修改（SSE）
  GET  /api/papers                     试卷列表
  GET  /api/papers/{id}?version=N      查看试卷
  POST /api/papers/{id}/confirm        老师确认题目 {"positions": [3, 7]}
  POST /api/papers/{id}/undo           撤销上一次修改
  DELETE /api/papers/{id}              删除一份试卷（全部版本）
  DELETE /api/papers                   清空全部试卷
  GET  /api/papers/{id}/pdf/{kind}     下载 PDF，kind = student / answers
  GET  /api/health                     健康检查
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..store import VersionConflict
from .service import ExamService

STATIC = Path(__file__).parent / "static"


class TextIn(BaseModel):
    text: str = Field(min_length=2, max_length=500)


class ConfirmIn(BaseModel):
    positions: list[int] = Field(min_length=1)
    note: str = ""


def _client(request: Request) -> str:
    """经过隧道 / 反向代理时取真实来源地址，用于限流。"""
    for h in ("cf-connecting-ip", "x-forwarded-for"):
        if request.headers.get(h):
            return request.headers[h].split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sse(events):
    for ev in events:
        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"


def create_app(service: ExamService) -> FastAPI:
    app = FastAPI(title="exam-agent", description="教师参与审核的中小学对话式试卷编制助手")

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"") or "找不到这份试卷或这道题")
        except VersionConflict as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        g = service.grounding
        return {"ok": True, "knowledge_base": g.size() if g is not None and hasattr(g, "size") else "未加载"}

    @app.post("/api/papers")
    def create(body: TextIn, request: Request):
        return StreamingResponse(_sse(service.create(body.text, _client(request))), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/papers/{paper_id}/messages")
    def edit(paper_id: str, body: TextIn, request: Request):
        guard(service.store.current_version, paper_id)   # 试卷不存在时直接 404，不进入模型调用
        return StreamingResponse(_sse(service.edit(paper_id, body.text, _client(request))),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/papers")
    def list_papers():
        return service.list()

    @app.get("/api/papers/{paper_id}")
    def get_paper(paper_id: str, version: int | None = None):
        return guard(service.get, paper_id, version)

    @app.post("/api/papers/{paper_id}/confirm")
    def confirm(paper_id: str, body: ConfirmIn):
        return guard(service.confirm, paper_id, body.positions, body.note)

    @app.post("/api/papers/{paper_id}/undo")
    def undo(paper_id: str):
        return guard(service.undo, paper_id)

    @app.delete("/api/papers/{paper_id}")
    def delete_paper(paper_id: str):
        return guard(service.delete, paper_id)

    @app.delete("/api/papers")
    def clear_all():
        return service.clear_all()

    @app.get("/api/papers/{paper_id}/pdf/{kind}")
    def pdf(paper_id: str, kind: str, version: int | None = None):
        if kind not in ("student", "answers"):
            raise HTTPException(400, "kind 只能是 student 或 answers")
        student, answers = guard(service.export, paper_id, version)
        path = student if kind == "student" else answers
        return FileResponse(path, media_type="application/pdf", filename=path.name)

    return app


def build_default_app() -> FastAPI:
    """读取 .env 配置，加载模型客户端与课程知识库。"""
    from ..config import load_settings
    from ..grounding import MultiGrounding
    from ..llm import make_llm
    from ..store import PaperStore

    settings = load_settings()
    try:
        grounding = MultiGrounding.from_env()
    except Exception:
        grounding = None
    store = PaperStore(settings.db_path)
    return create_app(ExamService(store, lambda: make_llm(settings), grounding))
