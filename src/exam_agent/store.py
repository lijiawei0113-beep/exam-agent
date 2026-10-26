"""SQLite 存储：试卷版本快照 + 运行轨迹。

- 每次修改都写入一个新的不可变版本（paper_versions），可回滚、可对比。
- commit 时做乐观并发检查：expected_version 不是当前版本就拒绝，防止两次修改互相覆盖。
- trace_events 记录每一步 Agent 调用，是后续可观测性和评测的数据来源。
"""
from __future__ import annotations

import functools
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import PaperSnapshot

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    current_version INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_versions (
    paper_id       TEXT NOT NULL,
    version        INTEGER NOT NULL,
    parent_version INTEGER,
    operation      TEXT NOT NULL,
    summary        TEXT NOT NULL,
    snapshot_json  TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (paper_id, version)
);
CREATE TABLE IF NOT EXISTS chat_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id   TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    version    INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trace_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    paper_id   TEXT,
    step       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class VersionConflict(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_paper_id() -> str:
    return "p_" + uuid.uuid4().hex[:8]


def _locked(fn):
    """网页服务里多个请求在不同线程中运行：同一个连接上的读写串行执行，避免事务交错。"""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


class PaperStore:
    def __init__(self, db_path: str | Path = ":memory:"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------- 版本 ----------
    @_locked
    def create_paper(self, snap: PaperSnapshot, summary: str) -> int:
        with self.conn:
            self.conn.execute(
                "INSERT INTO papers(paper_id, title, current_version, created_at) VALUES (?,?,?,?)",
                (snap.paper_id, snap.title, 1, _now()),
            )
            self._insert_version(snap, 1, None, "create", summary)
        return 1

    @_locked
    def commit_version(self, snap: PaperSnapshot, expected_version: int, operation: str, summary: str,
                       parent_version: int | None = -1) -> int:
        """parent_version 表示“内容上的父版本”，默认就是 expected_version；
        撤销时它指向被恢复版本的父版本，这样连续撤销能一步步往回退。"""
        with self.conn:  # 单事务：读当前版本 + 写新版本
            row = self.conn.execute(
                "SELECT current_version FROM papers WHERE paper_id=?", (snap.paper_id,)
            ).fetchone()
            if row is None:
                raise KeyError(snap.paper_id)
            if row["current_version"] != expected_version:
                raise VersionConflict(
                    f"试卷已被更新到 v{row['current_version']}，本次修改基于 v{expected_version}"
                )
            new_v = expected_version + 1
            parent = expected_version if parent_version == -1 else parent_version
            self._insert_version(snap, new_v, parent, operation, summary)
            self.conn.execute(
                "UPDATE papers SET current_version=?, title=? WHERE paper_id=?",
                (new_v, snap.title, snap.paper_id),
            )
        return new_v

    def _insert_version(self, snap, version, parent, operation, summary):
        self.conn.execute(
            "INSERT INTO paper_versions VALUES (?,?,?,?,?,?,?)",
            (snap.paper_id, version, parent, operation, summary,
             snap.model_dump_json(), _now()),
        )

    @_locked
    def current_version(self, paper_id: str) -> int:
        row = self.conn.execute("SELECT current_version FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        if row is None:
            raise KeyError(paper_id)
        return row["current_version"]

    @_locked
    def load(self, paper_id: str, version: int | None = None) -> tuple[PaperSnapshot, int]:
        v = version or self.current_version(paper_id)
        row = self.conn.execute(
            "SELECT snapshot_json FROM paper_versions WHERE paper_id=? AND version=?", (paper_id, v)
        ).fetchone()
        if row is None:
            raise KeyError(f"{paper_id} v{v}")
        return PaperSnapshot.model_validate_json(row["snapshot_json"]), v

    @_locked
    def history(self, paper_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT version, parent_version, operation, summary, created_at FROM paper_versions "
            "WHERE paper_id=? ORDER BY version", (paper_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    @_locked
    def list_papers(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT paper_id, title, current_version, created_at FROM papers ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 删除 ----------
    _TABLES = ("paper_versions", "chat_turns", "trace_events", "papers")

    @_locked
    def delete_paper(self, paper_id: str) -> None:
        """删除一份试卷的全部版本、对话和运行轨迹。不存在时抛出 KeyError。"""
        if self.conn.execute("SELECT 1 FROM papers WHERE paper_id=?", (paper_id,)).fetchone() is None:
            raise KeyError(paper_id)
        with self.conn:
            for t in self._TABLES:
                self.conn.execute(f"DELETE FROM {t} WHERE paper_id=?", (paper_id,))

    @_locked
    def clear_all(self) -> int:
        """清空全部试卷，返回删除的试卷数。"""
        n = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        with self.conn:
            for t in self._TABLES:
                self.conn.execute(f"DELETE FROM {t}")
        return n

    # ---------- 对话 ----------
    @_locked
    def add_turn(self, paper_id: str, role: str, content: str, version: int | None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_turns(paper_id, role, content, version, created_at) VALUES (?,?,?,?,?)",
                (paper_id, role, content, version, _now()),
            )

    @_locked
    def recent_turns(self, paper_id: str, limit: int = 6) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT role, content, version FROM chat_turns WHERE paper_id=? ORDER BY id DESC LIMIT ?",
            (paper_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ---------- 轨迹 ----------
    @_locked
    def log_event(self, run_id: str, paper_id: str | None, step: str, payload: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO trace_events(run_id, paper_id, step, payload, created_at) VALUES (?,?,?,?,?)",
                (run_id, paper_id, step, json.dumps(payload, ensure_ascii=False), _now()),
            )

    @_locked
    def events(self, paper_id: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT run_id, paper_id, step, payload, created_at FROM trace_events WHERE 1=1", []
        if paper_id:
            q += " AND paper_id=?"; args.append(paper_id)
        if run_id:
            q += " AND run_id=?"; args.append(run_id)
        rows = self.conn.execute(q + " ORDER BY id", args).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]
