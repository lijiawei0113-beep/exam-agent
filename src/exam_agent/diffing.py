"""版本差异：告诉老师两个版本之间到底改了什么。以 slot_id 对齐，而不是卷面题号。"""
from __future__ import annotations

from pydantic import BaseModel

from .domain import PaperSnapshot


class SlotChange(BaseModel):
    slot_id: str
    kind: str  # request / added / removed / content / score / position
    old_position: int | None = None
    new_position: int | None = None
    detail: str = ""


def diff_snapshots(old: PaperSnapshot, new: PaperSnapshot) -> list[SlotChange]:
    old_slots = {s.slot_id: s for s in old.blueprint.slots}
    new_slots = {s.slot_id: s for s in new.blueprint.slots}
    old_q = {q.slot_id: q for q in old.questions}
    new_q = {q.slot_id: q for q in new.questions}
    changes: list[SlotChange] = []

    for sid, ns in new_slots.items():
        os_ = old_slots.get(sid)
        if os_ is None:
            changes.append(SlotChange(slot_id=sid, kind="added", new_position=ns.position,
                                      detail=f"新增{ns.qtype.value}（{ns.score}分）：{new_q[sid].stem[:30]}"))
            continue
        if old_q[sid].content_fingerprint() != new_q[sid].content_fingerprint():
            o, n = old_q[sid], new_q[sid]
            attrs = []
            if o.qtype != n.qtype:
                attrs.append(f"题型 {o.qtype.value}→{n.qtype.value}")
            if o.difficulty != n.difficulty:
                attrs.append(f"难度 {o.difficulty.value}→{n.difficulty.value}")
            if o.knowledge_point != n.knowledge_point:
                attrs.append(f"知识点 {o.knowledge_point}→{n.knowledge_point}")
            if os_.position != ns.position:
                attrs.append(f"题号 {os_.position}→{ns.position}")
            attrs.append(f"新题：{n.stem[:30]}")
            changes.append(SlotChange(slot_id=sid, kind="content", old_position=os_.position,
                                      new_position=ns.position, detail="；".join(attrs)))
        if os_.score != ns.score:
            changes.append(SlotChange(slot_id=sid, kind="score", old_position=os_.position,
                                      new_position=ns.position, detail=f"{os_.score}分→{ns.score}分"))
        content_changed = old_q[sid].content_fingerprint() != new_q[sid].content_fingerprint()
        if os_.position != ns.position and not content_changed:
            changes.append(SlotChange(slot_id=sid, kind="position", old_position=os_.position,
                                      new_position=ns.position, detail=f"第{os_.position}题→第{ns.position}题"))

    for sid, os_ in old_slots.items():
        if sid not in new_slots:
            changes.append(SlotChange(slot_id=sid, kind="removed", old_position=os_.position,
                                      detail=f"删除{os_.qtype.value}：{old_q[sid].stem[:30]}"))
    oc = {c.text for c in old.request.constraints}
    nc = {c.text for c in new.request.constraints}
    for t in sorted(nc - oc):
        changes.append(SlotChange(slot_id="-", kind="request", detail=f"新增约束：{t}"))
    for t in sorted(oc - nc):
        changes.append(SlotChange(slot_id="-", kind="request", detail=f"移除约束：{t}"))
    if old.request.total_score != new.request.total_score:
        changes.append(SlotChange(slot_id="-", kind="request",
                                  detail=f"满分 {old.request.total_score}→{new.request.total_score}"))
    if old.request.question_count != new.request.question_count:
        changes.append(SlotChange(slot_id="-", kind="request",
                                  detail=f"题数 {old.request.question_count}→{new.request.question_count}"))
    order = {"request": -1, "removed": 0, "content": 1, "added": 2, "score": 3, "position": 4}
    changes.sort(key=lambda c: (order[c.kind], c.new_position or c.old_position or 0))
    return changes
