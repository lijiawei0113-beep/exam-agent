import pytest

from exam_agent.store import PaperStore, VersionConflict

from test_invariants import make_paper


def test_versioning_and_conflict():
    store = PaperStore(":memory:")
    paper = make_paper()
    assert store.create_paper(paper, "新建") == 1

    edited = paper.model_copy(deep=True)
    edited.questions[0].stem = "改过的题干"
    assert store.commit_version(edited, expected_version=1, operation="edit", summary="改第1题") == 2

    # 基于过期版本提交，必须被拒绝
    with pytest.raises(VersionConflict):
        store.commit_version(edited, expected_version=1, operation="edit", summary="重复提交")

    latest, v = store.load("p_test")
    original, _ = store.load("p_test", version=1)
    assert v == 2 and latest.questions[0].stem == "改过的题干"
    assert original.questions[0].stem != "改过的题干"
    assert [h["version"] for h in store.history("p_test")] == [1, 2]
