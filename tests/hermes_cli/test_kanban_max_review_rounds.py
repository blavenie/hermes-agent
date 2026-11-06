"""Review/rework round cap (``kanban.max_review_rounds``).

Regression coverage for #149149: a card could cycle ``review_requested`` ->
``changes_requested`` forever because none of the existing stop conditions
counted clean review rounds. The cap is enforced in ``request_changes()`` so
the CLI and the dashboard route share one choke point.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB and the shipped default cap."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # A worker's pinned board env must not leak into the temp home's DB.
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_BOARD",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(var, raising=False)
    # Config default 0 = disabled, matching config_defaults.py.
    monkeypatch.setattr(kb, "_kanban_max_review_rounds_default", lambda: 0)
    kb.init_db()
    return home


def _events(conn, tid, kind: str | None = None):
    rows = conn.execute(
        "SELECT kind, payload, run_id FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None, r["run_id"])
        for r in rows
    ]
    return [e for e in out if kind is None or e[0] == kind]


def _task(conn, tid):
    return kb.get_task(conn, tid)


def _review_round(conn, tid, *, reason: str = "fix it"):
    """worker -> review -> changes_requested; returns ``request_changes``' result."""
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    assert kb.request_review(
        conn, tid, summary="v", reviewer="reviewer",
        expected_run_id=claimed.current_run_id,
    )
    review = kb.claim_review_task(conn, tid)
    assert review is not None
    return kb.request_changes(
        conn, tid, reason=reason, expected_run_id=review.current_run_id,
    )


def test_default_zero_leaves_rework_unchanged(kanban_home: Path) -> None:
    """Cap disabled (the default): five rounds stay normal, no cap event."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cycle", assignee="builder")
        for _ in range(5):
            ok, detail = _review_round(conn, tid)
            assert ok is True
            assert detail == "builder"
            assert _task(conn, tid).status in ("ready", "todo")

        assert _events(conn, tid, kind="rework_cap_reached") == []
        assert _events(conn, tid, kind="blocked") == []
        assert _events(conn, tid, kind="rework_reset") == []


def test_cap_blocks_on_nth_round_with_one_event(kanban_home: Path) -> None:
    """N=3: the third ``changes_requested`` parks the card blocked."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="capped", assignee="builder", max_review_rounds=3,
        )
        for _ in range(2):
            ok, detail = _review_round(conn, tid)
            assert ok is True
            assert detail == "builder"

        ok, detail = _review_round(conn, tid)
        assert ok is True
        expected_reason = "rework_round=3/3 (reviewer: reviewer)"
        assert detail == f"blocked — {expected_reason}"

        task = _task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"

        cap_events = _events(conn, tid, kind="rework_cap_reached")
        assert len(cap_events) == 1
        payload = cap_events[0][1]
        assert payload["round"] == 3
        assert payload["max"] == 3
        assert payload["reviewer"] == "reviewer"
        assert payload["reason"] == expected_reason
        assert len(payload["run_ids"]) == 3

        blocked = _events(conn, tid, kind="blocked")
        assert len(blocked) == 1
        assert blocked[0][1]["reason"] == expected_reason
        assert blocked[0][1]["kind"] == "needs_input"


def test_plain_unblock_does_not_grant_a_round(kanban_home: Path) -> None:
    """A plain unblock leaves the cap in force; the next verdict blocks again."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="no reset", assignee="builder", max_review_rounds=2,
        )
        for _ in range(2):
            _review_round(conn, tid)
        assert _task(conn, tid).status == "blocked"

        assert kb.unblock_task(conn, tid) is True
        assert _task(conn, tid).status == "ready"
        assert _events(conn, tid, kind="rework_reset") == []

        ok, detail = _review_round(conn, tid)
        assert ok is True
        assert detail == "blocked — rework_round=3/2 (reviewer: reviewer)"
        assert _task(conn, tid).status == "blocked"


def test_unblock_reset_rework_grants_a_new_budget(kanban_home: Path) -> None:
    """``--reset-rework`` records a baseline and allows N more rounds."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="reset", assignee="builder", max_review_rounds=2,
        )
        for _ in range(2):
            _review_round(conn, tid)
        assert _task(conn, tid).status == "blocked"

        assert kb.unblock_task(conn, tid, reset_rework=True) is True
        resets = _events(conn, tid, kind="rework_reset")
        assert len(resets) == 1
        assert resets[0][1]["rounds_before"] == 2

        ok, detail = _review_round(conn, tid)
        assert ok is True
        assert detail == "builder"
        assert _task(conn, tid).status in ("ready", "todo")


def test_create_override_wins_over_config(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-card ``--max-review-rounds`` beats the config default."""
    monkeypatch.setattr(kb, "_kanban_max_review_rounds_default", lambda: 2)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="override", assignee="builder", max_review_rounds=4,
        )
        assert _task(conn, tid).max_review_rounds == 4

        for _ in range(2):
            ok, _ = _review_round(conn, tid)
            assert ok is True
        # Two rounds would trip the config's 2-round cap, but not this card's 4.
        assert _task(conn, tid).status in ("ready", "todo")
        assert _events(conn, tid, kind="rework_cap_reached") == []


def test_show_displays_effective_cap(
    kanban_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``hermes kanban show`` prints the effective per-card cap."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="shown", assignee="builder", max_review_rounds=4,
        )
    args = SimpleNamespace(task_id=tid, json=False, state_type=None, state_name=None)
    assert kcli._cmd_show(args) == 0
    assert "max-review-rounds: 4 (task)" in capsys.readouterr().out


def test_show_displays_disabled_cap(
    kanban_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without an override the default cap reads as disabled."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="shown default", assignee="builder")
    args = SimpleNamespace(task_id=tid, json=False, state_type=None, state_name=None)
    assert kcli._cmd_show(args) == 0
    assert "max-review-rounds: 0 (disabled)" in capsys.readouterr().out
