"""Unit tests for scoring and API prompt secrecy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.scoring import clamp_elapsed_ms, compute_points
from app.main import resolve_round_set


ROOT = Path(__file__).resolve().parent.parent


def test_resolve_round_set_keeps_classic_and_generated_data_separate():
    data_dir = Path("C:/round-data")
    classic_rounds, classic_frames = resolve_round_set(data_dir, "classic")
    generated_rounds, generated_frames = resolve_round_set(data_dir, "roboflow")

    assert classic_rounds == data_dir / "rounds.json"
    assert classic_frames == data_dir / "frames"
    assert generated_rounds == data_dir / "round_sets" / "roboflow" / "rounds.json"
    assert generated_frames == data_dir / "round_sets" / "roboflow" / "frames"


def test_resolve_round_set_rejects_path_traversal():
    with pytest.raises(ValueError, match="ROUND_SET"):
        resolve_round_set(Path("C:/round-data"), "../outside")


def test_clamp_elapsed():
    assert clamp_elapsed_ms(0) == 150
    assert clamp_elapsed_ms(149) == 150
    assert clamp_elapsed_ms(150) == 150
    assert clamp_elapsed_ms(5000) == 5000
    assert clamp_elapsed_ms(30000) == 30000
    assert clamp_elapsed_ms(99999) == 30000


def test_scoring_correct_fast():
    # streak 0, elapsed 0 → clamped to 150 → nearly full speed bonus
    pts = compute_points(correct=True, elapsed_ms=150, streak=0)
    assert pts == round((100 + round(50 * max(0, 1 - 150 / 10000))) * 1.0)


def test_scoring_wrong_is_zero():
    assert compute_points(correct=False, elapsed_ms=200, streak=5) == 0


def test_scoring_streak_cap():
    # multiplier caps at 3.0 → streak 8 would be 1+2=3
    pts = compute_points(correct=True, elapsed_ms=10000, streak=8)
    assert pts == round((100 + 0) * 3.0)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Ensure stub data exists
    rounds = ROOT / "data" / "rounds.json"
    if not rounds.exists():
        pytest.skip("run tools/make_stub_rounds.py first")

    import os

    os.environ["FORCE_MEMORY_STORE"] = "1"
    # Fresh app import with memory store
    from app import main as mainmod

    mainmod._source = None
    mainmod._store = None
    with TestClient(mainmod.app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_prompt_hides_answer(client):
    """GET /api/round/{id} must not contain winner_track_id or finish_frame_index."""
    sess = client.post("/api/session").json()
    nxt = client.get(f"/api/session/{sess['session_id']}/next-round").json()
    assert nxt["status"] == "ready"
    rid = nxt["round"]["id"]

    body = client.get(f"/api/round/{rid}").json()
    raw = json.dumps(body)
    assert "winner_track_id" not in body
    assert "finish_frame_index" not in body
    assert "winner_track_id" not in raw
    assert "finish_frame_index" not in raw
    assert body["id"] == rid
    assert "frames" in body and len(body["frames"]) == body["green_index"] + 1
    assert all(frame.startswith("/frames/classic/") for frame in body["frames"])


def test_resolution_pending_then_resolved(client):
    sess = client.post("/api/session").json()
    sid = sess["session_id"]
    nxt = client.get(f"/api/session/{sid}/next-round").json()
    rid = nxt["round"]["id"]
    track = nxt["round"]["candidates"][0]["track_id"]

    g = client.post(
        f"/api/round/{rid}/guess",
        json={"session_id": sid, "track_id": track, "elapsed_ms": 2000, "streak": 0},
    ).json()
    assert g["status"] == "pending"

    first = client.get(f"/api/round/{rid}/resolution", params={"session_id": sid}).json()
    assert first["status"] == "pending"

    second = client.get(f"/api/round/{rid}/resolution", params={"session_id": sid}).json()
    assert second["status"] == "resolved"
    assert "winner_track_id" in second
    assert "reveal" in second
    assert second["reveal"]["frames_complete"] is True


def test_main_does_not_import_rounds_json_directly():
    """Grep-style: main.py must not load rounds.json or touch the filesystem for rounds."""
    text = (ROOT / "app" / "main.py").read_text()
    assert "rounds.json" not in text or "ROUNDS_PATH" in text
    # Handlers must go through RoundSource — no json.load of rounds in main
    assert "json.load" not in text
    assert "json.loads" not in text
