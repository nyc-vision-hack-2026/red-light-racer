"""FastAPI app — API routes + static frontend. No CV inference."""

from __future__ import annotations

import os
import re
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.source import StaticRoundSource, resolution_to_dict
from app.store import ScoreStore, build_score_store

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"


def resolve_round_set(data_dir: Path, name: str) -> tuple[Path, Path]:
    """Return the rounds document and frame root for a safe named set."""
    clean_name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", clean_name):
        raise ValueError("ROUND_SET must contain only letters, numbers, '-' or '_'")
    root = data_dir if clean_name == "classic" else data_dir / "round_sets" / clean_name
    return root / "rounds.json", root / "frames"


ROUND_SET = os.environ.get("ROUND_SET", "classic").strip()
if os.environ.get("ROUNDS_PATH"):
    ROUNDS_PATH = Path(os.environ["ROUNDS_PATH"])
    FRAMES_DIR = Path(os.environ.get("FRAMES_PATH", DATA_DIR / "frames"))
else:
    ROUNDS_PATH, FRAMES_DIR = resolve_round_set(DATA_DIR, ROUND_SET)
FRAME_NAMESPACE = os.environ.get("FRAME_NAMESPACE", ROUND_SET).strip()
if not re.fullmatch(r"[A-Za-z0-9_-]+", FRAME_NAMESPACE):
    raise ValueError("FRAME_NAMESPACE must contain only letters, numbers, '-' or '_'")

_source: StaticRoundSource | None = None
_store: ScoreStore | None = None


def get_source() -> StaticRoundSource:
    global _source
    if _source is None:
        drip = os.environ.get("DRIP_REVEAL", "0") == "1"
        _source = StaticRoundSource(
            ROUNDS_PATH,
            frame_url_prefix=f"/frames/{FRAME_NAMESPACE}/",
            drip_reveal=drip,
            drip_retry_after_ms=int(os.environ.get("DRIP_RETRY_MS", "6000")),
            pending_retry_after_ms=int(os.environ.get("PENDING_RETRY_MS", "0")),
        )
    return _source


def get_store() -> ScoreStore:
    global _store
    if _store is None:
        _store = build_score_store()
    return _store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_source()
    get_store()
    yield


app = FastAPI(title="Red Light Racer", docs_url=None, redoc_url=None, lifespan=lifespan)

# --- models ---


class GuessBody(BaseModel):
    session_id: str
    track_id: int | None = None
    elapsed_ms: int | float = 5000
    streak: int = 0


class ScoreBody(BaseModel):
    initials: str
    score: int = Field(..., ge=0, le=100000)


# --- API ---


@app.get("/healthz")
@app.get("/health")
def healthz() -> dict:
    return {"ok": True, "round_set": ROUND_SET, "frame_namespace": FRAME_NAMESPACE}


@app.post("/api/session")
def create_session() -> dict:
    sid = get_source().create_session(target_rounds=10)
    return {"session_id": sid, "target_rounds": 10}


@app.get("/api/session/{session_id}/next-round")
def next_round(session_id: str) -> dict:
    try:
        offer = get_source().next_round(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session") from None

    if offer.status == "ready":
        return {"status": "ready", "round": offer.round}
    if offer.status == "waiting":
        return {
            "status": "waiting",
            "retry_after_ms": offer.retry_after_ms or 2000,
        }
    return {"status": "session_complete"}


@app.get("/api/round/{round_id}")
def get_round(round_id: str) -> dict:
    prompt = get_source().get_prompt(round_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="unknown round")
    # Hard guard: answer must never leak in the prompt payload.
    assert "winner_track_id" not in prompt
    assert "finish_frame_index" not in prompt
    return prompt


@app.post("/api/round/{round_id}/guess")
def submit_guess(round_id: str, body: GuessBody) -> dict:
    try:
        return get_source().submit_guess(
            body.session_id,
            round_id,
            body.track_id,
            body.elapsed_ms,
            body.streak,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from None


@app.get("/api/round/{round_id}/resolution")
def get_resolution(round_id: str, session_id: str = Query(...)) -> dict:
    try:
        res = get_source().resolution(session_id, round_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from None
    return resolution_to_dict(res)


@app.post("/api/score")
def post_score(body: ScoreBody) -> dict:
    initials = body.initials.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", initials):
        raise HTTPException(status_code=400, detail="initials must be 3 letters A-Z")
    rank = get_store().submit(initials, int(body.score))
    return {"rank": rank}


@app.get("/api/leaderboard")
def leaderboard(limit: int = Query(20)) -> dict:
    limit = max(1, min(50, limit))
    return {"entries": get_store().leaderboard(limit)}


# --- static assets ---


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class CachedStaticFiles(StaticFiles):
    def __init__(self, *args, cache_control: str = "public, max-age=86400", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_control = cache_control

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = self.cache_control
        return resp


app.mount(
    f"/frames/{FRAME_NAMESPACE}",
    CachedStaticFiles(directory=str(FRAMES_DIR), cache_control="public, max-age=604800, immutable"),
    name="frames",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
