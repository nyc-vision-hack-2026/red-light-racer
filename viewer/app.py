#!/usr/bin/env python3
"""Tiny side app to review find_candidates.py windows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
VIEWER_STATIC = Path(__file__).resolve().parent / "static"
CANDIDATES_PATH = Path(os.environ.get("CANDIDATES_PATH", ROOT / "candidates.json"))

app = FastAPI(title="Candidate Viewer", docs_url=None, redoc_url=None)


def _load() -> dict:
    if not CANDIDATES_PATH.is_file():
        raise HTTPException(status_code=404, detail=f"missing {CANDIDATES_PATH}")
    return json.loads(CANDIDATES_PATH.read_text())


def _safe_file(rel: str) -> Path:
    """Resolve a candidate frame path and keep it inside the repo root."""
    raw = Path(rel)
    path = (ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path outside repo") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="frame not found")
    return path


@app.get("/api/candidates")
def list_candidates() -> dict:
    data = _load()
    out = []
    for i, c in enumerate(data.get("candidates", [])):
        out.append({
            "id": i,
            "green_index_global": c["green_index_global"],
            "red_start_global": c["red_start_global"],
            "red_len_frames": c["red_len_frames"],
            "motion_at_green": c.get("motion_at_green"),
            "occupancy_at_green": c.get("occupancy_at_green"),
            "prompt_frames": [f"/api/frame?path={quote(p, safe='')}" for p in c["prompt_frames"]],
            "reveal_frames": [f"/api/frame?path={quote(p, safe='')}" for p in c["reveal_frames"]],
        })
    return {
        "source": str(CANDIDATES_PATH.relative_to(ROOT) if CANDIDATES_PATH.is_relative_to(ROOT) else CANDIDATES_PATH),
        "thresholds": data.get("thresholds", {}),
        "candidates": out,
    }


@app.get("/api/frame")
def frame(path: str) -> FileResponse:
    file_path = _safe_file(path)
    return FileResponse(file_path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(VIEWER_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(VIEWER_STATIC)), name="static")
