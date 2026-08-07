#!/usr/bin/env python3
"""Tiny side app to review find_candidates.py windows across datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
VIEWER_STATIC = Path(__file__).resolve().parent / "static"
CANDIDATES_DIR = Path(os.environ.get("CANDIDATES_DIR", ROOT / "candidates"))

app = FastAPI(title="Candidate Viewer", docs_url=None, redoc_url=None)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _discover() -> list[dict]:
    """Find candidate JSON files under candidates/ (or CANDIDATES_DIR)."""
    found: list[Path] = []
    if CANDIDATES_DIR.is_dir():
        found.extend(sorted(CANDIDATES_DIR.glob("*.json")))
    if not found:
        # legacy single-file dumps at repo root
        for p in sorted(ROOT.glob("candidates*.json")):
            found.append(p)

    out = []
    for path in found:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        n = len(data.get("candidates") or [])
        out.append({
            "id": path.stem,
            "path": _rel(path),
            "count": n,
            "thresholds": data.get("thresholds") or {},
        })
    return out


def _resolve_dataset(dataset_id: str | None) -> Path:
    datasets = _discover()
    if not datasets:
        raise HTTPException(404, f"no candidate JSON under {_rel(CANDIDATES_DIR)}")
    if dataset_id:
        for d in datasets:
            if d["id"] == dataset_id:
                return ROOT / d["path"]
        raise HTTPException(404, f"unknown dataset {dataset_id}")
    return ROOT / datasets[0]["path"]


def _safe_file(rel: str) -> Path:
    raw = Path(rel)
    path = (ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path outside repo") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="frame not found")
    return path


def _serialize_candidates(data: dict) -> list[dict]:
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
    return out


@app.get("/api/datasets")
def list_datasets() -> dict:
    return {"datasets": _discover()}


@app.get("/api/candidates")
def list_candidates(dataset: str | None = Query(None)) -> dict:
    path = _resolve_dataset(dataset)
    data = json.loads(path.read_text())
    return {
        "dataset": path.stem,
        "source": _rel(path),
        "thresholds": data.get("thresholds", {}),
        "candidates": _serialize_candidates(data),
    }


@app.get("/api/frame")
def frame(path: str) -> FileResponse:
    file_path = _safe_file(path)
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(VIEWER_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(VIEWER_STATIC)), name="static")
