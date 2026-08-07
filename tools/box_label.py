#!/usr/bin/env python3
"""
Interactive vehicle bounding-box labeler for candidate windows.

Draw boxes on cars across prompt/reveal frames, assign track IDs, then export
detections JSON compatible with tools/03_track.py / tools/04_cut_rounds.py.

Usage
-----
    # label candidates from find_candidates.py
    python tools/box_label.py --candidates candidates.json

    # or label a raw capture folder (every frame)
    python tools/box_label.py --frames captures/park_ave_23st_20260807_181047

    # open http://127.0.0.1:8766/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "box_label_static"
DEFAULT_OUT = ROOT / "data" / "tracks" / "labels.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path):
    name = path.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_frame_paths(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=natural_key)


def rel_to_root(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


class BoxLabelApp:
    def __init__(self, candidates_path: Path | None, frames_dir: Path | None, out_path: Path):
        self.out_path = out_path
        self.candidates_path = candidates_path
        self.session_dir: Path | None = None
        self.session_frames: list[Path] = []
        self.path_to_index: dict[str, int] = {}
        self.windows: list[dict] = []
        self._build(candidates_path, frames_dir)
        self.state = self._load_or_init()

    def _build(self, candidates_path: Path | None, frames_dir: Path | None) -> None:
        if candidates_path:
            data = json.loads(candidates_path.read_text())
            cands = data.get("candidates") or []
            if not cands:
                raise SystemExit(f"no candidates in {candidates_path}")
            sample = Path(cands[0]["prompt_frames"][0])
            if not sample.is_absolute():
                sample = ROOT / sample
            self.session_dir = sample.parent
            self.session_frames = list_frame_paths(self.session_dir)
            self.path_to_index = {rel_to_root(p): i for i, p in enumerate(self.session_frames)}
            # also index by basename for resilience
            by_name = {p.name: i for i, p in enumerate(self.session_frames)}

            for i, c in enumerate(cands):
                paths = []
                seen = set()
                for p in c["prompt_frames"] + c["reveal_frames"]:
                    if p in seen:
                        continue
                    seen.add(p)
                    paths.append(p)
                indices = []
                for p in paths:
                    rel = rel_to_root(Path(p) if Path(p).is_absolute() else ROOT / p)
                    if rel in self.path_to_index:
                        indices.append(self.path_to_index[rel])
                    elif Path(p).name in by_name:
                        indices.append(by_name[Path(p).name])
                    else:
                        raise SystemExit(f"frame not in session: {p}")
                self.windows.append({
                    "id": i,
                    "green_index_global": c["green_index_global"],
                    "red_start_global": c["red_start_global"],
                    "red_len_frames": c["red_len_frames"],
                    "frame_indices": indices,
                    "frame_paths": [rel_to_root(self.session_frames[j]) for j in indices],
                })
        elif frames_dir:
            self.session_dir = frames_dir
            self.session_frames = list_frame_paths(frames_dir)
            if not self.session_frames:
                raise SystemExit(f"no frames in {frames_dir}")
            self.path_to_index = {rel_to_root(p): i for i, p in enumerate(self.session_frames)}
            indices = list(range(len(self.session_frames)))
            self.windows.append({
                "id": 0,
                "green_index_global": None,
                "red_start_global": None,
                "red_len_frames": None,
                "frame_indices": indices,
                "frame_paths": [rel_to_root(p) for p in self.session_frames],
            })
        else:
            raise SystemExit("pass --candidates or --frames")

    def _load_or_init(self) -> dict:
        if self.out_path.is_file():
            return json.loads(self.out_path.read_text())
        return {
            "version": 1,
            "session": self.session_dir.name if self.session_dir else "",
            "session_dir": rel_to_root(self.session_dir) if self.session_dir else "",
            "source_candidates": rel_to_root(self.candidates_path) if self.candidates_path else None,
            "tracks": {},  # tid -> {label, keyframes: {frame_idx: {x,y,w,h}}}
            "next_track_id": 1,
        }

    def save(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.state, indent=2) + "\n")

    def export_tracks(self, dest: Path) -> dict:
        """Expand keyframes → per-frame detections (linear interpolate gaps)."""
        detections: list[dict] = []
        for tid_str, track in self.state.get("tracks", {}).items():
            tid = int(tid_str)
            kfs = {int(k): v for k, v in (track.get("keyframes") or {}).items()}
            if not kfs:
                continue
            frames = sorted(kfs)
            # fill inclusive range with interpolation between keyframes
            for a, b in zip(frames, frames[1:]):
                ka, kb = kfs[a], kfs[b]
                span = b - a
                for f in range(a, b):
                    t = (f - a) / span if span else 0.0
                    detections.append({
                        "frame": f,
                        "track_id": tid,
                        "x": ka["x"] + t * (kb["x"] - ka["x"]),
                        "y": ka["y"] + t * (kb["y"] - ka["y"]),
                        "w": ka["w"] + t * (kb["w"] - ka["w"]),
                        "h": ka["h"] + t * (kb["h"] - ka["h"]),
                        "cls": "vehicle",
                        "conf": 1.0,
                        "keyframe": f in kfs,
                    })
            last = frames[-1]
            kb = kfs[last]
            detections.append({
                "frame": last,
                "track_id": tid,
                "x": kb["x"], "y": kb["y"], "w": kb["w"], "h": kb["h"],
                "cls": "vehicle",
                "conf": 1.0,
                "keyframe": True,
            })

        detections.sort(key=lambda d: (d["frame"], d["track_id"]))
        payload = {
            "session": self.session_dir.name if self.session_dir else "",
            "frame_count": len(self.session_frames),
            "frames": [p.name for p in self.session_frames],
            "detections": detections,
            "labels_source": rel_to_root(self.out_path),
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n")
        return payload


def create_app(core: BoxLabelApp) -> FastAPI:
    app = FastAPI(title="Box Labeler", docs_url=None, redoc_url=None)

    class KeyframeBody(BaseModel):
        track_id: int
        frame: int
        x: float
        y: float
        w: float
        h: float
        label: str | None = None

    class TrackBody(BaseModel):
        label: str | None = None

    class DeleteBody(BaseModel):
        track_id: int
        frame: int | None = None  # None = delete whole track

    def safe_file(rel: str) -> Path:
        raw = Path(unquote(rel))
        path = (ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise HTTPException(400, "path outside repo") from exc
        if not path.is_file():
            raise HTTPException(404, "frame not found")
        return path

    @app.get("/api/meta")
    def meta():
        return {
            "session": core.state.get("session"),
            "session_dir": core.state.get("session_dir"),
            "out_path": rel_to_root(core.out_path),
            "frame_count": len(core.session_frames),
            "windows": [
                {
                    **w,
                    "frame_urls": [f"/api/frame?path={quote(p, safe='')}" for p in w["frame_paths"]],
                }
                for w in core.windows
            ],
            "tracks": core.state.get("tracks", {}),
            "next_track_id": core.state.get("next_track_id", 1),
        }

    @app.get("/api/frame")
    def frame(path: str):
        return FileResponse(
            safe_file(path),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/api/track")
    def new_track(body: TrackBody):
        tid = int(core.state.get("next_track_id", 1))
        label = body.label or chr(ord("A") + (tid - 1) % 26)
        core.state.setdefault("tracks", {})[str(tid)] = {"label": label, "keyframes": {}}
        core.state["next_track_id"] = tid + 1
        core.save()
        return {"track_id": tid, "label": label, "tracks": core.state["tracks"]}

    @app.post("/api/keyframe")
    def upsert_keyframe(body: KeyframeBody):
        tracks = core.state.setdefault("tracks", {})
        tid = str(body.track_id)
        if tid not in tracks:
            tracks[tid] = {
                "label": body.label or chr(ord("A") + (body.track_id - 1) % 26),
                "keyframes": {},
            }
        if body.w <= 1 or body.h <= 1:
            raise HTTPException(400, "box too small")
        tracks[tid]["keyframes"][str(body.frame)] = {
            "x": round(body.x, 2),
            "y": round(body.y, 2),
            "w": round(body.w, 2),
            "h": round(body.h, 2),
        }
        core.save()
        return {"ok": True, "tracks": tracks}

    @app.post("/api/delete")
    def delete_box(body: DeleteBody):
        tracks = core.state.setdefault("tracks", {})
        tid = str(body.track_id)
        if tid not in tracks:
            raise HTTPException(404, "unknown track")
        if body.frame is None:
            del tracks[tid]
        else:
            tracks[tid]["keyframes"].pop(str(body.frame), None)
        core.save()
        return {"ok": True, "tracks": tracks}

    @app.post("/api/save")
    def save():
        core.save()
        return {"ok": True, "path": rel_to_root(core.out_path)}

    @app.post("/api/export")
    def export(dest: str | None = None):
        session = core.session_dir.name if core.session_dir else "session"
        path = Path(dest) if dest else ROOT / "data" / "tracks" / f"{session}.json"
        if not path.is_absolute():
            path = ROOT / path
        payload = core.export_tracks(path)
        return {
            "ok": True,
            "path": rel_to_root(path),
            "detections": len(payload["detections"]),
            "tracks": len(core.state.get("tracks", {})),
        }

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--candidates", type=Path, help="candidates.json from find_candidates.py")
    g.add_argument("--frames", type=Path, help="raw capture folder of sequential frames")
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="label keyframes JSON (default: data/tracks/<session>_labels.json)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    cand = args.candidates.resolve() if args.candidates else None
    frames = args.frames.resolve() if args.frames else None
    if cand and not cand.is_file():
        raise SystemExit(f"not found: {cand}")
    if frames and not frames.is_dir():
        raise SystemExit(f"not found: {frames}")

    # provisional session name for default out path
    if args.output:
        out = args.output if args.output.is_absolute() else ROOT / args.output
    else:
        if frames:
            name = frames.name
        else:
            data = json.loads(cand.read_text())
            sample = Path(data["candidates"][0]["prompt_frames"][0])
            name = (sample if sample.is_absolute() else ROOT / sample).parent.name
        out = ROOT / "data" / "tracks" / f"{name}_labels.json"

    core = BoxLabelApp(cand, frames, out)
    app = create_app(core)
    print(f"Box labeler → http://{args.host}:{args.port}/")
    print(f"Labels save to {out.relative_to(ROOT)}")
    print(f"{len(core.windows)} window(s), {len(core.session_frames)} session frames")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
