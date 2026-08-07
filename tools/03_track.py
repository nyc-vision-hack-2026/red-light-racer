#!/usr/bin/env python3
"""Run frames through a Roboflow Workflow (vehicle detect + ByteTrack).

Writes data/tracks/<session>.json. Caches API responses by frame content hash.
Requires ROBOFLOW_API_KEY and optionally ROBOFLOW_WORKFLOW_URL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "tracks" / "_cache"


def frame_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def call_workflow(image_path: Path, workflow_url: str, api_key: str) -> dict:
    """POST image to Roboflow workflow. Response shape depends on workflow config."""
    boundary = "----RLRBoundary"
    body = b""
    # multipart with file + api_key
    def part(name: str, filename: str | None, data: bytes, ct: str | None = None) -> bytes:
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disp += f'; filename="{filename}"'
        headers = disp.encode() + b"\r\n"
        if ct:
            headers += f"Content-Type: {ct}\r\n".encode()
        headers += b"\r\n"
        return b"--" + boundary.encode() + b"\r\n" + headers + data + b"\r\n"

    raw = image_path.read_bytes()
    payload = (
        part("api_key", None, api_key.encode())
        + part("file", image_path.name, raw, "image/jpeg")
        + f"--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        workflow_url,
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def normalize_detections(wf_out: dict, frame_index: int) -> list[dict]:
    """Best-effort extract of boxes + track_id from common Roboflow shapes."""
    dets = []
    # Try several common nesting patterns
    candidates = []
    if isinstance(wf_out, dict):
        for key in ("predictions", "output", "outputs", "data"):
            if key in wf_out:
                candidates.append(wf_out[key])
        candidates.append(wf_out)
    for blob in candidates:
        preds = blob
        if isinstance(blob, dict):
            preds = blob.get("predictions") or blob.get("detections") or blob
        if isinstance(preds, list):
            for p in preds:
                if not isinstance(p, dict):
                    continue
                tid = p.get("tracker_id") or p.get("track_id") or p.get("detection_id")
                # xywh or xyxy
                if "x" in p and "width" in p:
                    x = p["x"] - p["width"] / 2
                    y = p["y"] - p["height"] / 2
                    w, h = p["width"], p["height"]
                elif "x1" in p:
                    x, y = p["x1"], p["y1"]
                    w, h = p["x2"] - p["x1"], p["y2"] - p["y1"]
                else:
                    continue
                dets.append(
                    {
                        "frame": frame_index,
                        "track_id": int(tid) if tid is not None else -1,
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                        "cls": p.get("class") or p.get("class_name") or "vehicle",
                        "conf": float(p.get("confidence") or p.get("conf") or 0),
                    }
                )
            if dets:
                break
    return dets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path, help="Folder of frames (data/raw/<session>)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output tracks JSON (default: data/tracks/<session>.json)",
    )
    ap.add_argument(
        "--workflow-url",
        default=os.environ.get("ROBOFLOW_WORKFLOW_URL", ""),
        help="Roboflow workflow infer URL",
    )
    ap.add_argument("--dry-run", action="store_true", help="List frames only")
    args = ap.parse_args()

    session = args.session_dir
    if not session.is_dir():
        raise SystemExit(f"Not a directory: {session}")

    frames = sorted(
        [p for p in session.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    if not frames:
        raise SystemExit("No frames found")

    out = args.output or (ROOT / "data" / "tracks" / f"{session.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"{len(frames)} frames → {out}")
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key or not args.workflow_url:
        print(
            "Set ROBOFLOW_API_KEY and ROBOFLOW_WORKFLOW_URL (or --workflow-url).\n"
            "Until then, stub tracks can be produced via make_stub_rounds.py.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    all_dets: list[dict] = []
    for i, frame in enumerate(frames):
        h = frame_hash(frame)
        cache_path = CACHE / f"{h}.json"
        if cache_path.exists():
            wf = json.loads(cache_path.read_text())
            print(f"[{i:04d}] cache hit {frame.name}")
        else:
            print(f"[{i:04d}] infer {frame.name} …", flush=True)
            try:
                wf = call_workflow(frame, args.workflow_url, api_key)
            except urllib.error.URLError as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                continue
            cache_path.write_text(json.dumps(wf))
        dets = normalize_detections(wf, i)
        all_dets.extend(dets)
        print(f"         {len(dets)} dets")

    payload = {
        "session": session.name,
        "frame_count": len(frames),
        "frames": [f.name for f in frames],
        "detections": all_dets,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out} ({len(all_dets)} detections)")


if __name__ == "__main__":
    main()
