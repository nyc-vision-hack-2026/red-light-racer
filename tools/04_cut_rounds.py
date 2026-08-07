#!/usr/bin/env python3
"""Segment tracks into rounds, infer green_index from motion, compute winners.

Green light is inferred from motion (median displacement of staging cars), not
from detecting the traffic light itself.

Shared geometry helpers (crossing / DNF) are factored here so a future live
ingest worker can import them rather than copy-paste.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Import validity from the app package when run from repo root.
import sys

sys.path.insert(0, str(ROOT))
from app.rounds import assert_round_valid  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def point_in_poly(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def box_center(b: dict) -> tuple[float, float]:
    return b["x"] + b["w"] / 2, b["y"] + b["h"] / 2


def box_fully_in_zone(b: dict, zone: list[list[float]]) -> bool:
    corners = [
        (b["x"], b["y"]),
        (b["x"] + b["w"], b["y"]),
        (b["x"] + b["w"], b["y"] + b["h"]),
        (b["x"], b["y"] + b["h"]),
    ]
    return all(point_in_poly(x, y, zone) for x, y in corners)


def leading_edge_midpoint(b: dict, travel: list[float]) -> tuple[float, float]:
    """Midpoint of the box edge facing travel_direction."""
    tx, ty = travel
    cx, cy = box_center(b)
    if abs(tx) >= abs(ty):
        # horizontal-ish: use left or right edge
        if tx >= 0:
            return b["x"] + b["w"], cy
        return b["x"], cy
    # vertical-ish
    if ty >= 0:
        return cx, b["y"] + b["h"]
    return cx, b["y"]


def side_of_segment(px: float, py: float, a: list[float], b: list[float]) -> float:
    """Signed cross product of AB × AP — sign indicates side."""
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


def crossed_finish(
    prev_box: dict,
    curr_box: dict,
    finish: list[list[float]],
    travel: list[float],
) -> bool:
    """True if leading-edge midpoint crossed the finish segment between frames."""
    a, b = finish[0], finish[1]
    p0 = leading_edge_midpoint(prev_box, travel)
    p1 = leading_edge_midpoint(curr_box, travel)
    s0 = side_of_segment(p0[0], p0[1], a, b)
    s1 = side_of_segment(p1[0], p1[1], a, b)
    if s0 == 0 or s1 == 0:
        return True
    if (s0 > 0) == (s1 > 0):
        return False
    # Also require movement roughly along travel
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx * travel[0] + dy * travel[1] <= 0:
        return False
    return True


def angle_diff_deg(v1: tuple[float, float], v2: list[float]) -> float:
    n1 = math.hypot(v1[0], v1[1]) or 1e-9
    n2 = math.hypot(v2[0], v2[1]) or 1e-9
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def is_dnf(
    boxes_by_frame: dict[int, dict],
    frame: int,
    prev_frame: int,
    *,
    frame_w: int,
    frame_h: int,
    travel: list[float],
    frames_since_green: int,
    diverge_deg: float = 40.0,
    max_frames: int = 20,
) -> bool:
    if frames_since_green >= max_frames:
        return True
    b = boxes_by_frame.get(frame)
    if b is None:
        return True
    # Left frame
    if b["x"] + b["w"] < 0 or b["x"] > frame_w or b["y"] + b["h"] < 0 or b["y"] > frame_h:
        return True
    pb = boxes_by_frame.get(prev_frame)
    if pb:
        c0, c1 = box_center(pb), box_center(b)
        motion = (c1[0] - c0[0], c1[1] - c0[1])
        if math.hypot(*motion) > 3 and angle_diff_deg(motion, travel) > diverge_deg:
            return True
    return False


def group_tracks(detections: list[dict]) -> dict[int, dict[int, dict]]:
    """track_id -> {frame_index -> box}"""
    tracks: dict[int, dict[int, dict]] = {}
    for d in detections:
        tid = int(d["track_id"])
        if tid < 0:
            continue
        tracks.setdefault(tid, {})[int(d["frame"])] = {
            "frame": int(d["frame"]),
            "x": float(d["x"]),
            "y": float(d["y"]),
            "w": float(d["w"]),
            "h": float(d["h"]),
        }
    return tracks


def near_stationary(boxes: dict[int, dict], frames: list[int], max_disp: float = 8.0) -> bool:
    if len(frames) < 2:
        return False
    centres = []
    for f in frames:
        b = boxes.get(f)
        if not b:
            return False
        centres.append(box_center(b))
    for i in range(1, len(centres)):
        dx = centres[i][0] - centres[i - 1][0]
        dy = centres[i][1] - centres[i - 1][1]
        if math.hypot(dx, dy) > max_disp:
            return False
    return True


def cut_rounds(
    tracks: dict[int, dict[int, dict]],
    camera: dict,
    frame_files: list[str],
    *,
    motion_thresh: float = 8.0,
    session_name: str = "session",
) -> list[dict]:
    zone = camera["staging_zone"]
    finish = camera["finish_line"]
    travel = camera["travel_direction"]
    fw, fh = camera["frame_width"], camera["frame_height"]
    n_frames = len(frame_files)
    rounds: list[dict] = []
    used_greens: set[int] = set()

    for green_candidate in range(2, n_frames - 3):
        if any(abs(green_candidate - g) < 8 for g in used_greens):
            continue

        # Find tracks stationary in zone for ≥2 frames before this index
        pre = list(range(green_candidate - 2, green_candidate))
        candidates_ids = []
        for tid, boxes in tracks.items():
            if not near_stationary(boxes, pre, max_disp=8.0):
                continue
            b_green = boxes.get(green_candidate)
            if not b_green or not box_fully_in_zone(b_green, zone):
                # Also accept centre-in-zone for low-res cams where full containment is harsh
                if not b_green:
                    continue
            cx, cy = box_center(boxes[pre[-1]])
            if not point_in_poly(cx, cy, zone):
                continue
            # Motion kick at green_candidate vs previous
            candidates_ids.append(tid)

        if len(candidates_ids) < 2:
            continue

        # Median displacement at green_candidate must exceed threshold
        disps = []
        for tid in candidates_ids:
            b0 = tracks[tid].get(green_candidate - 1)
            b1 = tracks[tid].get(green_candidate)
            if not b0 or not b1:
                continue
            c0, c1 = box_center(b0), box_center(b1)
            disps.append(math.hypot(c1[0] - c0[0], c1[1] - c0[1]))
        if not disps or sorted(disps)[len(disps) // 2] < motion_thresh:
            continue

        green_index = green_candidate
        # Walk forward until all candidates finish or DNF
        finish_idx: dict[int, int | None] = {tid: None for tid in candidates_ids}
        active = set(candidates_ids)
        for f in range(green_index + 1, min(n_frames, green_index + 21)):
            done = set()
            for tid in active:
                boxes = tracks[tid]
                if finish_idx[tid] is not None:
                    done.add(tid)
                    continue
                if is_dnf(
                    boxes,
                    f,
                    f - 1,
                    frame_w=fw,
                    frame_h=fh,
                    travel=travel,
                    frames_since_green=f - green_index,
                ):
                    finish_idx[tid] = None
                    done.add(tid)
                    continue
                pb, cb = boxes.get(f - 1), boxes.get(f)
                if pb and cb and crossed_finish(pb, cb, finish, travel):
                    finish_idx[tid] = f
                    done.add(tid)
            active -= done
            if not active:
                break

        end_frame = max(
            [green_index + 3]
            + [v for v in finish_idx.values() if v is not None]
            + [green_index + 5]
        )
        end_frame = min(end_frame + 1, n_frames - 1)

        # Build candidate list (2–4), label L→R at green
        cand_meta = []
        for tid in candidates_ids:
            b = tracks[tid].get(green_index)
            if not b:
                continue
            cand_meta.append((tid, b["x"] + b["w"] / 2))
        cand_meta.sort(key=lambda t: t[1])
        cand_meta = cand_meta[:4]
        if len(cand_meta) < 2:
            continue

        labels = "ABCD"
        prompt_start = max(0, green_index - 4)
        local_green = green_index - prompt_start
        local_end = end_frame - prompt_start
        if local_green < 2 or (local_end - local_green) < 2:
            continue

        candidates = []
        finish_local: dict[str, int | None] = {}
        for i, (tid, _) in enumerate(cand_meta):
            boxes = []
            for abs_f in range(prompt_start, end_frame + 1):
                b = tracks[tid].get(abs_f)
                if b:
                    boxes.append(
                        {
                            "frame": abs_f - prompt_start,
                            "x": round(b["x"], 1),
                            "y": round(b["y"], 1),
                            "w": round(b["w"], 1),
                            "h": round(b["h"], 1),
                        }
                    )
            if not any(bb["frame"] == local_green for bb in boxes):
                continue
            candidates.append({"track_id": tid, "label": labels[i], "boxes": boxes})
            fi = finish_idx.get(tid)
            finish_local[str(tid)] = None if fi is None else fi - prompt_start

        if len(candidates) < 2:
            continue

        finishers = [(int(k), v) for k, v in finish_local.items() if v is not None]
        if len(finishers) < 2:
            continue
        finishers.sort(key=lambda x: x[1])
        if finishers[0][1] == finishers[1][1]:
            continue
        winner = finishers[0][0]

        frame_paths = [frame_files[abs_f] for abs_f in range(prompt_start, end_frame + 1)]

        rnd = {
            "id": f"tmp_{green_index}",
            "fps": 0.5,
            "frames": frame_paths,
            "green_index": local_green,
            "candidates": candidates,
            "winner_track_id": winner,
            "finish_frame_index": finish_local,
            "finish_line": finish,
        }
        try:
            assert_round_valid(rnd)
        except Exception:
            continue

        used_greens.add(green_index)
        rounds.append(rnd)

    return rounds


def materialize(
    raw_rounds: list[dict],
    session_dir: Path,
    out_frames: Path,
    start_id: int = 1,
) -> list[dict]:
    """Copy frame slices into data/frames/r_XXX/ and rewrite paths."""
    import shutil

    finalized = []
    for i, rnd in enumerate(raw_rounds):
        rid = f"r_{start_id + i:03d}"
        dest = out_frames / rid
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        new_frames = []
        for j, name in enumerate(rnd["frames"]):
            src = session_dir / name
            ext = src.suffix.lower() or ".jpg"
            rel = f"{rid}/{j:03d}{ext}"
            shutil.copy2(src, out_frames / rel)
            new_frames.append(rel)
        out = {
            "id": rid,
            "fps": rnd["fps"],
            "frames": new_frames,
            "green_index": rnd["green_index"],
            "candidates": rnd["candidates"],
            "winner_track_id": rnd["winner_track_id"],
            "finish_frame_index": rnd["finish_frame_index"],
            "finish_line": rnd["finish_line"],
        }
        assert_round_valid(out)
        finalized.append(out)
        print(f"  kept {rid} green={out['green_index']} winner={out['winner_track_id']}")
    return finalized


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tracks_json", type=Path)
    ap.add_argument("--camera", type=Path, default=ROOT / "camera" / "cam_01.json")
    ap.add_argument("--session-dir", type=Path, required=True, help="Raw frames directory")
    ap.add_argument("--motion-thresh", type=float, default=8.0)
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data" / "rounds_candidates.json")
    args = ap.parse_args()

    tracks_doc = load_json(args.tracks_json)
    camera = load_json(args.camera)
    detections = tracks_doc["detections"]
    frame_files = tracks_doc.get("frames") or sorted(
        p.name for p in args.session_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    tracks = group_tracks(detections)
    print(f"{len(tracks)} tracks, {len(frame_files)} frames")

    raw = cut_rounds(
        tracks,
        camera,
        frame_files,
        motion_thresh=args.motion_thresh,
        session_name=args.session_dir.name,
    )
    print(f"{len(raw)} candidate rounds before materialize")

    finalized = materialize(raw, args.session_dir, ROOT / "data" / "frames")
    payload = {"version": 1, "camera_id": camera["camera_id"], "rounds": finalized}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output} — run 05_review.py next before promoting to rounds.json")


if __name__ == "__main__":
    main()
