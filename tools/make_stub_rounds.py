#!/usr/bin/env python3
"""Generate schema-valid stub rounds for frontend/backend development.

Creates synthetic candidate boxes over placeholder (or real capture) images.
Satisfies §5.3 validity rules. Run early to unblock M1/M2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FRAMES = DATA / "frames"
ROUNDS_JSON = DATA / "rounds.json"


def write_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    """Minimal solid-color PNG (stdlib only)."""
    r, g, b = rgb
    raw = b""
    row = bytes([0] + [r, g, b] * width)  # filter=None per scanline
    raw = row * height
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def write_gradient_png(path: Path, width: int, height: int, seed: int) -> None:
    """Dark asphalt-ish gradient so stubs feel like a night camera."""
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            v = 18 + ((x + y + seed * 7) % 40)
            # faint lane lines
            if abs(x - width // 3) < 1 or abs(x - 2 * width // 3) < 1:
                v = min(255, v + 35)
            if y > height * 0.55 and (x + y) % 17 == 0:
                v = min(255, v + 20)
            row.extend([v, v + 2, v + 6])
        rows.append(bytes(row))
    compressed = zlib.compress(b"".join(rows), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


# Staging zone + finish line in 352x240 space (matches NYC TMC low-res cams).
W, H = 352, 240
STAGING = [[40, 140], [200, 130], [210, 200], [30, 210]]
FINISH = [[260, 80], [340, 100]]
TRAVEL = [1.0, -0.25]


def build_round(
    idx: int,
    n_candidates: int,
    green_index: int = 4,
    reveal_frames: int = 6,
) -> dict:
    """Build one valid stub round with moving boxes after green."""
    rid = f"r_{idx:03d}"
    total_frames = green_index + reveal_frames
    frame_paths = [f"{rid}/{i:03d}.png" for i in range(total_frames)]

    # Spread candidates left-to-right in staging zone.
    labels = "ABCD"
    base_xs = [50 + i * 45 for i in range(n_candidates)]
    base_y = 155
    track_ids = [10 + idx * 10 + i for i in range(n_candidates)]

    # Winner is always first track for odd rounds, second for even — variety.
    winner_slot = idx % n_candidates
    # Finish order: winner earliest, then others staggered; last may DNF sometimes.
    finish_offsets = list(range(n_candidates))
    # Put winner at offset 0 (earliest finish)
    finish_offsets.remove(winner_slot)
    finish_order_slots = [winner_slot] + finish_offsets

    finish_frame_index: dict[int, int | None] = {}
    for order, slot in enumerate(finish_order_slots):
        tid = track_ids[slot]
        if order == n_candidates - 1 and n_candidates >= 3 and idx % 5 == 0:
            finish_frame_index[tid] = None  # occasional DNF
        else:
            # green + 3 + order — strict no ties
            finish_frame_index[tid] = green_index + 3 + order

    # Ensure ≥2 finishers
    non_null = sum(1 for v in finish_frame_index.values() if v is not None)
    if non_null < 2:
        finish_frame_index[track_ids[0]] = green_index + 3
        finish_frame_index[track_ids[1]] = green_index + 4

    winner_tid = min(
        ((tid, fi) for tid, fi in finish_frame_index.items() if fi is not None),
        key=lambda x: x[1],
    )[0]

    candidates = []
    for i, tid in enumerate(track_ids):
        boxes = []
        x0 = base_xs[i] + (idx % 3) * 2
        y0 = base_y + (i % 2) * 3
        w, h = 36, 28
        finish_at = finish_frame_index[tid]

        for f in range(total_frames):
            if f <= green_index:
                # Near-stationary (±1px jitter max — under 8px threshold)
                x = x0 + (f % 2)
                y = y0
            else:
                # Move toward finish line after green
                progress = f - green_index
                # Winner moves faster
                speed = 28 if tid == winner_tid else 18 - i * 2
                if finish_at is None and progress > 4:
                    # DNF: drift off-axis / leave frame
                    x = x0 + progress * 8
                    y = y0 - progress * 12
                else:
                    x = x0 + int(progress * speed * 0.55)
                    y = y0 - int(progress * speed * 0.22)
            boxes.append({"frame": f, "x": int(x), "y": int(y), "w": w, "h": h})

        candidates.append(
            {
                "track_id": tid,
                "label": labels[i],
                "boxes": boxes,
            }
        )

    # Labels left-to-right by centre x at green_index
    def cx_at_green(c: dict) -> float:
        b = next(bb for bb in c["boxes"] if bb["frame"] == green_index)
        return b["x"] + b["w"] / 2

    candidates.sort(key=cx_at_green)
    for i, c in enumerate(candidates):
        c["label"] = labels[i]

    return {
        "id": rid,
        "fps": 0.5,
        "frames": frame_paths,
        "green_index": green_index,
        "candidates": candidates,
        "winner_track_id": winner_tid,
        "finish_frame_index": {str(k): v for k, v in finish_frame_index.items()},
        "finish_line": FINISH,
    }


def materialize_frames(round_data: dict, source_images: list[Path] | None) -> None:
    rid = round_data["id"]
    out_dir = FRAMES / rid
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    n = len(round_data["frames"])
    for i in range(n):
        dest = FRAMES / round_data["frames"][i]
        if source_images:
            src = source_images[i % len(source_images)]
            # Prefer copy; if png dest and jpg src, still copy bytes — browsers
            # sniff content. Use .jpg extension if source is jpeg.
            if src.suffix.lower() in {".jpg", ".jpeg"}:
                # rewrite frame path to .jpg
                new_rel = f"{rid}/{i:03d}.jpg"
                round_data["frames"][i] = new_rel
                dest = FRAMES / new_rel
                dest.write_bytes(src.read_bytes())
            else:
                shutil.copy2(src, dest)
        else:
            write_gradient_png(dest, W, H, seed=int(rid.split("_")[1]) * 100 + i)


def find_capture_images(captures_root: Path) -> list[Path]:
    if not captures_root.exists():
        return []
    imgs: list[Path] = []
    for p in sorted(captures_root.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file():
            imgs.append(p)
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate stub rounds.json + frames")
    ap.add_argument("-n", "--num-rounds", type=int, default=12)
    ap.add_argument(
        "--captures",
        type=Path,
        default=ROOT / "captures",
        help="Optional folder of real camera frames to reuse as placeholders",
    )
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic PNGs")
    args = ap.parse_args()

    source = None if args.synthetic else find_capture_images(args.captures)
    if source:
        print(f"Using {len(source)} capture frames from {args.captures}")
    else:
        print("Using synthetic gradient frames")

    FRAMES.mkdir(parents=True, exist_ok=True)
    rounds = []
    for i in range(1, args.num_rounds + 1):
        n_cand = 2 + (i % 3)  # 2, 3, or 4
        if n_cand > 4:
            n_cand = 4
        r = build_round(i, n_candidates=n_cand)
        materialize_frames(r, source)
        rounds.append(r)
        print(f"  wrote {r['id']} ({n_cand} candidates, winner={r['winner_track_id']})")

    payload = {
        "version": 1,
        "camera_id": "cam_01",
        "rounds": rounds,
    }
    ROUNDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    ROUNDS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {ROUNDS_JSON} ({len(rounds)} rounds)")

    # Also write a matching camera annotation stub.
    cam_path = ROOT / "camera" / "cam_01.json"
    cam_path.parent.mkdir(parents=True, exist_ok=True)
    cam_path.write_text(
        json.dumps(
            {
                "camera_id": "cam_01",
                "frame_width": W,
                "frame_height": H,
                "staging_zone": STAGING,
                "finish_line": FINISH,
                "travel_direction": TRAVEL,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {cam_path}")


if __name__ == "__main__":
    main()
