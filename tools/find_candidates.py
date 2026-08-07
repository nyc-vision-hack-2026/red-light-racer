#!/usr/bin/env python3
"""
Find candidate red-light -> green-light windows in a sequential frame dump.

No object detection required. Uses two cheap per-frame signals inside a
hand-drawn approach-lane mask:

    motion    = mean |frame[i] - frame[i-1]|      -> is traffic moving?
    occupancy = mean |frame[i] - background|      -> are cars present at all?

A queue waiting at a red light is the (low motion, high occupancy) quadrant.
Green onset is the rising edge of motion out of that state.

Usage
-----
    # one-time: draw the approach-lane polygon (needs a display)
    python tools/find_candidates.py pick-mask captures/SESSION/ --out mask.json

    # or convert an existing camera staging_zone (pixel coords) to a mask
    python tools/find_candidates.py from-camera camera/cam_01.json --out mask.json

    # find candidates
    python tools/find_candidates.py scan captures/SESSION/ --mask mask.json --out candidates.json

    # eyeball the results
    python tools/find_candidates.py sheets --candidates candidates.json --out sheets/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------- tuning knobs

WORK_W = 320          # downscale width for analysis; also acts as a low-pass filter
BLUR = 5              # gaussian kernel; suppresses rain speckle and sensor noise
CHUNK = 300           # frames per background-estimation chunk

MIN_RED_FRAMES = 5    # at 0.5 fps this is 10s. Real reds run 20-60s.
MIN_GREEN_FRAMES = 4  # need enough post-green frames to see a winner
PRE_FRAMES = 6        # frames of prompt footage to keep before green
POST_FRAMES = 16      # frames of reveal footage to keep after green
# Motion hysteresis trips AFTER cars already start rolling. End the prompt this
# many frames before detected green so "before" is still a stopped queue.
PROMPT_END_MARGIN = 4  # ~8s at 0.5 fps
# Also bias prompt toward mid-red (fraction of the way through the hold).
PROMPT_RED_FRAC = 0.65

MOT_LO_PCTL = 35      # hysteresis low edge  -> enter IDLE (stopped)
MOT_HI_PCTL = 65      # hysteresis high edge -> enter ACTIVE (flowing)

# Occupancy does NOT distinguish red from green -- the same cars are in frame
# either way. Motion does all of that work. Occupancy's ONLY job is rejecting an
# empty road at 4am, which otherwise looks exactly like a queue: no motion.
#
# So the gate belongs just above the sensor noise floor, not near the median. A
# median-style threshold silently drops short queues, which are the rounds you
# most want -- 2-3 candidates is a better guessing game than 6.
OCC_FLOOR_PCTL = 2    # this percentile of occupancy ~= bare road + noise
OCC_MARGIN = 0.05     # gate sits this far up the floor->p90 range

# Assumed capture interval for human-readable duration prints (matches capture_*.py default).
DEFAULT_INTERVAL_S = 2.0


# ---------------------------------------------------------------------- io bits

def natural_key(path):
    """Sort frame_9.jpg before frame_10.jpg."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_frames(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        sys.exit(f"no image files found in {folder}")
    return sorted(files, key=natural_key)


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    img = cv2.resize(img, (WORK_W, max(1, int(h * WORK_W / w))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (BLUR, BLUR), 0)


def build_mask(poly, shape):
    """poly is in source-image fractional coords so it survives any resize."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    pts = np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def load_approach_poly(mask_path):
    """Accept either find_candidates mask.json or camera/*.json with staging_zone."""
    data = json.load(open(mask_path))
    if "approach_poly" in data:
        return data["approach_poly"]
    if "staging_zone" in data:
        fw = float(data.get("frame_width") or 0)
        fh = float(data.get("frame_height") or 0)
        zone = data["staging_zone"]
        if fw <= 0 or fh <= 0:
            sys.exit(f"{mask_path}: staging_zone needs frame_width/frame_height")
        return [[x / fw, y / fh] for x, y in zone]
    sys.exit(f"{mask_path}: expected approach_poly or staging_zone")


# ------------------------------------------------------------------ mask picker

def cmd_pick_mask(args):
    if os.environ.get("DISPLAY") is None and sys.platform != "darwin":
        # On macOS OpenCV can still open a window without $DISPLAY.
        print("WARNING: no DISPLAY set; OpenCV window may fail. "
              "Use from-camera if you already have a staging_zone.")

    frames = list_frames(args.frames)
    ref = cv2.imread(frames[len(frames) // 2])
    if ref is None:
        sys.exit("could not read a reference frame")
    h, w = ref.shape[:2]
    pts = []

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow("mask")
    cv2.setMouseCallback("mask", on_click)
    print("left click = add point, right click = undo, ENTER = save, ESC = quit")
    print("draw around YOUR approach lanes only -- exclude cross traffic, or the")
    print("cross street's green light will look like your traffic moving.")

    while True:
        disp = ref.copy()
        if len(pts) > 1:
            cv2.polylines(disp, [np.array(pts, np.int32)], False, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(disp, p, 4, (0, 0, 255), -1)
        cv2.imshow("mask", disp)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            sys.exit("cancelled")
        if k in (13, 10) and len(pts) >= 3:
            break

    cv2.destroyAllWindows()
    poly = [[x / w, y / h] for x, y in pts]
    with open(args.out, "w") as f:
        json.dump({"approach_poly": poly}, f, indent=2)
    print(f"wrote {args.out} with {len(poly)} points")


def cmd_from_camera(args):
    poly = load_approach_poly(args.camera)
    with open(args.out, "w") as f:
        json.dump({"approach_poly": poly, "source": args.camera}, f, indent=2)
    print(f"wrote {args.out} with {len(poly)} points from {args.camera}")


# -------------------------------------------------------------------- the scan

def compute_signals(frames, poly):
    """Return per-frame motion and occupancy arrays, both length len(frames)."""
    n = len(frames)
    motion = np.zeros(n, np.float32)
    occupancy = np.zeros(n, np.float32)
    mask = None
    skipped = 0

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        # pad one frame back so motion is defined at the chunk boundary
        lo = max(0, start - 1)
        buf, idx = [], []
        for i in range(lo, stop):
            g = load_gray(frames[i])
            if g is None:
                skipped += 1
                continue
            buf.append(g)
            idx.append(i)
        if not buf:
            continue

        stack = np.stack(buf)
        if mask is None:
            mask = build_mask(poly, stack.shape[1:]) if poly else np.ones(stack.shape[1:], bool)
            if mask.sum() == 0:
                sys.exit("mask is empty -- check the polygon coordinates")

        # median over the chunk approximates the empty road under current lighting
        bg = np.median(stack, axis=0).astype(np.float32)

        for k, i in enumerate(idx):
            cur = stack[k].astype(np.float32)
            occupancy[i] = np.abs(cur - bg)[mask].mean()
            # Only compare temporally adjacent frames (skip gaps from failed reads).
            if k > 0 and idx[k - 1] == i - 1:
                motion[i] = np.abs(cur - stack[k - 1].astype(np.float32))[mask].mean()

        print(f"  frames {start}-{stop}  ", end="\r", flush=True)

    # Fill leading / gap motion with the next valid sample so percentiles aren't skewed by zeros.
    valid = np.flatnonzero(motion > 0)
    if len(valid):
        first = int(valid[0])
        motion[:first] = motion[first]
        for i in range(first + 1, n):
            if motion[i] == 0.0:
                motion[i] = motion[i - 1]
    if skipped:
        print(f"\nWARNING: skipped {skipped} unreadable frame(s)")
    else:
        print()
    return motion, occupancy


def find_windows(motion, occupancy):
    """Hysteresis state machine over motion, gated on occupancy."""
    mot_lo = float(np.percentile(motion, MOT_LO_PCTL))
    mot_hi = float(np.percentile(motion, MOT_HI_PCTL))
    occ_floor = float(np.percentile(occupancy, OCC_FLOOR_PCTL))
    occ_span = float(np.percentile(occupancy, 90) - occ_floor)
    occ_hi = occ_floor + OCC_MARGIN * max(occ_span, 1e-6)

    # A frame is "queued" when traffic in our lanes is stopped but cars are there.
    # The occupancy gate is what separates a red light from an empty road at 4am.
    state = "active"
    queue_start = None
    candidates = []

    for i in range(len(motion)):
        if state == "active":
            if motion[i] < mot_lo and occupancy[i] > occ_hi:
                state = "queued"
                queue_start = i
        else:  # queued
            if motion[i] > mot_hi:
                red_len = i - queue_start
                if red_len >= MIN_RED_FRAMES:
                    candidates.append({
                        "green_index": i,
                        "red_start": queue_start,
                        "red_len": int(red_len),
                    })
                state = "active"
                queue_start = None
            elif occupancy[i] < occ_hi * 0.9:
                # queue drained without a green edge -- probably a detection artifact
                state = "active"
                queue_start = None

    # require enough footage after green to actually resolve a winner
    out = []
    for c in candidates:
        g = c["green_index"]
        if g + MIN_GREEN_FRAMES >= len(motion):
            continue
        if motion[g:g + MIN_GREEN_FRAMES].mean() < mot_hi * 0.7:
            continue  # brief blip, not a real green
        out.append(c)

    return out, {
        "mot_lo": mot_lo,
        "mot_hi": mot_hi,
        "occ_hi": occ_hi,
        "occ_floor": occ_floor,
    }


def cmd_scan(args):
    frames = list_frames(args.frames)
    print(f"{len(frames)} frames")

    poly = None
    if args.mask:
        poly = load_approach_poly(args.mask)
    else:
        print("WARNING: no --mask given, using the whole frame. Cross traffic will")
        print("         look like your traffic moving. Draw a mask before trusting this.")

    motion, occupancy = compute_signals(frames, poly)
    cands, thresholds = find_windows(motion, occupancy)

    interval = args.interval
    rounds = []
    for c in cands:
        g = c["green_index"]
        red_start = c["red_start"]
        red_len = max(1, g - red_start)
        # Prompt from deep in the red hold — not the onset frames where cars creep.
        # Cap by (green - margin) and by ~65% through the red so we sit in the queue.
        onset_cap = g - PROMPT_END_MARGIN
        mid_cap = red_start + max(PRE_FRAMES, int(PROMPT_RED_FRAC * red_len))
        prompt_hi = max(red_start + 1, min(onset_cap, mid_cap))
        prompt_lo = max(red_start, prompt_hi - PRE_FRAMES)
        if prompt_hi <= prompt_lo:
            prompt_lo = red_start
            prompt_hi = max(red_start + 1, min(g, red_start + PRE_FRAMES))
        hi = min(len(frames), g + POST_FRAMES)
        # Reveal continues from the prompt freeze frame so Before→After is continuous
        # (prompt ends mid-red; jumping to g-1 left a multi-frame hole).
        reveal_lo = max(0, prompt_hi - 1)
        rounds.append({
            "green_index_global": g,
            "red_start_global": red_start,
            "red_len_frames": c["red_len"],
            "prompt_end_global": int(prompt_hi),
            # prompt must END well BEFORE green onset — cars roll before mot_hi trips
            "prompt_frames": [frames[i] for i in range(prompt_lo, prompt_hi)],
            "reveal_frames": [frames[i] for i in range(reveal_lo, hi)],
            "motion_at_green": float(motion[g]),
            "occupancy_at_green": float(occupancy[g]),
        })

    with open(args.out, "w") as f:
        json.dump({"thresholds": thresholds, "candidates": rounds}, f, indent=2)

    np.save(os.path.splitext(args.out)[0] + "_signals.npy",
            np.stack([motion, occupancy]))

    print(f"\nfound {len(rounds)} candidate windows -> {args.out}")
    print(f"thresholds: {thresholds}")
    for r in rounds[:20]:
        secs = r["red_len_frames"] * interval
        print(f"  green at frame {r['green_index_global']:6d}   "
              f"red held {r['red_len_frames']:3d} frames "
              f"({secs:g}s)")
    if len(rounds) > 20:
        print(f"  ... and {len(rounds) - 20} more")

    if args.plot:
        make_plot(motion, occupancy, rounds, thresholds, args.plot)


def make_plot(motion, occupancy, rounds, th, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    ax[0].plot(motion, lw=0.8)
    ax[0].axhline(th["mot_lo"], color="g", ls="--", lw=0.8)
    ax[0].axhline(th["mot_hi"], color="r", ls="--", lw=0.8)
    ax[0].set_ylabel("motion")
    ax[1].plot(occupancy, lw=0.8, color="purple")
    ax[1].axhline(th["occ_hi"], color="r", ls="--", lw=0.8)
    ax[1].set_ylabel("occupancy")
    for r in rounds:
        for a in ax:
            a.axvspan(r["red_start_global"], r["green_index_global"],
                      color="orange", alpha=0.25)
            a.axvline(r["green_index_global"], color="k", lw=0.8)
    ax[1].set_xlabel("frame index")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


# ------------------------------------------------------------------ contact sheets

def cmd_sheets(args):
    data = json.load(open(args.candidates))
    os.makedirs(args.out, exist_ok=True)
    written = 0
    for n, r in enumerate(data["candidates"]):
        paths = r["prompt_frames"][-4:] + r["reveal_frames"][:6]
        tiles = []
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                continue
            h, w = im.shape[:2]
            tiles.append(cv2.resize(im, (320, max(1, int(h * 320 / w)))))
        if not tiles:
            continue
        hh = min(t.shape[0] for t in tiles)
        tiles = [t[:hh] for t in tiles]
        rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
        wmax = max(r_.shape[1] for r_ in rows)
        rows = [np.pad(r_, ((0, 0), (0, wmax - r_.shape[1]), (0, 0))) for r_ in rows]
        sheet = np.vstack(rows)
        cv2.putText(sheet, f"green@{r['green_index_global']}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(args.out, f"cand_{n:03d}.jpg"), sheet)
        written += 1
    print(f"wrote {written} sheets to {args.out}")
    print("last row-1 tile before the divider is your prompt freeze frame.")


# ------------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pick-mask", help="interactively draw approach-lane polygon")
    p.add_argument("frames")
    p.add_argument("--out", default="mask.json")
    p.set_defaults(func=cmd_pick_mask)

    p = sub.add_parser("from-camera", help="build mask.json from camera staging_zone")
    p.add_argument("camera", help="camera/*.json with staging_zone + frame size")
    p.add_argument("--out", default="mask.json")
    p.set_defaults(func=cmd_from_camera)

    p = sub.add_parser("scan", help="find red->green candidate windows")
    p.add_argument("frames")
    p.add_argument("--mask", help="mask.json or camera JSON with staging_zone")
    p.add_argument("--out", default="candidates.json")
    p.add_argument("--plot", default="signals.png")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                   help="seconds between frames (for duration prints only)")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("sheets", help="write contact sheets for candidates")
    p.add_argument("--candidates", default="candidates.json")
    p.add_argument("--out", default="sheets")
    p.set_defaults(func=cmd_sheets)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
