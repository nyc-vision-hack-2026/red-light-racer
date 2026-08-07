#!/usr/bin/env python3
"""
Score whether an NYC traffic camera is a good Red Light Racer source.

The game needs a mid-frame stop-line queue that races toward a near-camera
finish, with clean enough red→green motion edges to harvest rounds offline.

What this measures
------------------
1. Geometry (optional camera/*.json)
   - staging mid-frame, finish downstream toward the lens
   - travel_direction aligns with staging→finish
   - staging area and race runway are usable sizes

2. Signal quality (frame dump + approach mask)
   - motion high/low separation (can hysteresis see red vs green?)
   - occupancy above empty-road floor during stops
   - red→green candidate rate and red hold duration

3. Image quality (sampled frames)
   - raindrop/glare proxy and blur — tracking dies when these spike

4. Vision judge (optional, Gemini 3.6 Flash)
   - layout suitability for mid-frame queue → near-camera finish
   - current conditions (rain/lens drops) kept separate from layout

Usage
-----
    # score an existing capture (mask from camera JSON or mask.json)
    python tools/score_camera.py score captures/SESSION/ \\
        --mask camera/park_ave_23st.json --out reports/park_ave.json

    # same + Gemini vision layout/conditions judge
    python tools/score_camera.py score captures/SESSION/ --mask camera/X.json --vision

    # short live probe: grab N minutes then score (whole-frame if no mask yet)
    python tools/score_camera.py probe "Park Ave @ 23 St" --minutes 8 --vision

    # batch-score every annotated camera that has a capture folder
    python tools/score_camera.py batch --vision --out reports/batch.json

    # still-only Gemini screen of named NYC cams (no motion signals)
    python tools/score_camera.py screen "Madison Ave @ 42 St" "1 Ave @ 57 st" --vision
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Reuse the cheap motion/occupancy pipeline — same thresholds as round finding.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from find_candidates import (  # noqa: E402
    DEFAULT_INTERVAL_S,
    compute_signals,
    find_windows,
    list_frames,
    load_approach_poly,
)

CAMERAS_URL = "https://webcams.nyctmc.org/api/cameras"
USER_AGENT = "Mozilla/5.0 (compatible; redlight-racer-score/1.0)"

# ---------------------------------------------------------------- thresholds
# Tuned against the three annotated cams + game validity rules (2–4 cars,
# ≥3 prompt + ≥3 reveal frames, finish still large near camera).

MIN_FRAMES = 60                 # ~2 min at 0.5 fps — below this, rate is noisy
TARGET_INTERVAL_S = 2.0

# Geometry
STAGING_Y_LO, STAGING_Y_HI = 0.30, 0.78
STAGING_AREA_LO, STAGING_AREA_HI = 0.03, 0.28
MIN_RUNWAY = 0.12               # staging→finish / frame diagonal
MIN_TRAVEL_ALIGN = 0.55

# Signals
MIN_MOTION_RATIO = 1.6          # p65 / p35
MIN_MOTION_GAP = 4.0            # absolute grey-level separation
MIN_CAND_PER_HOUR = 2.0
MAX_CAND_PER_HOUR = 40.0        # above this often means cross-traffic bleed
RED_LEN_LO, RED_LEN_HI = 5, 45  # frames @ ~0.5 fps → ~10–90s

# Image quality (harder on rainy night dumps — soft penalties, not instant kill)
MAX_GLARE_FRAC = 0.12
MIN_SHARPNESS = 18.0            # Laplacian variance on downscaled grey

# Vision (Gemini 3.6 Flash via Vertex AI or GEMINI_API_KEY)
VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-3.6-flash")
DEFAULT_GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")


# --------------------------------------------------------------------- helpers

def http_get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "https://webcams.nyctmc.org/",
            "Accept": "*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def resolve_camera(query: str) -> dict:
    cameras = json.loads(http_get(CAMERAS_URL))
    q = query.strip().lower()
    exact = [c for c in cameras if c.get("name", "").lower() == q]
    if exact:
        return exact[0]
    partial = [c for c in cameras if q in c.get("name", "").lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise SystemExit(f"No camera matched {query!r}")
    names = "\n  ".join(c["name"] for c in partial[:20])
    raise SystemExit(f"Ambiguous camera {query!r}. Matches:\n  {names}")


def slugify(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "camera"


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def soft_band(x: float, lo: float, hi: float, soft: float = 0.25) -> float:
    """1.0 inside [lo,hi], linear falloff outside over `soft`*(hi-lo) or soft abs."""
    if lo <= x <= hi:
        return 1.0
    width = max(hi - lo, 1e-6)
    margin = max(soft * width, soft)
    if x < lo:
        return clamp01(1.0 - (lo - x) / margin)
    return clamp01(1.0 - (x - hi) / margin)


# ------------------------------------------------------------------- geometry

def score_geometry(cam: dict) -> dict:
    """Return geometry sub-score + metrics from camera/*.json."""
    fw = float(cam.get("frame_width") or 0)
    fh = float(cam.get("frame_height") or 0)
    zone = cam.get("staging_zone") or []
    finish = cam.get("finish_line") or []
    travel = cam.get("travel_direction") or [0, 1]

    hard_fails: list[str] = []
    notes: list[str] = []
    metrics: dict = {}

    if fw <= 0 or fh <= 0 or len(zone) < 3 or len(finish) < 2:
        return {
            "score": None,
            "available": False,
            "hard_fails": ["missing staging_zone / finish_line / frame size"],
            "notes": [],
            "metrics": {},
        }

    zs = np.array(zone, dtype=np.float64)
    fs = np.array(finish, dtype=np.float64)
    staging_c = zs.mean(axis=0)
    finish_c = fs.mean(axis=0)
    staging_yn = staging_c[1] / fh
    finish_yn = finish_c[1] / fh

    # polygon area via shoelace, fraction of frame
    x, y = zs[:, 0], zs[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area_frac = float(area / (fw * fh))

    delta = finish_c - staging_c
    diag = float(np.hypot(fw, fh))
    runway = float(np.linalg.norm(delta) / diag)
    travel_v = np.array(travel, dtype=np.float64)
    tn = np.linalg.norm(travel_v)
    dn = np.linalg.norm(delta)
    align = float(np.dot(travel_v, delta) / (tn * dn + 1e-9)) if tn > 0 and dn > 0 else 0.0

    metrics.update({
        "staging_y_norm": round(staging_yn, 3),
        "finish_y_norm": round(finish_yn, 3),
        "staging_area_frac": round(area_frac, 3),
        "runway_frac": round(runway, 3),
        "travel_align": round(align, 3),
        "finish_downstream": bool(finish_yn > staging_yn - 0.02),
    })

    # Hard fails: finish upstream of staging (wrong race direction for toward-cam setup)
    if finish_yn + 0.03 < staging_yn and align < 0.3:
        hard_fails.append("finish looks upstream of staging / travel misaligned")
    if align < MIN_TRAVEL_ALIGN:
        hard_fails.append(f"travel_direction poorly aligned with staging→finish ({align:.2f})")
    if runway < MIN_RUNWAY * 0.5:
        hard_fails.append(f"race runway too short ({runway:.2f} of diagonal)")

    if not (STAGING_Y_LO <= staging_yn <= STAGING_Y_HI):
        notes.append(f"staging y={staging_yn:.2f} outside mid-frame band "
                     f"[{STAGING_Y_LO},{STAGING_Y_HI}]")
    if area_frac < STAGING_AREA_LO:
        notes.append("staging zone tiny — may struggle to fit 2–4 cars")
    elif area_frac > STAGING_AREA_HI:
        notes.append("staging zone wide — risk of parked/opposite/cross traffic")

    # Composite: mid staging, finish nearer camera (higher y), area, runway, align
    parts = {
        "staging_y": soft_band(staging_yn, STAGING_Y_LO, STAGING_Y_HI),
        "finish_near": soft_band(finish_yn, 0.70, 1.05, soft=0.4),
        "area": soft_band(area_frac, STAGING_AREA_LO, STAGING_AREA_HI, soft=0.5),
        "runway": soft_band(runway, MIN_RUNWAY, 0.55, soft=0.4),
        "align": clamp01((align - 0.3) / 0.7),
        "downstream": 1.0 if finish_yn >= staging_yn - 0.02 else 0.2,
    }
    score = float(np.mean(list(parts.values())))
    return {
        "score": round(score, 3),
        "available": True,
        "parts": {k: round(v, 3) for k, v in parts.items()},
        "hard_fails": hard_fails,
        "notes": notes,
        "metrics": metrics,
    }


# -------------------------------------------------------------- image quality

def score_image_quality(frames: list[str], n_samples: int = 12) -> dict:
    if not frames:
        return {"score": 0.0, "hard_fails": ["no frames"], "notes": [], "metrics": {}}

    idxs = np.linspace(0, len(frames) - 1, num=min(n_samples, len(frames)), dtype=int)
    glares, sharps = [], []
    for i in idxs:
        img = cv2.imread(frames[i])
        if img is None:
            continue
        small = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # glare / rain-blob proxy: near-saturated bright fraction
        glares.append(float((gray > 235).mean()))
        sharps.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    if not glares:
        return {"score": 0.0, "hard_fails": ["unreadable frames"], "notes": [], "metrics": {}}

    glare = float(np.median(glares))
    sharp = float(np.median(sharps))
    notes, hard = [], []
    if glare > MAX_GLARE_FRAC:
        notes.append(f"high glare/rain blob fraction ({glare:.3f})")
    if sharp < MIN_SHARPNESS:
        notes.append(f"soft/blurry frames (laplacian {sharp:.1f})")

    glare_s = clamp01(1.0 - glare / (MAX_GLARE_FRAC * 2.5))
    sharp_s = clamp01(sharp / 80.0)
    score = 0.55 * glare_s + 0.45 * sharp_s
    return {
        "score": round(score, 3),
        "hard_fails": hard,
        "notes": notes,
        "metrics": {
            "glare_frac": round(glare, 4),
            "sharpness": round(sharp, 2),
            "samples": len(glares),
        },
    }


# ------------------------------------------------------------------- signals

def score_signals(
    frames: list[str],
    poly: list[list[float]] | None,
    interval_s: float,
) -> dict:
    hard, notes = [], []
    if len(frames) < MIN_FRAMES:
        notes.append(f"only {len(frames)} frames — rate estimates are noisy "
                     f"(want ≥{MIN_FRAMES})")

    motion, occupancy = compute_signals(frames, poly)
    cands, thresholds = find_windows(motion, occupancy)

    duration_h = max(len(frames) * interval_s / 3600.0, 1e-9)
    rate = len(cands) / duration_h
    red_lens = [c["red_len"] for c in cands]
    med_red = float(np.median(red_lens)) if red_lens else 0.0

    p35 = float(np.percentile(motion, 35))
    p65 = float(np.percentile(motion, 65))
    gap = p65 - p35
    ratio = p65 / (p35 + 1e-6)
    occ_floor = float(np.percentile(occupancy, 2))
    occ_p90 = float(np.percentile(occupancy, 90))
    occ_span = occ_p90 - occ_floor

    # Fraction of frames that look queued under the same thresholds.
    queued = (motion < thresholds["mot_lo"]) & (occupancy > thresholds["occ_hi"])
    queue_frac = float(queued.mean())

    metrics = {
        "n_frames": len(frames),
        "duration_min": round(len(frames) * interval_s / 60.0, 2),
        "n_candidates": len(cands),
        "candidates_per_hour": round(rate, 2),
        "median_red_frames": round(med_red, 1),
        "median_red_seconds": round(med_red * interval_s, 1),
        "motion_p35": round(p35, 2),
        "motion_p65": round(p65, 2),
        "motion_gap": round(gap, 2),
        "motion_ratio": round(ratio, 2),
        "occ_span": round(occ_span, 2),
        "queue_frac": round(queue_frac, 3),
        "thresholds": {k: round(float(v), 3) for k, v in thresholds.items()},
        "used_mask": poly is not None,
    }

    if poly is None:
        notes.append("no approach mask — cross traffic will inflate motion; treat as provisional")

    if ratio < MIN_MOTION_RATIO and gap < MIN_MOTION_GAP:
        hard.append(f"weak red/green motion contrast (ratio={ratio:.2f}, gap={gap:.1f})")
    elif ratio < MIN_MOTION_RATIO:
        notes.append(f"modest motion ratio ({ratio:.2f}) — rain/noise may muddy edges")

    if len(cands) == 0 and len(frames) >= MIN_FRAMES:
        hard.append("zero red→green candidates in this dump")
    if rate > MAX_CAND_PER_HOUR and poly is not None:
        notes.append(f"very high candidate rate ({rate:.0f}/h) — check mask for cross traffic")
    if red_lens and not (RED_LEN_LO <= med_red <= RED_LEN_HI):
        notes.append(f"median red hold {med_red:.0f} frames atypical for a light cycle")
    if queue_frac < 0.08:
        notes.append("rarely sees a stopped queue — light traffic or wrong lanes")
    elif queue_frac > 0.75:
        notes.append("almost always 'queued' — congestion or stuck detection")

    parts = {
        "motion_contrast": clamp01((ratio - 1.2) / 2.5) * 0.5
        + clamp01(gap / 15.0) * 0.5,
        "harvest_rate": soft_band(rate, MIN_CAND_PER_HOUR, 24.0, soft=0.6)
        if len(frames) >= MIN_FRAMES // 2 else 0.5,
        "red_duration": soft_band(med_red, RED_LEN_LO, RED_LEN_HI, soft=0.5)
        if red_lens else 0.2,
        "queue_presence": soft_band(queue_frac, 0.12, 0.55, soft=0.5),
        "occ_dynamic": clamp01(occ_span / 12.0),
    }
    score = float(np.mean(list(parts.values())))
    return {
        "score": round(score, 3),
        "parts": {k: round(v, 3) for k, v in parts.items()},
        "hard_fails": hard,
        "notes": notes,
        "metrics": metrics,
        "candidates": [
            {
                "green_index": int(c["green_index"]),
                "red_start": int(c["red_start"]),
                "red_len": int(c["red_len"]),
            }
            for c in cands
        ],
    }


# -------------------------------------------------------------- Gemini vision

VISION_PROMPT = """You are judging NYC DOT traffic-camera stills for "Red Light Racer".

Game needs:
- Cars waiting at a red in a mid-frame staging/queue zone (approach lanes).
- A finish line near the camera where cars are still large (dozens of pixels wide).
- Primary traffic flowing toward the camera (or clearly toward a near-camera finish).
- Ideally 2–4 side-by-side lanes of through traffic (not only turning/cross street).
- Stable fixed camera (not PTZ chaos / highway-only / sky-heavy).

IMPORTANT: Separate permanent LAYOUT from temporary CONDITIONS.
- layout_score: would this camera work on a clear day? Geometry/viewpoint only.
- conditions_score: how usable are THESE frames right now (rain, lens drops, night glare)?

Return ONLY JSON with this schema:
{
  "layout_score": 0.0-1.0,
  "conditions_score": 0.0-1.0,
  "facing": "toward_camera" | "away" | "across" | "mixed" | "unclear",
  "lanes_usable": <int>,
  "stop_line_or_crosswalk_visible": true/false,
  "finish_near_camera_feasible": true/false,
  "cars_large_enough_near_bottom": true/false,
  "layout_issues": [string],
  "condition_issues": [string],
  "why": "one sentence"
}
"""


def _gcp_project() -> str | None:
    if DEFAULT_GCP_PROJECT:
        return DEFAULT_GCP_PROJECT
    try:
        return subprocess.check_output(
            ["gcloud", "config", "get-value", "project"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except Exception:
        return None


def _gcloud_access_token() -> str | None:
    try:
        return subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except Exception:
        return None


def _encode_frame_b64(path: str, max_side: int = 768) -> tuple[str, str]:
    raw = Path(path).read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return base64.b64encode(raw).decode("ascii"), "image/jpeg"
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return base64.b64encode(raw).decode("ascii"), "image/jpeg"
    return base64.b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"


def _gemini_generate(parts: list[dict]) -> dict:
    """Call gemini-3.6-flash. Prefer Vertex (gcloud auth), else GEMINI_API_KEY."""
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        },
    }
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    project = _gcp_project()
    token = _gcloud_access_token() if project else None

    errors: list[str] = []
    if project and token:
        url = (
            f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global"
            f"/publishers/google/models/{VISION_MODEL}:generateContent"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            errors.append(f"vertex: {e.read().decode()[:240]}")
        except Exception as e:
            errors.append(f"vertex: {e}")

    if api_key:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{VISION_MODEL}:generateContent"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            errors.append(f"ai_studio: {e.read().decode()[:240]}")
        except Exception as e:
            errors.append(f"ai_studio: {e}")

    raise RuntimeError(
        "Gemini vision unavailable. Set GEMINI_API_KEY or gcloud auth "
        f"with Vertex access. Details: {' | '.join(errors) or 'no credentials'}"
    )


def _parse_vision_json(resp: dict) -> dict:
    text = ""
    for part in resp.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "text" in part:
            text += part["text"]
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def score_vision(frames: list[str], n_samples: int = 3) -> dict:
    """Gemini layout + conditions judge over a few stills."""
    if not frames:
        return {
            "score": None,
            "available": False,
            "hard_fails": ["no frames"],
            "notes": [],
            "metrics": {},
        }

    idxs = np.linspace(0, len(frames) - 1, num=min(n_samples, len(frames)), dtype=int)
    parts: list[dict] = [{"text": VISION_PROMPT}]
    used = []
    for i in idxs:
        b64, mime = _encode_frame_b64(frames[int(i)])
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
        used.append(frames[int(i)])

    try:
        raw = _gemini_generate(parts)
        data = _parse_vision_json(raw)
    except Exception as e:
        return {
            "score": None,
            "available": False,
            "hard_fails": [],
            "notes": [f"vision judge failed: {e}"],
            "metrics": {"model": VISION_MODEL},
        }

    layout = clamp01(float(data.get("layout_score", 0)))
    conditions = clamp01(float(data.get("conditions_score", 0)))
    # Camera pick uses layout; conditions only soft-penalize tonight's usability.
    score = 0.75 * layout + 0.25 * conditions

    hard, notes = [], []
    facing = data.get("facing")
    if facing in {"away", "across"}:
        hard.append(f"viewpoint facing={facing} (want toward_camera)")
    if data.get("finish_near_camera_feasible") is False:
        hard.append("vision: finish near camera not feasible")
    if data.get("cars_large_enough_near_bottom") is False:
        notes.append("vision: cars may be too small near bottom of frame")
    for issue in data.get("layout_issues") or []:
        notes.append(f"layout: {issue}")
    for issue in data.get("condition_issues") or []:
        notes.append(f"conditions: {issue}")
    if conditions < 0.35:
        notes.append("poor current conditions — prefer a clearer-day recapture before tracking")

    return {
        "score": round(score, 3),
        "available": True,
        "hard_fails": hard,
        "notes": notes,
        "metrics": {
            "model": VISION_MODEL,
            "layout_score": round(layout, 3),
            "conditions_score": round(conditions, 3),
            "facing": facing,
            "lanes_usable": data.get("lanes_usable"),
            "stop_line_or_crosswalk_visible": data.get("stop_line_or_crosswalk_visible"),
            "finish_near_camera_feasible": data.get("finish_near_camera_feasible"),
            "cars_large_enough_near_bottom": data.get("cars_large_enough_near_bottom"),
            "why": data.get("why"),
            "frames": used,
            "raw": data,
        },
    }


# ------------------------------------------------------------------- combine

def verdict_from(parts: dict) -> tuple[str, float]:
    """
    overall in [0,1]. Verdict:
      good     ≥ 0.72 and no hard fails
      marginal ≥ 0.50
      poor     otherwise or any hard fail
    """
    hard = []
    for key in ("geometry", "signals", "image", "vision"):
        hard.extend(parts.get(key, {}).get("hard_fails") or [])

    scores = []
    weights = []
    if parts.get("geometry", {}).get("available"):
        scores.append(parts["geometry"]["score"])
        weights.append(0.22)
    if parts.get("signals", {}).get("score") is not None:
        scores.append(parts["signals"]["score"])
        weights.append(0.38)
    if parts.get("image", {}).get("score") is not None:
        scores.append(parts["image"]["score"])
        weights.append(0.12)
    if parts.get("vision", {}).get("available") and parts["vision"].get("score") is not None:
        scores.append(parts["vision"]["score"])
        weights.append(0.28)

    if not scores:
        return "poor", 0.0

    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    overall = float(np.dot(w, np.array(scores, dtype=np.float64)))

    if hard:
        return "poor", round(min(overall, 0.49), 3)
    if overall >= 0.72:
        return "good", round(overall, 3)
    if overall >= 0.50:
        return "marginal", round(overall, 3)
    return "poor", round(overall, 3)


def evaluate(
    frames_dir: Path,
    mask_path: Path | None,
    camera_path: Path | None,
    interval_s: float,
    *,
    use_vision: bool = False,
) -> dict:
    frames = list_frames(str(frames_dir))
    poly = None
    cam = None

    if camera_path and camera_path.is_file():
        cam = json.load(open(camera_path))
    if mask_path and mask_path.is_file():
        poly = load_approach_poly(str(mask_path))
    elif cam and cam.get("staging_zone"):
        poly = load_approach_poly(str(camera_path))

    geo = score_geometry(cam) if cam else {
        "score": None, "available": False, "hard_fails": [], "notes": [
            "no camera JSON — geometry not scored (draw with tools/02_annotate.py)"
        ], "metrics": {},
    }
    sig = score_signals(frames, poly, interval_s)
    img = score_image_quality(frames)
    vision = score_vision(frames) if use_vision else {
        "score": None, "available": False, "hard_fails": [], "notes": [], "metrics": {},
    }

    parts = {"geometry": geo, "signals": sig, "image": img, "vision": vision}
    verdict, overall = verdict_from(parts)

    reasons = []
    for section, data in parts.items():
        for h in data.get("hard_fails") or []:
            reasons.append(f"[fail:{section}] {h}")
        for n in data.get("notes") or []:
            reasons.append(f"[note:{section}] {n}")

    return {
        "verdict": verdict,
        "score": overall,
        "frames_dir": str(frames_dir),
        "camera_id": (cam or {}).get("camera_id"),
        "mask": str(mask_path) if mask_path else (
            str(camera_path) if cam and cam.get("staging_zone") else None
        ),
        "geometry": geo,
        "signals": {k: v for k, v in sig.items() if k != "candidates"},
        "candidates": sig.get("candidates", []),
        "image": img,
        "vision": vision,
        "reasons": reasons,
    }


def print_report(report: dict) -> None:
    v = report["verdict"].upper()
    print(f"\n=== {report.get('camera_id') or report.get('name') or report['frames_dir']} ===")
    print(f"verdict: {v}   score={report['score']}")
    if report.get("signals") and report["signals"].get("metrics"):
        sig = report["signals"]["metrics"]
        print(f"signals: {sig['n_candidates']} candidates in {sig['duration_min']} min "
              f"({sig['candidates_per_hour']}/h), "
              f"motion_ratio={sig['motion_ratio']}, "
              f"median_red={sig['median_red_seconds']}s")
    if report.get("geometry", {}).get("available"):
        g = report["geometry"]["metrics"]
        print(f"geometry: staging_y={g['staging_y_norm']} finish_y={g['finish_y_norm']} "
              f"align={g['travel_align']} area={g['staging_area_frac']}")
    if report.get("image", {}).get("metrics"):
        iq = report["image"]["metrics"]
        print(f"image: glare={iq.get('glare_frac')} sharpness={iq.get('sharpness')}")
    if report.get("vision", {}).get("available"):
        vm = report["vision"]["metrics"]
        print(f"vision({vm.get('model')}): layout={vm.get('layout_score')} "
              f"conditions={vm.get('conditions_score')} facing={vm.get('facing')}")
        if vm.get("why"):
            print(f"  why: {vm['why']}")
    for r in report.get("reasons") or []:
        print(f"  {r}")


# ----------------------------------------------------------------------- cmds

def cmd_score(args):
    mask = Path(args.mask) if args.mask else None
    camera = Path(args.camera) if args.camera else None
    # A camera JSON doubles as mask + geometry when --camera is omitted.
    if camera is None and mask and mask.is_file():
        try:
            if "staging_zone" in json.load(open(mask)):
                camera = mask
        except Exception:
            pass

    report = evaluate(
        Path(args.frames), mask, camera, args.interval, use_vision=args.vision,
    )
    print_report(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {out}")
    return report


def cmd_probe(args):
    cam = resolve_camera(args.query)
    if str(cam.get("isOnline")).lower() not in {"true", "1", "yes"}:
        print(f"WARNING: camera reports isOnline={cam.get('isOnline')!r}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(cam["name"])
    out_dir = Path(args.out_dir) / f"probe_{slug}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    n = int(round(args.minutes * 60.0 / args.interval))
    print(f"Probing {cam['name']} for ~{args.minutes:g} min "
          f"({n} frames @ {args.interval:g}s) → {out_dir}")

    image_url = cam["imageUrl"]
    for i in range(n):
        t0 = time.monotonic()
        path = out_dir / f"{i:04d}.jpg"
        try:
            data = http_get(f"{image_url}?t={int(time.time() * 1000)}")
            if len(data) < 1000:
                raise ValueError("tiny response")
            path.write_bytes(data)
            print(f"  {i+1}/{n}  {path.name}", end="\r", flush=True)
        except Exception as e:
            print(f"\n  frame {i} failed: {e}")
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, args.interval - elapsed))
    print()

    meta = {
        "camera": {k: cam.get(k) for k in ("id", "name", "latitude", "longitude", "area", "imageUrl")},
        "interval_s": args.interval,
        "minutes": args.minutes,
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))

    camera_json = ROOT / "camera" / f"{slug}.json"
    mask = camera_json if camera_json.is_file() else None
    report = evaluate(out_dir, mask, mask, args.interval, use_vision=args.vision)
    report["probe"] = meta
    print_report(report)

    report_path = out_dir / "score.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
    if report["verdict"] != "good" and mask is None:
        print("tip: draw staging/finish with tools/02_annotate.py on this probe, "
              "then re-run score — geometry + masked signals are the real test.")
    return report


def discover_pairs() -> list[tuple[str, Path, Path]]:
    """(camera_id, camera_json, best_capture_dir) for each annotated camera."""
    pairs = []
    seen_ids: set[str] = set()
    cam_dir = ROOT / "camera"
    cap_root = ROOT / "captures"
    if not cam_dir.is_dir() or not cap_root.is_dir():
        return pairs
    # Prefer camera/<id>.json over aliases like cam_01.json.
    paths = sorted(cam_dir.glob("*.json"), key=lambda p: (p.stem.startswith("cam_"), p.name))
    for cam_path in paths:
        if cam_path.name.startswith("_"):
            continue
        data = json.load(open(cam_path))
        cid = data.get("camera_id") or cam_path.stem
        if cid in seen_ids:
            continue
        slugs = {cam_path.stem, cid}
        if "23st" in cid:
            slugs.add(cid.replace("23st", "23_st"))
        candidates = []
        for d in cap_root.iterdir():
            if not d.is_dir():
                continue
            if any(d.name.startswith(s) for s in slugs):
                n = sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
                if n >= 10:
                    candidates.append((n, d))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        pairs.append((cid, cam_path, candidates[0][1]))
        seen_ids.add(cid)
    return pairs


def cmd_batch(args):
    pairs = discover_pairs()
    if not pairs:
        raise SystemExit("no camera/*.json with matching captures/ found")
    reports = []
    for cid, cam_path, frames in pairs:
        print(f"\nscoring {cid} on {frames.name} …")
        report = evaluate(
            frames, cam_path, cam_path, args.interval, use_vision=args.vision,
        )
        print_report(report)
        reports.append(report)
    summary = {
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "vision": bool(args.vision),
        "results": [
            {
                "camera_id": r.get("camera_id"),
                "verdict": r["verdict"],
                "score": r["score"],
                "candidates_per_hour": r["signals"]["metrics"]["candidates_per_hour"],
                "motion_ratio": r["signals"]["metrics"]["motion_ratio"],
                "layout_score": (r.get("vision") or {}).get("metrics", {}).get("layout_score"),
                "conditions_score": (r.get("vision") or {}).get("metrics", {}).get("conditions_score"),
                "frames_dir": r["frames_dir"],
            }
            for r in reports
        ],
    }
    summary["results"].sort(key=lambda x: (-(x["verdict"] == "good"), -x["score"]))
    print("\n=== ranking ===")
    for r in summary["results"]:
        vis = ""
        if r.get("layout_score") is not None:
            vis = f"  layout={r['layout_score']} cond={r['conditions_score']}"
        print(f"  {r['verdict']:9s}  {r['score']:.3f}  {r['camera_id']}  "
              f"({r['candidates_per_hour']}/h, ratio={r['motion_ratio']}){vis}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"summary": summary, "reports": reports}, f, indent=2)
        print(f"wrote {out}")


def cmd_screen(args):
    """Fetch one still per named camera and run Gemini layout judge."""
    if not args.vision:
        print("note: screen is vision-first; enabling --vision")
        args.vision = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for query in args.queries:
        cam = resolve_camera(query)
        slug = slugify(cam["name"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        still = out_dir / f"{slug}_{stamp}.jpg"
        try:
            data = http_get(f"{cam['imageUrl']}?t={int(time.time() * 1000)}")
            still.write_bytes(data)
        except Exception as e:
            print(f"FAIL {cam['name']}: {e}")
            continue
        vision = score_vision([str(still)], n_samples=1)
        layout = (vision.get("metrics") or {}).get("layout_score")
        cond = (vision.get("metrics") or {}).get("conditions_score")
        facing = (vision.get("metrics") or {}).get("facing")
        score = vision.get("score")
        verdict = "good" if (layout or 0) >= 0.72 and not vision.get("hard_fails") else (
            "marginal" if (layout or 0) >= 0.5 else "poor"
        )
        if vision.get("hard_fails"):
            verdict = "poor"
        report = {
            "name": cam["name"],
            "camera_id": slug,
            "verdict": verdict,
            "score": score,
            "frames_dir": str(still),
            "vision": vision,
            "geometry": {"available": False},
            "signals": {},
            "image": {},
            "reasons": [
                *[f"[fail:vision] {h}" for h in vision.get("hard_fails") or []],
                *[f"[note:vision] {n}" for n in vision.get("notes") or []],
            ],
        }
        print_report(report)
        results.append({
            "name": cam["name"],
            "verdict": verdict,
            "score": score,
            "layout_score": layout,
            "conditions_score": cond,
            "facing": facing,
            "why": (vision.get("metrics") or {}).get("why"),
            "still": str(still),
        })

    results.sort(key=lambda r: (-(r["verdict"] == "good"), -(r.get("layout_score") or 0)))
    print("\n=== screen ranking (by layout) ===")
    for r in results:
        print(f"  {r['verdict']:9s}  layout={r.get('layout_score')}  "
              f"cond={r.get('conditions_score')}  {r['name']}  ({r.get('facing')})")
    out = Path(args.out) if args.out else out_dir / "screen.json"
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_vision_flag(p):
        p.add_argument(
            "--vision", action="store_true",
            help="run Gemini 3.6 Flash layout/conditions judge on sample frames",
        )

    p = sub.add_parser("score", help="score an existing frame dump")
    p.add_argument("frames", help="captures/SESSION folder")
    p.add_argument("--mask", help="mask.json or camera/*.json with staging_zone")
    p.add_argument("--camera", help="camera/*.json for geometry scoring")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    p.add_argument("--out", help="write full JSON report")
    add_vision_flag(p)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("probe", help="capture a short dump from NYC TMC then score")
    p.add_argument("query", help='camera name, e.g. "Park Ave @ 23 St"')
    p.add_argument("--minutes", type=float, default=6.0)
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    p.add_argument("--out-dir", default=str(ROOT / "captures"))
    add_vision_flag(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("batch", help="score all annotated cameras with captures")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    p.add_argument("--out", default=str(ROOT / "reports" / "camera_scores.json"))
    add_vision_flag(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("screen", help="Gemini still-only screen of named NYC cameras")
    p.add_argument("queries", nargs="+", help="camera name queries")
    p.add_argument("--out-dir", default=str(ROOT / "camera" / "_preview" / "screen"))
    p.add_argument("--out", help="write ranking JSON")
    add_vision_flag(p)
    p.set_defaults(func=cmd_screen, vision=True)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
