#!/usr/bin/env python3
"""Capture NYC TMC webcam frames every N seconds into a timestamped folder."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

CAMERAS_URL = "https://webcams.nyctmc.org/api/cameras"
DEFAULT_CAMERA = "Park Ave @ 23 St"
DEFAULT_INTERVAL = 2.0
USER_AGENT = "Mozilla/5.0 (compatible; nyc-webcam-capture/1.0)"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "camera"


def http_get(url: str, timeout: float = 15.0) -> bytes:
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


def make_session_dir(base: Path, camera_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"{slugify(camera_name)}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def fetch_image(url: str, timeout: float = 15.0) -> bytes:
    # Cache-bust so we get a fresh frame rather than a stale CDN hit.
    return http_get(f"{url}?t={int(time.time() * 1000)}", timeout=timeout)


def capture_loop(
    camera_name: str,
    image_url: str,
    out_dir: Path,
    interval: float,
    max_frames: int | None,
) -> None:
    print(f"Camera:  {camera_name}")
    print(f"Source:  {image_url}")
    print(f"Saving:  {out_dir.resolve()}")
    print(f"Every:   {interval:g}s  (Ctrl+C to stop)")
    print()

    count = 0
    failures = 0
    while max_frames is None or count < max_frames:
        started = time.monotonic()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = out_dir / f"frame_{stamp}.jpg"
        try:
            data = fetch_image(image_url)
            if len(data) < 1000:
                raise ValueError(f"response too small ({len(data)} bytes)")
            path.write_bytes(data)
            count += 1
            failures = 0
            print(f"[{count:05d}] saved {path.name} ({len(data)} bytes)", flush=True)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            failures += 1
            print(f"[warn] fetch failed ({failures}): {exc}", file=sys.stderr)
            if failures >= 10:
                print("Too many consecutive failures; exiting.", file=sys.stderr)
                sys.exit(1)

        elapsed = time.monotonic() - started
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save NYC TMC webcam images into a timestamped folder."
    )
    p.add_argument(
        "-c",
        "--camera",
        default=DEFAULT_CAMERA,
        help=f'Camera name (default: "{DEFAULT_CAMERA}")',
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("captures"),
        help="Parent directory for the timestamped session folder (default: ./captures)",
    )
    p.add_argument(
        "-i",
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between captures (default: {DEFAULT_INTERVAL:g})",
    )
    p.add_argument(
        "-n",
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (default: run until Ctrl+C)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        print("Interval must be > 0", file=sys.stderr)
        sys.exit(2)

    cam = resolve_camera(args.camera)
    camera_name = cam["name"]
    image_url = cam.get("imageUrl") or f"{CAMERAS_URL}/{cam['id']}/image"
    if str(cam.get("isOnline")).lower() != "true":
        print(f"[warn] camera reports offline: {camera_name}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = make_session_dir(args.output_dir, camera_name)

    try:
        capture_loop(camera_name, image_url, session_dir, args.interval, args.max_frames)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
