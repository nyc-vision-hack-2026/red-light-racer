#!/usr/bin/env python3
"""Pull frames from an NYC TMC (or similar) camera into data/raw/<session>/."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CAMERAS_URL = "https://webcams.nyctmc.org/api/cameras"
DEFAULT_CAMERA = "1 Ave @ 110 St"
DEFAULT_INTERVAL = 2.0  # ~0.5 fps
USER_AGENT = "Mozilla/5.0 (compatible; redlight-racer/1.0)"
ROOT = Path(__file__).resolve().parent.parent


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


def capture_loop(
    camera_name: str,
    image_url: str,
    out_dir: Path,
    interval: float,
    max_frames: int | None,
) -> None:
    sidecar = out_dir / "timestamps.json"
    stamps: list[dict] = []
    print(f"Camera:  {camera_name}")
    print(f"Source:  {image_url}")
    print(f"Saving:  {out_dir.resolve()}")
    print(f"Every:   {interval:g}s")

    count = 0
    failures = 0
    while max_frames is None or count < max_frames:
        started = time.monotonic()
        name = f"{count:04d}.jpg"
        path = out_dir / name
        try:
            data = http_get(f"{image_url}?t={int(time.time() * 1000)}")
            if len(data) < 1000:
                raise ValueError(f"response too small ({len(data)} bytes)")
            path.write_bytes(data)
            ts = datetime.now(timezone.utc).isoformat()
            stamps.append({"frame": name, "ts": ts})
            sidecar.write_text(json.dumps({"camera": camera_name, "frames": stamps}, indent=2))
            count += 1
            failures = 0
            print(f"[{count:05d}] {name} ({len(data)} bytes)", flush=True)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            failures += 1
            print(f"[warn] fetch failed ({failures}): {exc}", file=sys.stderr)
            if failures >= 10:
                sys.exit(1)

        sleep_for = interval - (time.monotonic() - started)
        if sleep_for > 0:
            time.sleep(sleep_for)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--camera", default=DEFAULT_CAMERA)
    p.add_argument("-o", "--output-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("-i", "--interval", type=float, default=DEFAULT_INTERVAL)
    p.add_argument("-n", "--max-frames", type=int, default=None)
    args = p.parse_args()

    cam = resolve_camera(args.camera)
    image_url = cam.get("imageUrl") or f"{CAMERAS_URL}/{cam['id']}/image"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"{slugify(cam['name'])}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    try:
        capture_loop(cam["name"], image_url, out, args.interval, args.max_frames)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
