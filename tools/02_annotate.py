#!/usr/bin/env python3
"""Draw camera geometry: staging_zone (start/queue), finish_line, travel_direction.

Not vehicle boxes — those come from Roboflow. This is the approach-lane polygon
and finish line the game / cut_rounds use.

Usage
-----
    # scan all captures/ and flip between cameras in the UI
    python tools/02_annotate.py

    # or start on one folder
    python tools/02_annotate.py captures/1_ave_110_st_20260807_182523

Click the tool buttons to switch what you draw. ←/→ cameras flip locations.
Edits auto-save into camera/<id>.json (git-friendly).
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
CAPTURES = ROOT / "captures"
CAMERA_DIR = ROOT / "camera"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def slug_from_name(name: str) -> str:
    return re.sub(r"_\d{8}_\d{6}$", "", name) or "cam"


def list_frames(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )


def discover_cameras(start: Path | None = None) -> list[dict]:
    """One entry per location slug, using the capture folder with the most frames."""
    by_slug: dict[str, dict] = {}

    folders: list[Path] = []
    if start and start.is_dir() and start.resolve() != CAPTURES.resolve():
        # include siblings under captures/ when started from one session
        parent = start.parent
        if parent.resolve() == CAPTURES.resolve() and CAPTURES.is_dir():
            folders = [p for p in CAPTURES.iterdir() if p.is_dir()]
        else:
            folders = [start]
    elif CAPTURES.is_dir():
        folders = [p for p in CAPTURES.iterdir() if p.is_dir()]

    for folder in folders:
        frames = list_frames(folder)
        if not frames:
            continue
        slug = slug_from_name(folder.name)
        # normalize park_ave_23_st / park_ave_23st
        slug = slug.replace("park_ave_23_st", "park_ave_23st")
        entry = {
            "camera_id": slug,
            "capture_dir": folder,
            "frame": frames[len(frames) // 2],
            "frame_count": len(frames),
        }
        prev = by_slug.get(slug)
        if prev is None or entry["frame_count"] > prev["frame_count"]:
            by_slug[slug] = entry

    cams = sorted(by_slug.values(), key=lambda c: c["camera_id"])
    if not cams and start:
        # single file fallback
        frame = start if start.is_file() else None
        if start.is_dir():
            frames = list_frames(start)
            frame = frames[len(frames) // 2] if frames else None
        if frame is None:
            raise SystemExit("no capture folders with images found")
        slug = slug_from_name(frame.parent.name)
        cams = [{
            "camera_id": slug,
            "capture_dir": frame.parent,
            "frame": frame,
            "frame_count": 1,
        }]
    if not cams:
        raise SystemExit(f"no captures found under {CAPTURES}")
    return cams


def out_path_for(camera_id: str, override: Path | None = None) -> Path:
    if override is not None:
        return override if override.is_absolute() else ROOT / override
    # prefer existing cam_01.json for park ave if present and no dedicated file yet
    dedicated = CAMERA_DIR / f"{camera_id}.json"
    if camera_id in {"park_ave_23st", "park_ave_23_st"} and not dedicated.is_file():
        legacy = CAMERA_DIR / "cam_01.json"
        if legacy.is_file():
            return legacy
    return dedicated


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_existing(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {}


HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>Annotate cameras</title>
<style>
  :root {
    --bg: #111; --panel: #1a1a1a; --line: #333; --text: #eee; --muted: #9a9a9a;
    --staging: #3ec8ff; --finish: #b8f000; --dir: #ff4fd8; --ok: #3d9a5f; --warn: #e8a317;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
  #wrap { display:flex; gap:16px; padding:12px; align-items:flex-start; flex-wrap:wrap; }
  canvas { background:#000; cursor:crosshair; max-width:min(92vw, 920px); image-rendering: pixelated; border:1px solid var(--line); }
  .panel { width:340px; max-width:92vw; }
  h1 { font:700 18px/1.2 system-ui,sans-serif; margin:0 0 4px; }
  .path { color:var(--muted); font-size:12px; margin:0 0 10px; word-break:break-all; }
  .cam-nav {
    display:grid; grid-template-columns: auto 1fr auto; gap:8px; align-items:center;
    margin-bottom:12px; padding:8px; background:var(--panel); border:1px solid var(--line);
  }
  .cam-nav button {
    appearance:none; cursor:pointer; width:2.4rem; height:2.4rem;
    background:#222; color:var(--text); border:1px solid var(--line); font:inherit; font-size:16px;
  }
  .cam-nav button:hover { border-color:#888; }
  .cam-nav .cam-meta { text-align:center; min-width:0; }
  .cam-nav .cam-id { font-weight:700; font-size:13px; word-break:break-all; }
  .cam-nav .cam-sub { color:var(--muted); font-size:11px; }
  .tools { display:flex; flex-direction:column; gap:8px; margin-bottom:12px; }
  .tool {
    appearance:none; text-align:left; cursor:pointer;
    background:var(--panel); color:var(--text); border:2px solid var(--line);
    padding:10px 12px; font:inherit;
  }
  .tool:hover { border-color:#666; }
  .tool.active { border-color: var(--accent, #fff); background:#222; }
  .tool .name { font-weight:700; display:block; margin-bottom:2px; }
  .tool .desc { color:var(--muted); font-size:12px; }
  .tool[data-mode="staging"] { --accent: var(--staging); }
  .tool[data-mode="finish"] { --accent: var(--finish); }
  .tool[data-mode="direction"] { --accent: var(--dir); }
  .checklist { list-style:none; padding:0; margin:0 0 12px; font-size:12px; }
  .checklist li { margin:4px 0; color:var(--muted); }
  .checklist li.done { color:var(--ok); }
  .checklist li.done::before { content:"✓ "; }
  .checklist li:not(.done)::before { content:"○ "; }
  .actions { display:flex; gap:8px; margin-bottom:8px; }
  .actions button {
    flex:1; appearance:none; cursor:pointer; padding:10px;
    background:#222; color:var(--text); border:1px solid var(--line); font:inherit;
  }
  .actions button.primary { background:var(--ok); border-color:var(--ok); color:#04140a; font-weight:700; }
  .status {
    min-height:2.4em; font-size:12px; color:var(--muted);
    padding:8px 10px; background:var(--panel); border:1px solid var(--line); margin-bottom:10px;
  }
  .status.saved { color:var(--ok); border-color:var(--ok); }
  .status.dirty { color:var(--warn); border-color:var(--warn); }
  .status.err { color:#ff6b6b; border-color:#ff6b6b; }
  .hint { color:var(--muted); font-size:12px; margin:0 0 10px; }
  pre { background:var(--panel); padding:8px; overflow:auto; font-size:11px; max-height:200px; border:1px solid var(--line); }
</style></head><body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div class="panel">
    <h1 id="title">…</h1>
    <p class="path">saves to <strong id="outpath">…</strong></p>

    <div class="cam-nav">
      <button type="button" id="prev-cam" title="Previous camera">←</button>
      <div class="cam-meta">
        <div class="cam-id" id="cam-label">—</div>
        <div class="cam-sub" id="cam-sub">—</div>
      </div>
      <button type="button" id="next-cam" title="Next camera">→</button>
    </div>

    <div class="tools" role="toolbar" aria-label="Drawing tools">
      <button type="button" class="tool active" data-mode="staging">
        <span class="name" style="color:var(--staging)">1 · Start / staging zone</span>
        <span class="desc">Click corners of the approach lanes where cars wait. Need ≥3 points.</span>
      </button>
      <button type="button" class="tool" data-mode="finish">
        <span class="name" style="color:var(--finish)">2 · Finish line</span>
        <span class="desc">Click exactly 2 points across the road further along travel.</span>
      </button>
      <button type="button" class="tool" data-mode="direction">
        <span class="name" style="color:var(--dir)">3 · Travel direction</span>
        <span class="desc">Click-drag an arrow showing which way cars leave the zone.</span>
      </button>
    </div>

    <p class="hint" id="mode-hint">Click on the image to add staging polygon points.</p>

    <ul class="checklist">
      <li id="check-staging">Start zone (0/3+ points)</li>
      <li id="check-finish">Finish line (0/2 points)</li>
      <li id="check-dir">Travel direction</li>
    </ul>

    <div class="actions">
      <button type="button" id="undo">Undo</button>
      <button type="button" id="clear-tool">Clear tool</button>
    </div>
    <div class="actions">
      <button type="button" class="primary" id="save">Save now</button>
    </div>
    <div class="status" id="status">Loading…</div>
    <pre id="out"></pre>
  </div>
</div>
<script>
const HINTS = {
  staging: 'Click on the image to add staging polygon points.',
  finish: 'Click two points to draw the finish line (third click restarts).',
  direction: 'Click and drag to draw the travel-direction arrow.',
};

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const hintEl = document.getElementById('mode-hint');
const img = new Image();

let cameras = [];
let camIndex = 0;
let mode = 'staging';
let staging = [];
let finish = [];
let dir = null;
let dragStart = null;
let dirty = false;
let saveTimer = null;
let current = null; // {camera_id, out_path, frame_count, capture_dir, had_file}

function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = 'status' + (kind ? ' ' + kind : '');
}

function setMode(next) {
  mode = next;
  document.querySelectorAll('.tool').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  hintEl.textContent = HINTS[mode];
  redraw();
}
document.querySelectorAll('.tool').forEach(btn => {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

function payload() {
  let travel = [1, 0];
  if (dir) {
    const dx = dir[2]-dir[0], dy = dir[3]-dir[1];
    const n = Math.hypot(dx,dy) || 1;
    travel = [+(dx/n).toFixed(4), +(dy/n).toFixed(4)];
  }
  return {
    camera_id: current?.camera_id || '',
    frame_width: canvas.width,
    frame_height: canvas.height,
    staging_zone: staging,
    finish_line: finish,
    travel_direction: travel,
  };
}

function updateChecklist() {
  const s = document.getElementById('check-staging');
  const f = document.getElementById('check-finish');
  const d = document.getElementById('check-dir');
  s.textContent = `Start zone (${staging.length}/3+ points)`;
  s.classList.toggle('done', staging.length >= 3);
  f.textContent = `Finish line (${finish.length}/2 points)`;
  f.classList.toggle('done', finish.length === 2);
  d.textContent = 'Travel direction';
  d.classList.toggle('done', !!dir);
}

function redraw() {
  if (!img.complete || !img.naturalWidth) return;
  ctx.drawImage(img, 0, 0);
  if (staging.length) {
    ctx.strokeStyle = '#3ec8ff'; ctx.lineWidth = 2; ctx.beginPath();
    staging.forEach((p,i) => i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
    if (staging.length > 2) ctx.closePath();
    ctx.stroke();
    staging.forEach(p => { ctx.fillStyle='#3ec8ff'; ctx.fillRect(p[0]-2,p[1]-2,4,4); });
  }
  if (finish.length) {
    ctx.strokeStyle = '#b8f000'; ctx.setLineDash([6,4]); ctx.beginPath();
    finish.forEach((p,i) => i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
    ctx.stroke(); ctx.setLineDash([]);
    finish.forEach(p => { ctx.fillStyle='#b8f000'; ctx.fillRect(p[0]-2,p[1]-2,4,4); });
  }
  if (dir) {
    ctx.strokeStyle = '#ff4fd8'; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(dir[0], dir[1]); ctx.lineTo(dir[2], dir[3]); ctx.stroke();
    const ang = Math.atan2(dir[3]-dir[1], dir[2]-dir[0]);
    ctx.beginPath();
    ctx.moveTo(dir[2], dir[3]);
    ctx.lineTo(dir[2]-10*Math.cos(ang-0.4), dir[3]-10*Math.sin(ang-0.4));
    ctx.lineTo(dir[2]-10*Math.cos(ang+0.4), dir[3]-10*Math.sin(ang+0.4));
    ctx.closePath(); ctx.fillStyle='#ff4fd8'; ctx.fill();
  }
  updateChecklist();
  document.getElementById('out').textContent = JSON.stringify(payload(), null, 2);
}

function applyExisting(existing) {
  staging = Array.isArray(existing.staging_zone) ? existing.staging_zone.map(p => [...p]) : [];
  finish = Array.isArray(existing.finish_line) ? existing.finish_line.map(p => [...p]) : [];
  dir = null;
  if (Array.isArray(existing.travel_direction) && existing.travel_direction.length === 2 && staging.length) {
    const cx = staging.reduce((s,p)=>s+p[0],0)/staging.length;
    const cy = staging.reduce((s,p)=>s+p[1],0)/staging.length;
    const [dx, dy] = existing.travel_direction;
    dir = [Math.round(cx), Math.round(cy), Math.round(cx+dx*60), Math.round(cy+dy*60)];
  }
}

function markDirty() {
  dirty = true;
  setStatus(`Unsaved changes… auto-saving to ${current.out_path}`, 'dirty');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => save(true), 400);
}

function canvasXY(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (canvas.width / r.width),
    y: (e.clientY - r.top) * (canvas.height / r.height),
  };
}

canvas.addEventListener('pointerdown', e => {
  const {x, y} = canvasXY(e);
  if (mode === 'staging') {
    staging.push([Math.round(x), Math.round(y)]);
    markDirty();
  } else if (mode === 'finish') {
    if (finish.length >= 2) finish = [];
    finish.push([Math.round(x), Math.round(y)]);
    markDirty();
  } else if (mode === 'direction') {
    dragStart = [x, y];
  }
  redraw();
});
canvas.addEventListener('pointerup', e => {
  if (mode !== 'direction' || !dragStart) return;
  const {x, y} = canvasXY(e);
  dir = [Math.round(dragStart[0]), Math.round(dragStart[1]), Math.round(x), Math.round(y)];
  dragStart = null;
  markDirty();
  redraw();
});

function undo() {
  if (mode === 'staging') staging.pop();
  else if (mode === 'finish') finish.pop();
  else if (mode === 'direction') dir = null;
  markDirty();
  redraw();
}
function clearTool() {
  if (mode === 'staging') staging = [];
  else if (mode === 'finish') finish = [];
  else if (mode === 'direction') dir = null;
  markDirty();
  redraw();
}

async function save(fromAuto) {
  if (!current) return;
  try {
    const res = await fetch('/api/save?id=' + encodeURIComponent(current.camera_id), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload()),
    });
    const text = await res.text();
    if (!res.ok) throw new Error(text);
    dirty = false;
    current.had_file = true;
    const ready = staging.length >= 3 && finish.length === 2 && !!dir;
    const note = ready ? 'complete — ready to commit' : 'draft saved';
    setStatus(`Saved ${current.out_path} · ${note}`, 'saved');
  } catch (err) {
    setStatus(`Save failed: ${err.message}`, 'err');
  }
}

async function loadCamera(i) {
  if (!cameras.length) return;
  if (dirty) {
    clearTimeout(saveTimer);
    await save(true);
  }
  camIndex = (i + cameras.length) % cameras.length;
  const summary = cameras[camIndex];
  setStatus(`Loading ${summary.camera_id}…`);
  const meta = await fetch('/api/camera?id=' + encodeURIComponent(summary.camera_id)).then(r => {
    if (!r.ok) throw new Error('failed to load camera');
    return r.json();
  });
  current = meta;
  document.getElementById('title').textContent = meta.camera_id;
  document.getElementById('outpath').textContent = meta.out_path;
  document.getElementById('cam-label').textContent =
    `${camIndex + 1}/${cameras.length} · ${meta.camera_id}`;
  document.getElementById('cam-sub').textContent =
    `${meta.frame_count} frames · ${meta.capture_dir}`;
  applyExisting(meta.existing || {});
  dirty = false;
  await new Promise((resolve, reject) => {
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      redraw();
      resolve();
    };
    img.onerror = () => reject(new Error('frame failed to load'));
    img.src = '/api/frame?id=' + encodeURIComponent(meta.camera_id) + '&t=' + Date.now();
  });
  setStatus(
    meta.had_file
      ? `Loaded existing ${meta.out_path}`
      : `New camera — draw & it will auto-save to ${meta.out_path}`,
    meta.had_file ? 'saved' : ''
  );
}

document.getElementById('undo').onclick = undo;
document.getElementById('clear-tool').onclick = clearTool;
document.getElementById('save').onclick = () => save(false);
document.getElementById('prev-cam').onclick = () => loadCamera(camIndex - 1);
document.getElementById('next-cam').onclick = () => loadCamera(camIndex + 1);

window.addEventListener('keydown', e => {
  if (e.target.matches('input, textarea')) return;
  if (e.key === '1') setMode('staging');
  if (e.key === '2') setMode('finish');
  if (e.key === '3') setMode('direction');
  if (e.key === 'u') undo();
  if (e.key === 's') { e.preventDefault(); save(false); }
  if (e.key === '[' || (e.key === 'ArrowLeft' && e.shiftKey)) loadCamera(camIndex - 1);
  if (e.key === ']' || (e.key === 'ArrowRight' && e.shiftKey)) loadCamera(camIndex + 1);
});

window.addEventListener('beforeunload', (e) => {
  if (dirty) { e.preventDefault(); e.returnValue = ''; }
});

fetch('/api/cameras')
  .then(r => r.json())
  .then(data => {
    cameras = data.cameras || [];
    if (!cameras.length) {
      setStatus('No cameras found under captures/', 'err');
      return;
    }
    const startId = data.start_id;
    let idx = cameras.findIndex(c => c.camera_id === startId);
    if (idx < 0) idx = 0;
    return loadCamera(idx);
  })
  .catch(err => setStatus(`Failed: ${err.message}`, 'err'));
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Optional capture folder to start on (default: scan captures/)",
    )
    ap.add_argument("--camera-id", default=None, help="Start on this camera id")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Override output JSON (single-camera only)")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    start = None
    if args.path:
        start = args.path if args.path.is_absolute() else ROOT / args.path
        if not start.exists():
            raise SystemExit(f"not found: {start}")

    cameras = discover_cameras(start)
    by_id = {c["camera_id"]: c for c in cameras}

    start_id = args.camera_id
    if start_id is None and start is not None:
        start_id = slug_from_name(start.name if start.is_dir() else start.parent.name)
        start_id = start_id.replace("park_ave_23_st", "park_ave_23st")
    if start_id not in by_id:
        start_id = cameras[0]["camera_id"]

    output_override = args.output
    if output_override and len(cameras) > 1:
        print("NOTE: -o ignored when multiple cameras are available; each saves to camera/<id>.json")

    page_bytes = HTML.encode()

    def camera_payload(cam: dict) -> dict:
        out = out_path_for(cam["camera_id"], output_override if len(cameras) == 1 else None)
        existing = load_existing(out)
        return {
            "camera_id": cam["camera_id"],
            "capture_dir": rel(cam["capture_dir"]),
            "frame_count": cam["frame_count"],
            "out_path": rel(out),
            "had_file": out.is_file(),
            "existing": existing,
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _json(self, code: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _text(self, code: int, msg: str):
            data = msg.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(page_bytes)))
                self.end_headers()
                self.wfile.write(page_bytes)
                return

            if path == "/api/cameras":
                self._json(200, {
                    "start_id": start_id,
                    "cameras": [
                        {
                            "camera_id": c["camera_id"],
                            "frame_count": c["frame_count"],
                            "capture_dir": rel(c["capture_dir"]),
                            "out_path": rel(out_path_for(c["camera_id"])),
                        }
                        for c in cameras
                    ],
                })
                return

            if path == "/api/camera":
                cid = (qs.get("id") or [None])[0]
                cam = by_id.get(cid or "")
                if not cam:
                    self._text(404, "unknown camera")
                    return
                self._json(200, camera_payload(cam))
                return

            if path == "/api/frame":
                cid = (qs.get("id") or [None])[0]
                cam = by_id.get(cid or "")
                if not cam:
                    self.send_error(404)
                    return
                frame = cam["frame"]
                raw = frame.read_bytes()
                ct = "image/jpeg" if frame.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self.send_error(404)
                return
            qs = parse_qs(parsed.query)
            cid = (qs.get("id") or [None])[0]
            cam = by_id.get(cid or "")
            if not cam:
                self._text(404, "unknown camera")
                return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            body["camera_id"] = cam["camera_id"]
            body["source_frame"] = rel(cam["frame"])
            out = out_path_for(cam["camera_id"], output_override if len(cameras) == 1 else None)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(body, indent=2) + "\n")
            msg = f"Saved {rel(out)}"
            print(msg, flush=True)
            self._text(200, msg)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Annotate at {url}")
    print(f"{len(cameras)} camera(s):")
    for c in cameras:
        mark = "←" if c["camera_id"] == start_id else " "
        print(f"  {mark} {c['camera_id']:24s}  {c['frame_count']:4d} frames  → {rel(out_path_for(c['camera_id']))}")
    print("UI: ←/→ buttons (or Shift+←/→, [/]) switch cameras · tool buttons switch draw mode")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
