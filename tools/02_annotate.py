#!/usr/bin/env python3
"""Local canvas tool to draw staging_zone, finish_line, travel_direction.

Serves a tiny page. Keyboard: 1 staging, 2 finish, 3 direction, s save, u undo, c clear.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "camera" / "cam_01.json"

HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>Annotate camera</title>
<style>
  body { margin:0; background:#111; color:#eee; font:14px monospace; }
  #wrap { display:flex; gap:16px; padding:12px; align-items:flex-start; }
  canvas { background:#000; cursor:crosshair; max-width:90vw; }
  .panel { max-width:280px; line-height:1.5; }
  button { margin:4px 0; display:block; width:100%; padding:8px; }
  .mode { color:#ffb020; }
</style></head><body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div class="panel">
    <p>Mode: <span class="mode" id="mode">staging</span></p>
    <p>1 staging polygon · 2 finish line (2 clicks) · 3 direction drag · s save · u undo · c clear</p>
    <button id="save">Save</button>
    <pre id="out"></pre>
  </div>
</div>
<script>
const FRAME = "__FRAME_URL__";
const OUT = "__OUT_PATH__";
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const img = new Image();
let mode = 'staging';
let staging = [], finish = [], dir = null;
let dragStart = null;

img.onload = () => {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  redraw();
};
img.src = FRAME;

function redraw() {
  ctx.drawImage(img, 0, 0);
  // staging
  if (staging.length) {
    ctx.strokeStyle = '#3ec8ff'; ctx.lineWidth = 2; ctx.beginPath();
    staging.forEach((p,i) => i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
    if (staging.length > 2) ctx.closePath();
    ctx.stroke();
  }
  // finish
  if (finish.length) {
    ctx.strokeStyle = '#b8f000'; ctx.setLineDash([6,4]); ctx.beginPath();
    finish.forEach((p,i) => i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
    ctx.stroke(); ctx.setLineDash([]);
  }
  // direction
  if (dir) {
    ctx.strokeStyle = '#ff4fd8'; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(dir[0], dir[1]); ctx.lineTo(dir[2], dir[3]); ctx.stroke();
  }
  document.getElementById('mode').textContent = mode;
  document.getElementById('out').textContent = JSON.stringify(payload(), null, 2);
}

function payload() {
  let travel = [1, 0];
  if (dir) {
    const dx = dir[2]-dir[0], dy = dir[3]-dir[1];
    const n = Math.hypot(dx,dy) || 1;
    travel = [dx/n, dy/n];
  }
  return {
    camera_id: 'cam_01',
    frame_width: canvas.width,
    frame_height: canvas.height,
    staging_zone: staging,
    finish_line: finish,
    travel_direction: travel,
  };
}

canvas.addEventListener('pointerdown', e => {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * (canvas.width / r.width);
  const y = (e.clientY - r.top) * (canvas.height / r.height);
  if (mode === 'staging') staging.push([Math.round(x), Math.round(y)]);
  else if (mode === 'finish') {
    if (finish.length >= 2) finish = [];
    finish.push([Math.round(x), Math.round(y)]);
  } else if (mode === 'direction') {
    dragStart = [x, y];
  }
  redraw();
});
canvas.addEventListener('pointerup', e => {
  if (mode !== 'direction' || !dragStart) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * (canvas.width / r.width);
  const y = (e.clientY - r.top) * (canvas.height / r.height);
  dir = [Math.round(dragStart[0]), Math.round(dragStart[1]), Math.round(x), Math.round(y)];
  dragStart = null;
  redraw();
});

window.addEventListener('keydown', e => {
  if (e.key === '1') mode = 'staging';
  if (e.key === '2') mode = 'finish';
  if (e.key === '3') mode = 'direction';
  if (e.key === 'u') {
    if (mode === 'staging') staging.pop();
    if (mode === 'finish') finish.pop();
    if (mode === 'direction') dir = null;
  }
  if (e.key === 'c') { staging=[]; finish=[]; dir=null; }
  if (e.key === 's') save();
  redraw();
});

async function save() {
  const body = payload();
  const res = await fetch('/save', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  alert(await res.text());
}
document.getElementById('save').onclick = save;
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame", type=Path, help="Sample frame to annotate")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not args.frame.exists():
        raise SystemExit(f"Frame not found: {args.frame}")

    frame_bytes = args.frame.read_bytes()
    frame_ct = "image/jpeg" if args.frame.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    out_path = args.output

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # quiet
            pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                page = (
                    HTML.replace("__FRAME_URL__", "/frame")
                    .replace("__OUT_PATH__", str(out_path))
                )
                data = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/frame":
                self.send_response(200)
                self.send_header("Content-Type", frame_ct)
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(body, indent=2) + "\n")
            msg = f"Saved {out_path}"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg.encode())

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Annotate at {url}")
    print(f"Will save to {out_path}")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
