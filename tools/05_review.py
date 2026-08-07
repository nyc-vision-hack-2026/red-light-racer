#!/usr/bin/env python3
"""Human verification pass for candidate rounds. Keep / Drop each one.

Mandatory before promoting to data/rounds.json.
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

HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Review rounds</title>
<style>
  body { margin:0; background:#0a0c10; color:#e8ecf4; font:14px 'IBM Plex Mono', monospace; }
  #ui { max-width:720px; margin:0 auto; padding:16px; }
  canvas { width:100%; background:#000; border:1px solid #333; }
  .row { display:flex; gap:8px; margin-top:12px; }
  button { flex:1; padding:14px; font:700 14px monospace; cursor:pointer; border:0; }
  #keep { background:#3dffb5; color:#041; }
  #drop { background:#ff5a3c; color:#200; }
  #meta { color:#8b93a7; margin:8px 0; }
</style></head><body>
<div id="ui">
  <h1>REVIEW</h1>
  <div id="meta"></div>
  <canvas id="c"></canvas>
  <div class="row">
    <button id="drop">DROP</button>
    <button id="keep">KEEP</button>
  </div>
</div>
<script>
let rounds = [];
let idx = 0;
let kept = [];
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const colors = {A:'#3ec8ff',B:'#ffb020',C:'#9b7bff',D:'#ff4fd8'};

async function init() {
  rounds = await (await fetch('/rounds')).json();
  if (!rounds.length) { document.getElementById('meta').textContent = 'No rounds'; return; }
  play();
}

async function play() {
  if (idx >= rounds.length) {
    await fetch('/done', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ids: kept}) });
    document.getElementById('meta').textContent = `Done. Kept ${kept.length}. Saved.`;
    return;
  }
  const r = rounds[idx];
  document.getElementById('meta').textContent = `${idx+1}/${rounds.length}  ${r.id}  winner=${r.winner_track_id}  green=${r.green_index}`;
  canvas.width = 704; canvas.height = 480;
  for (let f = 0; f < r.frames.length; f++) {
    await drawFrame(r, f);
    await sleep(f <= r.green_index ? 400 : 280);
  }
}

function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }

function drawFrame(r, f) {
  return new Promise(resolve => {
    const im = new Image();
    im.onload = () => {
      const sx = canvas.width / im.naturalWidth;
      const sy = canvas.height / im.naturalHeight;
      ctx.drawImage(im, 0, 0, canvas.width, canvas.height);
      // finish line
      const fl = r.finish_line;
      ctx.setLineDash([8,6]); ctx.strokeStyle = '#b8f000'; ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(fl[0][0]*sx, fl[0][1]*sy);
      ctx.lineTo(fl[1][0]*sx, fl[1][1]*sy);
      ctx.stroke(); ctx.setLineDash([]);
      for (const c of r.candidates) {
        const b = c.boxes.find(bb => bb.frame === f) || c.boxes.filter(bb => bb.frame <= f).pop();
        if (!b) continue;
        ctx.strokeStyle = colors[c.label] || '#fff'; ctx.lineWidth = 2;
        ctx.strokeRect(b.x*sx, b.y*sy, b.w*sx, b.h*sy);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(c.label, b.x*sx, b.y*sy - 4);
      }
      if (f === r.green_index) {
        ctx.fillStyle = 'rgba(184,240,0,0.25)';
        ctx.fillRect(0,0,canvas.width,40);
        ctx.fillStyle = '#b8f000'; ctx.font = 'bold 20px monospace';
        ctx.fillText('GREEN', 12, 28);
      }
      resolve();
    };
    im.src = '/frame?path=' + encodeURIComponent(r.frames[f]);
  });
}

document.getElementById('keep').onclick = () => { kept.push(rounds[idx].id); idx++; play(); };
document.getElementById('drop').onclick = () => { idx++; play(); };
init();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "candidates",
        type=Path,
        nargs="?",
        default=ROOT / "data" / "rounds_candidates.json",
    )
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data" / "rounds.json")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    doc = json.loads(args.candidates.read_text())
    rounds = doc["rounds"]
    frames_root = ROOT / "data" / "frames"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                data = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(data)
            elif u.path == "/rounds":
                # Rewrite frame paths as relative for the review UI
                data = json.dumps(rounds).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
            elif u.path == "/frame":
                rel = parse_qs(u.query).get("path", [""])[0]
                path = (frames_root / rel).resolve()
                if not str(path).startswith(str(frames_root.resolve())) or not path.exists():
                    self.send_error(404)
                    return
                raw = path.read_bytes()
                ct = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.end_headers()
                self.wfile.write(raw)
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/done":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            ids = set(body.get("ids") or [])
            kept = [r for r in rounds if r["id"] in ids]
            out = {"version": 1, "camera_id": doc.get("camera_id", "cam_01"), "rounds": kept}
            args.output.write_text(json.dumps(out, indent=2) + "\n")
            msg = f"Wrote {args.output} with {len(kept)} rounds"
            print(msg)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg.encode())

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Review {len(rounds)} rounds at {url}")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
