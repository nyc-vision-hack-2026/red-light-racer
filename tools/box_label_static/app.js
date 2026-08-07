(() => {
  const COLORS = [
    "#e8a317", "#3d9a5f", "#5b8def", "#c44b2f",
    "#c77ae0", "#2ec4b6", "#ff6b35", "#a8dadc",
  ];

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const windowsEl = document.getElementById("windows");
  const trackList = document.getElementById("track-list");
  const meta = document.getElementById("meta");
  const hud = document.getElementById("hud");
  const scrubber = document.getElementById("scrubber");
  const frameLabel = document.getElementById("frame-label");
  const status = document.getElementById("status");

  const state = {
    windows: [],
    tracks: {},
    nextTrackId: 1,
    win: 0,
    local: 0,
    activeTrack: null,
    img: null,
    drag: null, // {mode:'new'|'move'|'resize', startX, startY, orig, handle?}
    hoverHandle: null,
  };

  function colorFor(tid) {
    return COLORS[(Number(tid) - 1) % COLORS.length];
  }

  function currentWindow() {
    return state.windows[state.win];
  }

  function globalFrame() {
    const w = currentWindow();
    return w ? w.frame_indices[state.local] : null;
  }

  function boxAt(tid, frame) {
    const t = state.tracks[String(tid)];
    if (!t) return null;
    return t.keyframes[String(frame)] || null;
  }

  /** Interpolated box for display when no keyframe on this frame. */
  function displayBox(tid, frame) {
    const t = state.tracks[String(tid)];
    if (!t) return null;
    const kfs = Object.keys(t.keyframes).map(Number).sort((a, b) => a - b);
    if (!kfs.length) return null;
    if (t.keyframes[String(frame)]) {
      return { ...t.keyframes[String(frame)], keyframe: true };
    }
    let prev = null;
    let next = null;
    for (const f of kfs) {
      if (f < frame) prev = f;
      if (f > frame && next === null) next = f;
    }
    if (prev !== null && next !== null) {
      const a = t.keyframes[String(prev)];
      const b = t.keyframes[String(next)];
      const u = (frame - prev) / (next - prev);
      return {
        x: a.x + u * (b.x - a.x),
        y: a.y + u * (b.y - a.y),
        w: a.w + u * (b.w - a.w),
        h: a.h + u * (b.h - a.h),
        keyframe: false,
      };
    }
    if (prev !== null) return { ...t.keyframes[String(prev)], keyframe: false };
    if (next !== null) return { ...t.keyframes[String(next)], keyframe: false };
    return null;
  }

  function canvasPoint(e) {
    const r = canvas.getBoundingClientRect();
    const sx = canvas.width / r.width;
    const sy = canvas.height / r.height;
    return {
      x: (e.clientX - r.left) * sx,
      y: (e.clientY - r.top) * sy,
    };
  }

  function handles(box) {
    const { x, y, w, h } = box;
    return {
      nw: { x, y },
      ne: { x: x + w, y },
      sw: { x, y: y + h },
      se: { x: x + w, y: y + h },
    };
  }

  function hitHandle(box, p, tol = 8) {
    const hs = handles(box);
    for (const [name, q] of Object.entries(hs)) {
      if (Math.hypot(p.x - q.x, p.y - q.y) <= tol) return name;
    }
    return null;
  }

  function hitBox(p) {
    const frame = globalFrame();
    const ids = Object.keys(state.tracks).map(Number).sort((a, b) => a - b);
    // prefer active track, then topmost (highest id)
    const order = state.activeTrack
      ? [state.activeTrack, ...ids.filter((t) => t !== state.activeTrack)].reverse()
      : ids.slice().reverse();
    for (const tid of order) {
      const b = displayBox(tid, frame);
      if (!b) continue;
      const h = hitHandle(b, p);
      if (h) return { tid, box: b, handle: h };
      if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) {
        return { tid, box: b, handle: null };
      }
    }
    return null;
  }

  function draw() {
    if (!state.img) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.img, 0, 0);
    const frame = globalFrame();
    for (const [tidStr, track] of Object.entries(state.tracks)) {
      const tid = Number(tidStr);
      const b = displayBox(tid, frame);
      if (!b) continue;
      const col = colorFor(tid);
      const active = tid === state.activeTrack;
      ctx.save();
      ctx.strokeStyle = col;
      ctx.lineWidth = active ? 2.5 : 1.5;
      if (!b.keyframe) ctx.setLineDash([5, 4]);
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.font = "600 11px IBM Plex Mono, monospace";
      const tag = `${track.label || tid}${b.keyframe ? "" : " ~"}`;
      const tw = ctx.measureText(tag).width + 6;
      ctx.fillRect(b.x, Math.max(0, b.y - 14), tw, 14);
      ctx.fillStyle = "#12100e";
      ctx.fillText(tag, b.x + 3, Math.max(10, b.y - 3));
      if (active && b.keyframe) {
        ctx.fillStyle = col;
        for (const q of Object.values(handles(b))) {
          ctx.fillRect(q.x - 3, q.y - 3, 6, 6);
        }
      }
      ctx.restore();
    }
    if (state.drag && state.drag.mode === "new" && state.drag.curr) {
      const { startX, startY, curr } = state.drag;
      const x = Math.min(startX, curr.x);
      const y = Math.min(startY, curr.y);
      const w = Math.abs(curr.x - startX);
      const h = Math.abs(curr.y - startY);
      ctx.strokeStyle = colorFor(state.activeTrack || state.nextTrackId);
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }
  }

  function setStatus(msg) {
    status.textContent = msg || "";
  }

  async function api(path, opts) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(opts?.headers || {}) },
      ...opts,
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  }

  async function upsertKeyframe(tid, frame, box) {
    const data = await api("/api/keyframe", {
      method: "POST",
      body: JSON.stringify({ track_id: tid, frame, ...box }),
    });
    state.tracks = data.tracks;
    renderTracks();
    draw();
  }

  function renderWindows() {
    windowsEl.innerHTML = "";
    state.windows.forEach((w, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip" + (i === state.win ? " active" : "");
      const g = w.green_index_global != null ? `g${w.green_index_global}` : "all";
      btn.textContent = `#${i + 1} · ${g} · ${w.frame_indices.length}f`;
      btn.onclick = () => showWindow(i);
      windowsEl.appendChild(btn);
    });
  }

  function renderTracks() {
    trackList.innerHTML = "";
    const ids = Object.keys(state.tracks).map(Number).sort((a, b) => a - b);
    if (!ids.length) {
      trackList.innerHTML = `<p style="color:var(--muted);font-size:0.7rem;margin:0">No tracks yet — drag on the frame.</p>`;
      return;
    }
    for (const tid of ids) {
      const t = state.tracks[String(tid)];
      const n = Object.keys(t.keyframes || {}).length;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "track" + (tid === state.activeTrack ? " active" : "");
      btn.innerHTML = `<span class="swatch" style="background:${colorFor(tid)}"></span>` +
        `<span>${t.label || tid} · id ${tid} · ${n} kf</span>`;
      btn.onclick = () => {
        state.activeTrack = tid;
        renderTracks();
        draw();
      };
      trackList.appendChild(btn);
    }
  }

  function showWindow(i) {
    state.win = i;
    state.local = 0;
    renderWindows();
    scrubber.max = String(Math.max(0, currentWindow().frame_indices.length - 1));
    scrubber.value = "0";
    loadFrame();
  }

  function loadFrame() {
    const w = currentWindow();
    if (!w) return;
    const url = w.frame_urls[state.local];
    const g = w.frame_indices[state.local];
    const img = new Image();
    img.onload = () => {
      state.img = img;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const green = w.green_index_global;
      const tag = green != null
        ? (g < green ? "queued" : g === green ? "GREEN" : "reveal")
        : "frame";
      hud.textContent = `${tag} · local ${state.local + 1}/${w.frame_indices.length}`;
      frameLabel.textContent = `global ${g}` + (green != null ? ` · green ${green}` : "");
      scrubber.value = String(state.local);
      draw();
    };
    img.src = url;
  }

  function step(delta) {
    const w = currentWindow();
    if (!w) return;
    state.local = Math.max(0, Math.min(w.frame_indices.length - 1, state.local + delta));
    loadFrame();
  }

  async function ensureActiveTrack() {
    if (state.activeTrack && state.tracks[String(state.activeTrack)]) {
      return state.activeTrack;
    }
    const data = await api("/api/track", { method: "POST", body: "{}" });
    state.tracks = data.tracks;
    state.nextTrackId = data.track_id + 1;
    state.activeTrack = data.track_id;
    renderTracks();
    return data.track_id;
  }

  async function newTrack() {
    const data = await api("/api/track", { method: "POST", body: "{}" });
    state.tracks = data.tracks;
    state.nextTrackId = data.track_id + 1;
    state.activeTrack = data.track_id;
    renderTracks();
    setStatus(`track ${data.label} (id ${data.track_id})`);
    draw();
  }

  async function copyPrev() {
    const w = currentWindow();
    if (!w || state.local <= 0) return;
    const prevG = w.frame_indices[state.local - 1];
    const curG = w.frame_indices[state.local];
    let n = 0;
    for (const [tidStr, track] of Object.entries(state.tracks)) {
      const src = track.keyframes[String(prevG)] || displayBox(Number(tidStr), prevG);
      if (!src) continue;
      await upsertKeyframe(Number(tidStr), curG, {
        x: src.x, y: src.y, w: src.w, h: src.h,
      });
      n += 1;
    }
    setStatus(n ? `copied ${n} box(es) from prev frame` : "nothing to copy");
  }

  async function deleteCurrent() {
    if (!state.activeTrack) return;
    const frame = globalFrame();
    const data = await api("/api/delete", {
      method: "POST",
      body: JSON.stringify({ track_id: state.activeTrack, frame }),
    });
    state.tracks = data.tracks;
    renderTracks();
    draw();
    setStatus(`deleted box on frame ${frame}`);
  }

  async function exportTracks() {
    try {
      const data = await api("/api/export", { method: "POST", body: "{}" });
      setStatus(`exported ${data.detections} dets → ${data.path}`);
    } catch (err) {
      setStatus(`export failed: ${err.message}`);
    }
  }

  canvas.addEventListener("pointerdown", async (e) => {
    if (!state.img) return;
    canvas.setPointerCapture(e.pointerId);
    const p = canvasPoint(e);
    const hit = hitBox(p);
    if (hit) {
      state.activeTrack = hit.tid;
      renderTracks();
      const box = { ...hit.box };
      if (hit.handle) {
        state.drag = {
          mode: "resize",
          handle: hit.handle,
          startX: p.x,
          startY: p.y,
          orig: box,
        };
      } else {
        state.drag = {
          mode: "move",
          startX: p.x,
          startY: p.y,
          orig: box,
        };
      }
      draw();
      return;
    }
    const tid = await ensureActiveTrack();
    state.drag = {
      mode: "new",
      trackId: tid,
      startX: p.x,
      startY: p.y,
      curr: p,
    };
    draw();
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!state.drag) return;
    const p = canvasPoint(e);
    if (state.drag.mode === "new") {
      state.drag.curr = p;
      draw();
      return;
    }
    const o = state.drag.orig;
    if (state.drag.mode === "move") {
      const dx = p.x - state.drag.startX;
      const dy = p.y - state.drag.startY;
      state.drag.currBox = {
        x: o.x + dx,
        y: o.y + dy,
        w: o.w,
        h: o.h,
      };
      // temporary paint via mutating display — store on drag and draw overlay
      const frame = globalFrame();
      const tid = state.activeTrack;
      if (!state.tracks[String(tid)].keyframes[String(frame)]) {
        // keep interpolated look while dragging by writing temp
      }
      draw();
      // redraw with temp box
      ctx.save();
      ctx.strokeStyle = colorFor(tid);
      ctx.lineWidth = 2.5;
      ctx.strokeRect(state.drag.currBox.x, state.drag.currBox.y, state.drag.currBox.w, state.drag.currBox.h);
      ctx.restore();
      return;
    }
    if (state.drag.mode === "resize") {
      let { x, y, w, h } = o;
      const hx = state.drag.handle;
      const x2 = x + w;
      const y2 = y + h;
      let nx = x, ny = y, nx2 = x2, ny2 = y2;
      if (hx.includes("w")) nx = p.x;
      if (hx.includes("e")) nx2 = p.x;
      if (hx.includes("n")) ny = p.y;
      if (hx.includes("s")) ny2 = p.y;
      state.drag.currBox = {
        x: Math.min(nx, nx2),
        y: Math.min(ny, ny2),
        w: Math.abs(nx2 - nx),
        h: Math.abs(ny2 - ny),
      };
      draw();
      ctx.save();
      ctx.strokeStyle = colorFor(state.activeTrack);
      ctx.lineWidth = 2.5;
      const b = state.drag.currBox;
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.restore();
    }
  });

  canvas.addEventListener("pointerup", async (e) => {
    if (!state.drag) return;
    const drag = state.drag;
    state.drag = null;
    const frame = globalFrame();
    if (drag.mode === "new") {
      const p = canvasPoint(e);
      const x = Math.min(drag.startX, p.x);
      const y = Math.min(drag.startY, p.y);
      const w = Math.abs(p.x - drag.startX);
      const h = Math.abs(p.y - drag.startY);
      if (w < 4 || h < 4) {
        draw();
        return;
      }
      await upsertKeyframe(drag.trackId, frame, { x, y, w, h });
      setStatus(`keyframe track ${drag.trackId} @ ${frame}`);
      return;
    }
    if (drag.currBox && drag.currBox.w >= 4 && drag.currBox.h >= 4) {
      await upsertKeyframe(state.activeTrack, frame, drag.currBox);
      setStatus(`${drag.mode} track ${state.activeTrack} @ ${frame}`);
    } else {
      draw();
    }
  });

  document.getElementById("prev").onclick = () => step(-1);
  document.getElementById("next").onclick = () => step(1);
  scrubber.addEventListener("input", () => {
    state.local = Number(scrubber.value);
    loadFrame();
  });
  document.getElementById("new-track").onclick = () => newTrack();
  document.getElementById("copy-prev").onclick = () => copyPrev();
  document.getElementById("export").onclick = () => exportTracks();

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea")) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    if (e.key === "[") showWindow((state.win - 1 + state.windows.length) % state.windows.length);
    if (e.key === "]") showWindow((state.win + 1) % state.windows.length);
    if (e.key === "n") newTrack();
    if (e.key === "c") copyPrev();
    if (e.key === "e") exportTracks();
    if (e.key === "Backspace" || e.key === "Delete") {
      e.preventDefault();
      deleteCurrent();
    }
    if (/^[1-9]$/.test(e.key)) {
      const ids = Object.keys(state.tracks).map(Number).sort((a, b) => a - b);
      const tid = ids[Number(e.key) - 1];
      if (tid != null) {
        state.activeTrack = tid;
        renderTracks();
        draw();
      }
    }
  });

  fetch("/api/meta")
    .then((r) => r.json())
    .then((data) => {
      state.windows = data.windows;
      state.tracks = data.tracks || {};
      state.nextTrackId = data.next_track_id || 1;
      const ids = Object.keys(state.tracks).map(Number);
      state.activeTrack = ids.length ? Math.min(...ids) : null;
      meta.textContent = `${data.frame_count} frames · ${data.out_path}`;
      renderTracks();
      if (state.windows.length) showWindow(0);
      else setStatus("no windows");
    })
    .catch((err) => {
      meta.textContent = `failed: ${err.message}`;
    });
})();
