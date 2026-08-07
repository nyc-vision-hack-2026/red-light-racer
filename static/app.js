/**
 * Red Light Racer — client state machine
 * WAITING → PROMPT → PENDING → REVEAL → (WAITING | GAMEOVER)
 *
 * Frame URLs are opaque: never construct from round id / index.
 */

(() => {
  "use strict";

  const TARGET_ROUNDS = 10;
  const GUESS_WINDOW_MS = 10000;
  const BOX_COLORS = {
    A: "#3ec8ff",
    B: "#ffb020",
    C: "#9b7bff",
    D: "#ff4fd8",
  };

  const state = {
    phase: "BOOT",
    sessionId: null,
    round: null,
    roundNum: 0,
    score: 0,
    streak: 0,
    bestStreak: 0,
    pickTrackId: null,
    pickLabel: null,
    guessStartedAt: 0,
    reveal: null,
    pollTimer: null,
    animTimer: null,
    countdownTimer: null,
    waitingShownAt: 0,
    images: [],
    revealImages: [],
    frameIndex: 0,
    naturalW: 352,
    naturalH: 240,
  };

  const $ = (id) => document.getElementById(id);
  const views = {
    boot: $("view-boot"),
    waiting: $("view-waiting"),
    play: $("view-play"),
    gameover: $("view-gameover"),
  };

  const img = $("frame-img");
  const canvas = $("overlay");
  const ctx = canvas.getContext("2d");

  function showView(name) {
    Object.values(views).forEach((v) => v.classList.remove("active"));
    views[name].classList.add("active");
  }

  function updateHud() {
    $("hud-score").textContent = String(state.score);
    $("hud-streak").textContent = String(state.streak);
    $("hud-round").textContent =
      state.roundNum > 0 ? `${state.roundNum}/${TARGET_ROUNDS}` : "–";
  }

  async function api(path, opts) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(opts?.headers || {}) },
      ...opts,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${path}: ${text}`);
    }
    return res.json();
  }

  function clearTimers() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    if (state.animTimer) {
      clearTimeout(state.animTimer);
      cancelAnimationFrame(state.animTimer);
    }
    if (state.countdownTimer) clearInterval(state.countdownTimer);
    state.pollTimer = state.animTimer = state.countdownTimer = null;
  }

  function preload(urls) {
    return Promise.all(
      urls.map(
        (src) =>
          new Promise((resolve, reject) => {
            const im = new Image();
            im.onload = () => resolve(im);
            im.onerror = () => reject(new Error("img " + src));
            im.src = src;
          })
      )
    );
  }

  function syncCanvasSize() {
    const rect = img.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function scaleBox(box) {
    const rect = img.getBoundingClientRect();
    const sx = rect.width / state.naturalW;
    const sy = rect.height / state.naturalH;
    return {
      x: box.x * sx,
      y: box.y * sy,
      w: box.w * sx,
      h: box.h * sy,
    };
  }

  function boxAtFrame(cand, frameAbs) {
    let best = null;
    for (const b of cand.boxes) {
      if (b.frame <= frameAbs) best = b;
      if (b.frame === frameAbs) break;
    }
    return best;
  }

  function hitExpand(box, minPx) {
    const s = scaleBox(box);
    const padX = Math.max(0, (minPx - s.w) / 2);
    const padY = Math.max(0, (minPx - s.h) / 2);
    return {
      x: s.x - padX,
      y: s.y - padY,
      w: s.w + padX * 2,
      h: s.h + padY * 2,
    };
  }

  function drawPromptOverlay(frameAbs, opts = {}) {
    const rect = img.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!state.round) return;

    for (const cand of state.round.candidates) {
      const box = boxAtFrame(cand, frameAbs);
      if (!box) continue;
      const s = scaleBox(box);
      const color = BOX_COLORS[cand.label] || "#fff";
      const selected = state.pickTrackId === cand.track_id;
      const pulse = opts.pulsePick && selected;

      ctx.save();
      if (pulse) {
        const t = (performance.now() / 250) % (Math.PI * 2);
        ctx.globalAlpha = 0.65 + 0.35 * Math.sin(t);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = selected ? 3.5 : 2;
      ctx.strokeRect(s.x, s.y, s.w, s.h);

      // label badge
      const label = cand.label;
      ctx.font = "700 12px 'IBM Plex Mono', monospace";
      const tw = ctx.measureText(label).width + 10;
      const th = 18;
      const lx = s.x;
      const ly = Math.max(0, s.y - th - 2);
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly, tw, th);
      ctx.fillStyle = "#0a0c10";
      ctx.fillText(label, lx + 5, ly + 13);
      ctx.restore();
    }
  }

  function drawRevealOverlay(frameAbs, finishOrder) {
    const rect = img.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!state.reveal) return;

    // finish line
    const fl = state.reveal.finish_line;
    if (fl && fl.length >= 2) {
      const sx = rect.width / state.naturalW;
      const sy = rect.height / state.naturalH;
      ctx.save();
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = "#b8f000";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(fl[0][0] * sx, fl[0][1] * sy);
      ctx.lineTo(fl[1][0] * sx, fl[1][1] * sy);
      ctx.stroke();
      ctx.restore();
    }

    const ffi = state.reveal.finish_frame_index || {};
    for (const cand of state.reveal.candidates) {
      const finishAt = ffi[String(cand.track_id)];
      const isDnf = finishAt === null || finishAt === undefined;
      const finished = !isDnf && frameAbs >= Number(finishAt);
      const box = boxAtFrame(cand, frameAbs);
      if (!box) continue;

      // fade DNF after they should be gone (approx: last box + a bit)
      let alpha = 1;
      if (isDnf) {
        const last = cand.boxes[cand.boxes.length - 1];
        if (last && frameAbs > last.frame) alpha = 0.25;
      }

      const s = scaleBox(box);
      const color = BOX_COLORS[cand.label] || "#fff";
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = finished ? "#b8f000" : color;
      ctx.lineWidth = finished ? 3.5 : 2;
      ctx.strokeRect(s.x, s.y, s.w, s.h);

      if (finished && finishOrder[cand.track_id]) {
        const place = finishOrder[cand.track_id];
        const text = place === 1 ? "1st" : place === 2 ? "2nd" : place === 3 ? "3rd" : `${place}th`;
        ctx.font = "700 11px 'IBM Plex Mono', monospace";
        ctx.fillStyle = "#b8f000";
        ctx.fillText(text, s.x, s.y - 4);
      } else {
        ctx.font = "700 11px 'IBM Plex Mono', monospace";
        ctx.fillStyle = color;
        ctx.fillText(cand.label, s.x, s.y - 4);
      }
      ctx.restore();
    }
  }

  function placeOrderMap(ffi) {
    const entries = Object.entries(ffi)
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => [Number(k), Number(v)])
      .sort((a, b) => a[1] - b[1]);
    const map = {};
    entries.forEach(([tid], i) => {
      map[tid] = i + 1;
    });
    return map;
  }

  // --- phases ---

  async function startSession() {
    clearTimers();
    state.score = 0;
    state.streak = 0;
    state.bestStreak = 0;
    state.roundNum = 0;
    state.pickTrackId = null;
    updateHud();
    const data = await api("/api/session", { method: "POST", body: "{}" });
    state.sessionId = data.session_id;
    enterWaiting();
  }

  function enterWaiting() {
    state.phase = "WAITING";
    showView("waiting");
    $("waiting-msg").classList.remove("show");
    state.waitingShownAt = performance.now();
    const showPulse = setTimeout(() => {
      if (state.phase === "WAITING") $("waiting-msg").classList.add("show");
    }, 400);

    pollNextRound(showPulse);
  }

  async function pollNextRound(pulseTimer) {
    try {
      const data = await api(`/api/session/${state.sessionId}/next-round`);
      if (data.status === "waiting") {
        state.pollTimer = setTimeout(
          () => pollNextRound(pulseTimer),
          data.retry_after_ms || 2000
        );
        return;
      }
      if (data.status === "session_complete") {
        clearTimeout(pulseTimer);
        enterGameover();
        return;
      }
      clearTimeout(pulseTimer);
      state.round = data.round;
      state.roundNum += 1;
      updateHud();
      await enterPrompt();
    } catch (err) {
      console.error(err);
      state.pollTimer = setTimeout(() => pollNextRound(pulseTimer), 2000);
    }
  }

  async function enterPrompt() {
    state.phase = "PROMPT";
    state.pickTrackId = null;
    state.pickLabel = null;
    showView("play");
    $("result-bar").hidden = true;
    $("void-card").hidden = true;
    $("pending-banner").hidden = true;
    $("green-flash").hidden = true;
    $("prompt-hint").hidden = false;
    $("countdown").hidden = true;
    $("spinner").hidden = false;

    const urls = state.round.frames;
    try {
      state.images = await preload(urls);
    } catch (e) {
      console.error(e);
      state.images = [];
    }
    $("spinner").hidden = true;

    if (state.images.length) {
      state.naturalW = state.images[0].naturalWidth || 352;
      state.naturalH = state.images[0].naturalHeight || 240;
    }

    // Play prompt at 2× real speed. Real fps is ~0.5 → 2s/frame; 2× → 1s/frame.
    const realFps = state.round.fps || 0.5;
    const frameMs = (1000 / realFps) / 2;
    const green = state.round.green_index;
    state.frameIndex = 0;

    const playLoop = () => {
      if (state.phase !== "PROMPT") return;
      const im = state.images[state.frameIndex];
      if (im) {
        img.src = im.src;
        img.onload = () => {
          syncCanvasSize();
          drawPromptOverlay(state.frameIndex);
        };
        // if already cached
        if (img.complete) {
          syncCanvasSize();
          drawPromptOverlay(state.frameIndex);
        }
      }

      if (state.frameIndex >= green) {
        // GREEN LIGHT flash, then hold and open guessing
        $("green-flash").hidden = false;
        setTimeout(() => {
          $("green-flash").hidden = true;
        }, 500);
        startGuessWindow();
        return;
      }
      state.frameIndex += 1;
      state.animTimer = setTimeout(playLoop, frameMs);
    };
    playLoop();
  }

  function startGuessWindow() {
    $("countdown").hidden = false;
    state.guessStartedAt = performance.now();
    const ring = $("ring-fg");
    const circ = 97.4;
    const tick = () => {
      if (state.phase !== "PROMPT") return;
      const elapsed = performance.now() - state.guessStartedAt;
      const left = Math.max(0, GUESS_WINDOW_MS - elapsed);
      $("countdown-num").textContent = String(Math.ceil(left / 1000));
      ring.style.strokeDashoffset = String(circ * (1 - left / GUESS_WINDOW_MS));
      if (left <= 0) {
        submitGuess(null);
      }
    };
    tick();
    state.countdownTimer = setInterval(tick, 100);
  }

  function onCanvasPointer(ev) {
    if (state.phase !== "PROMPT" || !state.round) return;
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX ?? ev.touches?.[0]?.clientX) - rect.left;
    const y = (ev.clientY ?? ev.touches?.[0]?.clientY) - rect.top;
    const frameAbs = state.round.green_index;

    // hit-test largest area first (expanded targets)
    let hit = null;
    for (const cand of state.round.candidates) {
      const box = boxAtFrame(cand, frameAbs);
      if (!box) continue;
      const h = hitExpand(box, 44);
      if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) {
        hit = cand;
      }
    }
    if (hit) submitGuess(hit.track_id, hit.label);
  }

  async function submitGuess(trackId, label) {
    if (state.phase !== "PROMPT") return;
    clearTimers();
    state.pickTrackId = trackId;
    state.pickLabel = label || null;
    const elapsed = Math.round(performance.now() - state.guessStartedAt);
    $("countdown").hidden = true;
    $("prompt-hint").hidden = true;
    $("green-flash").hidden = true;

    state.phase = "PENDING";
    $("pending-banner").hidden = false;
    // lock pick visual
    const pulse = () => {
      if (state.phase !== "PENDING") return;
      syncCanvasSize();
      drawPromptOverlay(state.round.green_index, { pulsePick: true });
      state.animTimer = requestAnimationFrame(pulse);
    };
    pulse();

    try {
      await api(`/api/round/${state.round.id}/guess`, {
        method: "POST",
        body: JSON.stringify({
          session_id: state.sessionId,
          track_id: trackId,
          elapsed_ms: elapsed,
          streak: state.streak,
        }),
      });
      pollResolution();
    } catch (err) {
      console.error(err);
      // recover
      enterWaiting();
    }
  }

  async function pollResolution() {
    try {
      const data = await api(
        `/api/round/${state.round.id}/resolution?session_id=${encodeURIComponent(state.sessionId)}`
      );
      if (data.status === "pending") {
        state.pollTimer = setTimeout(
          () => pollResolution(),
          data.retry_after_ms || 0
        );
        return;
      }
      if (data.status === "void") {
        cancelAnimationFrame(state.animTimer);
        $("pending-banner").hidden = true;
        $("void-card").hidden = false;
        setTimeout(() => enterWaiting(), 1600);
        return;
      }
      // resolved
      cancelAnimationFrame(state.animTimer);
      state.reveal = data.reveal;
      const pts = data.points || 0;
      state.score += pts;
      if (data.correct) {
        state.streak += 1;
        state.bestStreak = Math.max(state.bestStreak, state.streak);
      } else {
        state.streak = 0;
      }
      updateHud();
      await enterReveal(data);
    } catch (err) {
      console.error(err);
      state.pollTimer = setTimeout(pollResolution, 1500);
    }
  }

  async function enterReveal(resolution) {
    state.phase = "REVEAL";
    $("pending-banner").hidden = true;
    $("result-bar").hidden = true;

    const urls = state.reveal.frames.slice();
    state.revealImages = await preload(urls).catch(() => []);

    // background poll if frames still arriving
    if (!state.reveal.frames_complete) {
      const keepPolling = async () => {
        if (state.phase !== "REVEAL") return;
        try {
          const data = await api(
            `/api/round/${state.round.id}/resolution?session_id=${encodeURIComponent(state.sessionId)}`
          );
          if (data.status === "resolved" && data.reveal) {
            const prev = state.reveal.frames.length;
            state.reveal = data.reveal;
            if (data.reveal.frames.length > prev) {
              const newUrls = data.reveal.frames.slice(prev);
              const more = await preload(newUrls).catch(() => []);
              state.revealImages = state.revealImages.concat(more);
            }
            if (!data.reveal.frames_complete) {
              state.pollTimer = setTimeout(
                keepPolling,
                data.retry_after_ms || 2000
              );
            }
          }
        } catch (_) {
          state.pollTimer = setTimeout(keepPolling, 2000);
        }
      };
      state.pollTimer = setTimeout(keepPolling, resolution.retry_after_ms || 2000);
    }

    const realFps = state.round.fps || 0.5;
    const frameMs = (1000 / realFps) / 1.5; // 1.5×
    const green = state.round.green_index;
    let localIdx = 0; // index into reveal.frames array
    const finishOrder = placeOrderMap(state.reveal.finish_frame_index || {});

    const play = () => {
      if (state.phase !== "REVEAL") return;
      const absFrame = green + localIdx;
      const im = state.revealImages[localIdx];
      if (im) {
        img.src = im.src;
        const paint = () => {
          syncCanvasSize();
          drawRevealOverlay(absFrame, finishOrder);
        };
        img.onload = paint;
        if (img.complete) paint();
      }

      const maxIdx = state.revealImages.length - 1;
      if (localIdx >= maxIdx) {
        if (state.reveal.frames_complete) {
          showResult(resolution);
          return;
        }
        // wait for more frames
        state.animTimer = setTimeout(play, frameMs);
        return;
      }
      localIdx += 1;
      state.animTimer = setTimeout(play, frameMs);
    };
    play();
  }

  function showResult(resolution) {
    $("result-bar").hidden = false;
    const verdict = $("result-verdict");
    if (resolution.correct) {
      verdict.textContent = "CLEAN TAKE";
      verdict.className = "result-verdict win";
    } else {
      verdict.textContent =
        state.pickTrackId == null
          ? "TOO SLOW"
          : `WRONG — ${resolution.winner_label} WINS`;
      verdict.className = "result-verdict lose";
    }
    $("result-points").innerHTML = `+<strong>${resolution.points}</strong> · total ${state.score}`;
  }

  async function enterGameover() {
    state.phase = "GAMEOVER";
    clearTimers();
    showView("gameover");
    $("final-score").textContent = String(state.score);
    $("best-streak").textContent = String(state.bestStreak);
    $("rank-msg").hidden = true;
    $("initials-form").hidden = false;
    await refreshLeaderboard(null);
  }

  async function refreshLeaderboard(highlightInitials) {
    try {
      const data = await api("/api/leaderboard?limit=20");
      const tbody = $("leaderboard").querySelector("tbody");
      tbody.innerHTML = "";
      (data.entries || []).forEach((e, i) => {
        const tr = document.createElement("tr");
        if (highlightInitials && e.initials === highlightInitials) tr.className = "me";
        tr.innerHTML = `<td>${i + 1}</td><td>${e.initials}</td><td>${e.score}</td>`;
        tbody.appendChild(tr);
      });
    } catch (err) {
      console.error(err);
    }
  }

  // --- wire UI ---
  $("btn-start").addEventListener("click", () => startSession());
  $("btn-again").addEventListener("click", () => {
    showView("boot");
    state.phase = "BOOT";
  });
  $("btn-next").addEventListener("click", () => enterWaiting());

  canvas.addEventListener("pointerup", onCanvasPointer);

  $("initials-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const raw = $("initials").value.trim().toUpperCase();
    if (!/^[A-Z]{3}$/.test(raw)) return;
    try {
      const data = await api("/api/score", {
        method: "POST",
        body: JSON.stringify({ initials: raw, score: state.score }),
      });
      $("initials-form").hidden = true;
      $("rank-msg").hidden = false;
      $("rank-msg").textContent = `RANK #${data.rank}`;
      await refreshLeaderboard(raw);
    } catch (err) {
      console.error(err);
    }
  });

  $("initials").addEventListener("input", (ev) => {
    ev.target.value = ev.target.value.toUpperCase().replace(/[^A-Z]/g, "").slice(0, 3);
  });

  window.addEventListener("resize", () => {
    if (!img.src) return;
    syncCanvasSize();
    if (state.phase === "PROMPT" || state.phase === "PENDING") {
      drawPromptOverlay(state.round?.green_index ?? 0, {
        pulsePick: state.phase === "PENDING",
      });
    }
  });
})();
