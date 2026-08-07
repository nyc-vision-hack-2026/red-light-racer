(() => {
  const picker = document.getElementById("picker");
  const meta = document.getElementById("meta");
  const stage = document.getElementById("stage");
  const controls = document.getElementById("controls");
  const detail = document.getElementById("detail");
  const promptImg = document.getElementById("prompt-img");
  const revealImg = document.getElementById("reveal-img");
  const promptIdx = document.getElementById("prompt-idx");
  const revealIdx = document.getElementById("reveal-idx");
  const speed = document.getElementById("speed");

  let candidates = [];
  let index = 0;
  let promptTimer = null;
  let revealTimer = null;
  let promptI = 0;
  let revealI = 0;
  let intervalMs = Number(speed.value);

  function preload(urls) {
    urls.forEach((u) => {
      const im = new Image();
      im.src = u;
    });
  }

  function stopLoops() {
    clearInterval(promptTimer);
    clearInterval(revealTimer);
    promptTimer = revealTimer = null;
  }

  function tick(img, urls, getI, setI, label) {
    if (!urls.length) return;
    const i = getI() % urls.length;
    img.classList.remove("ready");
    img.onload = () => img.classList.add("ready");
    img.src = urls[i];
    label.textContent = `${i + 1}/${urls.length}`;
    setI(i + 1);
  }

  function startLoops(c) {
    stopLoops();
    promptI = 0;
    revealI = 0;
    preload(c.prompt_frames);
    preload(c.reveal_frames);

    const stepPrompt = () =>
      tick(promptImg, c.prompt_frames, () => promptI, (v) => { promptI = v; }, promptIdx);
    const stepReveal = () =>
      tick(revealImg, c.reveal_frames, () => revealI, (v) => { revealI = v; }, revealIdx);

    stepPrompt();
    stepReveal();
    promptTimer = setInterval(stepPrompt, intervalMs);
    revealTimer = setInterval(stepReveal, intervalMs);
  }

  function show(i) {
    if (!candidates.length) return;
    index = (i + candidates.length) % candidates.length;
    const c = candidates[index];

    [...picker.children].forEach((el, n) => {
      el.classList.toggle("active", n === index);
    });

    const redSec = (c.red_len_frames * 2).toFixed(0);
    detail.innerHTML =
      `<strong>#${index + 1}</strong> · green @ frame <strong>${c.green_index_global}</strong>` +
      ` · red held <strong>${c.red_len_frames}</strong> frames (~${redSec}s)` +
      ` · mot ${c.motion_at_green?.toFixed?.(1) ?? "—"}` +
      ` · occ ${c.occupancy_at_green?.toFixed?.(1) ?? "—"}`;

    stage.hidden = false;
    controls.hidden = false;
    startLoops(c);
  }

  function renderPicker() {
    picker.innerHTML = "";
    candidates.forEach((c, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.textContent = `#${i + 1} · g${c.green_index_global}`;
      btn.addEventListener("click", () => show(i));
      picker.appendChild(btn);
    });
  }

  document.getElementById("prev").addEventListener("click", () => show(index - 1));
  document.getElementById("next").addEventListener("click", () => show(index + 1));
  speed.addEventListener("input", () => {
    intervalMs = Number(speed.value);
    if (candidates[index]) startLoops(candidates[index]);
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") show(index - 1);
    if (e.key === "ArrowRight") show(index + 1);
  });

  fetch("/api/candidates")
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((data) => {
      candidates = data.candidates || [];
      meta.textContent = candidates.length
        ? `${candidates.length} windows · ${data.source}`
        : `no candidates in ${data.source}`;
      renderPicker();
      if (candidates.length) show(0);
    })
    .catch((err) => {
      meta.textContent = `failed to load: ${err.message}`;
    });
})();
