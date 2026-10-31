// Video2URDF landing page: the before/after compare, showcase cards, the agent-log replay, the demo
// dialog and scroll reveals. The animated point field (bg.js) and the live measured laptop (scan3d.js)
// load after the page, only when needed. The viewer itself is index.html's inline module.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const finePointer = matchMedia("(hover: hover) and (pointer: fine)").matches;

/* ---------- copy buttons ---------- */
function initCopy() {
  for (const b of $$("[data-copy], [data-copy-visible]")) {
    b.addEventListener("click", async () => {
      const target = b.dataset.copy ? $(b.dataset.copy)
        : $$("pre.code", b.closest(".code-card")).find((p) => !p.hidden);
      const text = target ? target.textContent.trim() : "";
      try { await navigator.clipboard.writeText(text); } catch {
        const r = document.createRange(); r.selectNodeContents(target); getSelection().removeAllRanges(); getSelection().addRange(r);
      }
      const label = $("span", b);
      b.classList.add("done");
      if (label) label.textContent = "Copied";
      setTimeout(() => { b.classList.remove("done"); if (label) label.textContent = "Copy"; }, 1600);
    });
  }
}

/* ---------- before/after: the phone video and the measured model over the same frames ---------- */
const COMPARE = {
  laptop_lidar: 'lid <b class="m">309 × 205 mm</b> <span class="muted">spec 304 × 212</span>',
  drawer: 'pulls out <b class="m">248 mm</b> <span class="muted">hand fit 253</span>',
  microwave_lidar: 'door <b class="m">386 × 267 mm</b> <span class="muted">vertical hinge, opens 54°</span>',
  mouse_lidar: '<b class="m">121 × 75 × 55 mm</b> <span class="muted">spec 125 × 84 × 51</span>',
};

function initCompare() {
  const box = $("#compare");
  if (!box) return;
  const before = $(".cmp-before", box), after = $(".cmp-after", box);  // after leads, before follows its clock
  const handle = $(".cmp-handle", box), playBtn = $(".cmp-play", box);
  let pos = 50, touched = false, drag = null, want = !reduced, inView = false, raf = 0;

  function setPos(p) {
    pos = clamp(p, 0, 100);
    box.style.setProperty("--pos", pos.toFixed(2) + "%");
    handle.setAttribute("aria-valuenow", String(Math.round(pos)));
    handle.setAttribute("aria-valuetext", `${Math.round(pos)}% phone video`);
    box.classList.toggle("hide-left", pos < 24);
    box.classList.toggle("hide-right", pos > 76);
  }
  const pct = (e) => { const r = box.getBoundingClientRect(); return (e.clientX - r.left) / r.width * 100; };
  box.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".cmp-play") || (e.pointerType === "mouse" && e.button !== 0)) return;
    touched = true;
    drag = { id: e.pointerId, x: e.clientX, y: e.clientY, live: e.pointerType !== "touch" || !!e.target.closest(".cmp-handle") };
    if (drag.live) { box.setPointerCapture(e.pointerId); box.classList.add("dragging"); setPos(pct(e)); }
  });
  box.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    if (!drag.live) {  // a touch: follow it only once it moves sideways (an up/down swipe scrolls the page)
      const dx = Math.abs(e.clientX - drag.x), dy = Math.abs(e.clientY - drag.y);
      if (dx < 6 || dx < dy) return;
      drag.live = true; box.setPointerCapture(e.pointerId); box.classList.add("dragging");
    }
    setPos(pct(e));
  });
  const end = () => { drag = null; box.classList.remove("dragging"); };
  box.addEventListener("pointerup", end);
  box.addEventListener("pointercancel", end);
  handle.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 10 : 2;
    const k = { ArrowLeft: -step, ArrowDown: -step, ArrowRight: step, ArrowUp: step }[e.key];
    if (k !== undefined) { touched = true; setPos(pos + k); e.preventDefault(); }
    else if (e.key === "Home") { setPos(0); e.preventDefault(); }
    else if (e.key === "End") { setPos(100); e.preventDefault(); }
  });

  // Keep the two videos on one clock. The replay leads; each frame the phone video follows it: held while
  // the leader buffers, its rate nudged for small drift, a seek for large drift (a loop, a stall).
  for (const v of [before, after]) { v.loop = true; v.muted = true; }
  const live = () => want && inView && !document.hidden;
  function tryPlay() {
    if (!live()) return;
    after.play().catch(() => {});
    if (!raf) raf = requestAnimationFrame(sync);
    peek();
  }
  let resyncing = false;
  function resync() {  // big drift: hold the leader, seek the follower to it, then start both together
    resyncing = true;
    after.pause(); before.pause();
    before.addEventListener("seeked", () => {
      resyncing = false;
      if (live()) { after.play().catch(() => {}); before.play().catch(() => {}); }
    }, { once: true });
    before.currentTime = after.currentTime;
  }
  function sync() {
    raf = 0;
    if (!live()) return;
    if (!resyncing) {
      if (after.paused) after.play().catch(() => {});
      const leading = !after.paused && after.readyState >= 3;
      if (!leading) { if (!before.paused) before.pause(); }
      else if (before.paused) resync();
      else if (before.readyState >= 2) {
        const dur = after.duration || 1;  // both loop: measure drift around the loop point
        const d = ((before.currentTime - after.currentTime + dur * 1.5) % dur) - dur / 2;
        if (Math.abs(d) > 0.3) resync();
        else before.playbackRate = clamp(1 - d * 1.6, 0.85, 1.15);
      }
    }
    raf = requestAnimationFrame(sync);
  }
  function stop() { cancelAnimationFrame(raf); raf = 0; resyncing = false; before.pause(); after.pause(); }
  for (const v of [before, after]) v.addEventListener("loadeddata", () => {
    if (before.readyState >= 2 && after.readyState >= 2) box.classList.remove("loading");
  });
  new IntersectionObserver(([e]) => {
    inView = e.isIntersecting;
    if (inView) tryPlay(); else stop();
  }, { threshold: 0.2 }).observe(box);
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : tryPlay()));

  if (reduced) {  // no autoplay: a play button starts both
    playBtn.hidden = false;
    playBtn.addEventListener("click", () => { want = true; playBtn.hidden = true; tryPlay(); });
  }

  let peeked = false;  // once, show that the divider moves
  function peek() {
    if (peeked || touched || reduced) return;
    peeked = true;
    const keys = [[0, 50], [700, 30], [1500, 70], [2200, 50]];
    const t0 = performance.now();
    const ease = (x) => x < .5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2;
    (function step(now) {
      if (touched) return;
      const t = Math.max(0, now - t0);  // a frame's timestamp can precede t0
      const i = keys.findIndex(([k]) => k > t);
      if (i < 1) { setPos(50); return; }
      const [ta, pa] = keys[i - 1], [tb, pb] = keys[i];
      setPos(pa + (pb - pa) * ease((t - ta) / (tb - ta)));
      requestAnimationFrame(step);
    })(t0 + 0.001);
  }

  // object chips
  const chips = $$("[data-cmp]");
  for (const b of chips) b.addEventListener("click", () => select(b.dataset.cmp));
  function select(name) {
    for (const c of chips) c.setAttribute("aria-selected", String(c.dataset.cmp === name));
    box.classList.add("loading");
    box.classList.remove("sweep"); void box.offsetWidth; box.classList.add("sweep");
    stop();
    for (const [v, k] of [[before, "phone"], [after, "replay"]]) {
      v.poster = `assets/compare/${name}_${k}.jpg`;
      v.preload = "auto";
      v.src = `assets/compare/${name}_${k}.mp4`;
    }
    $("#cmpFact").innerHTML = COMPARE[name] || "";
    const open = $("#cmpOpen");
    open.href = "#" + name;
    open.dataset.model = name;
    tryPlay();
  }
  setPos(50);
}

/* ---------- showcase cards and result clips: play while hovered or in view ---------- */
function initClips() {
  const play = (card, v) => {
    if (!v.getAttribute("src")) v.src = v.dataset.src;
    v.play().then(() => card.classList.add("playing")).catch(() => {});
  };
  const pause = (card, v) => { v.pause(); card.classList.remove("playing"); };
  const cards = $$(".obj").map((card) => ({ card, v: $("video", card) })).filter((c) => c.v);
  if (finePointer) {
    for (const { card, v } of cards) {
      card.addEventListener("mouseenter", () => play(card, v));
      card.addEventListener("mouseleave", () => pause(card, v));
      card.addEventListener("focus", () => play(card, v));
      card.addEventListener("blur", () => pause(card, v));
    }
  } else if (!reduced) {
    const io = new IntersectionObserver((es) => {
      for (const e of es) {
        const c = cards.find((x) => x.card === e.target);
        if (e.intersectionRatio > 0.6) play(c.card, c.v); else pause(c.card, c.v);
      }
    }, { threshold: [0, 0.6, 1] });
    for (const { card } of cards) io.observe(card);
  }
  // results: robot and policy clips
  const clips = $$(".rc-img.vid video");
  if (!reduced) {
    const io = new IntersectionObserver((es) => {
      for (const e of es) (e.isIntersecting ? play(e.target.parentElement, e.target) : pause(e.target.parentElement, e.target));
    }, { threshold: 0.35 });
    for (const v of clips) io.observe(v);
  } else {
    for (const v of clips) { v.controls = true; v.src = v.dataset.src; v.preload = "none"; }
  }
}

/* ---------- the agent log, replayed ---------- */
// A real run, quoted from docs/models/notebook/agent/log.md (the side-view laptop), with gaps marked "…".
const LOG = [
  ["sys", "model muse-spark-1.3-contributor"],
  ["sys", "look [0, 136, 272, 407]"],
  ["spark", "Building your laptop's kinematic model — I'll isolate its base and lid to track the hinge."],
  ["call", "segment", '{"frame": 0, "prompt": "laptop keyboard"}'],
  ["out", "s1.1: box (376, 1231, 836, 1473), confidence 0.92, area 3.9% of frame"],
  ["call", "segment", '{"frame": 0, "prompt": "laptop lid"}'],
  ["out", "s2.0: box (644, 546, 937, 1481), confidence 0.97, area 8.5% of frame"],
  ["call", "segment", '{"frame": 0, "prompt": "green card"}'],
  ["out", "s3.0: box (344, 1566, 830, 1693), confidence 0.95, area 2.6% of frame"],
  ["spark", "The lid and keyboard segmented cleanly — now I'll set the metric scale and get the full machine base."],
  ["call", "measure_scale", '{"candidate": "s3.0"}'],
  ["out", "Scale set to 5680 px/m (from card s3.0: long side 486 px = 85.6 mm)."],
  ["gap", "…"],
  ["call", "track", '{"parts": [{"candidate": "s1.1", "name": "base"}, {"candidate": "s2.0", "name": "lid"}]}'],
  ["out", "Tracked: <w>base 319/408 frames</w>, lid 408/408 frames"],
  ["call", "fit", "{}"],
  ["out", "- base: root, replay IoU mean 0.70, <w>worst 0.00 (frame 222)</w>"],
  ["out", "- base -> lid: revolute at (21, -14) mm, <w>range -4.2..24.8 deg</w>, fit residual 0.7 mm, replay IoU mean 0.51, worst 0.10 (frame 161)"],
  ["spark", "The first fit missed the closing motion — the keyboard gets hidden when shut. I'll try more stable segmentations."],
  ["gap", "…"],
  ["call", "look", '{"frames": [0, 100, 136, 200, 272]}'],
  ["spark", "I see the problem — the lid shows its screen when open and its silver back when closed. I'll look for a consistent edge view."],
  ["gap", "…"],
  ["call", "segment", '{"frame": 200, "prompt": "laptop lid"}'],
  ["out", "s11.0: box (328, 772, 936, 1472), confidence 0.96, area 1.9% of frame"],
  ["call", "segment", '{"frame": 200, "prompt": "gray laptop base"}'],
  ["out", "s12.0: box (0, 1232, 921, 1531), confidence 0.70, area 6.4% of frame"],
  ["gap", "…"],
  ["spark", "Found a stable side view of the base and edge view of the lid — I'll track those through the full video."],
  ["call", "track", '{"parts": [{"candidate": "s12.0", "name": "base"}, {"candidate": "s11.0", "name": "lid"}]}'],
  ["out", "Tracked: <o>base 408/408 frames</o>, lid 408/408 frames"],
  ["call", "fit", "{}"],
  ["out", "- base: root, replay IoU mean 0.73, worst 0.46 (frame 141)"],
  ["out", "- base -> lid: revolute at (73, 2) mm, <o>range -3.9..73.3 deg</o>, fit residual 1.7 mm, replay IoU mean 0.63, worst 0.09 (frame 82)"],
  ["gap", "…"],
  ["call", "finish", '{"report": "Video notebook_2.MOV: laptop with 2 links, 1 revolute joint. …"}'],
  ["done", "done"],
];

function initTerminal() {
  const term = $("#term"), body = $("#termBody");
  if (!term) return;
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const marks = (s) => esc(s).replace(/&lt;w&gt;(.*?)&lt;\/w&gt;/g, '<span class="warn">$1</span>').replace(/&lt;o&gt;(.*?)&lt;\/o&gt;/g, '<span class="ok">$1</span>');
  let run = 0;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function line(kind) { const p = document.createElement("p"); p.className = "ln " + kind; body.append(p); return p; }
  async function type(el, html, cps) {  // types plain text, then swaps in the marked-up line
    const plain = html.replace(/<[^>]+>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
    const caret = document.createElement("span"); caret.className = "caret";
    const t = document.createTextNode("");
    el.append(t, caret);
    const id = run;
    for (let i = 0; i < plain.length; i += 2) {
      if (id !== run) return false;
      t.data = plain.slice(0, i + 2);
      await sleep(1000 / cps * 2);
    }
    caret.remove();
    el.innerHTML = html;
    return true;
  }
  async function play() {
    const id = ++run;
    body.innerHTML = "";
    term.classList.remove("done");
    for (const [kind, a, b] of LOG) {
      if (id !== run) return;
      if (reduced) { render(kind, a, b); continue; }
      if (kind === "call") {
        const el = line("call");
        if (!(await type(el, `<span class="fn">${esc(a)}</span> <span class="args">${esc(b)}</span>`, 95))) return;
        await sleep(380);
      } else if (kind === "spark") {
        await sleep(250);
        if (!(await type(line("spark"), esc(a), 120))) return;
        await sleep(420);
      } else {
        render(kind, a);
        await sleep(kind === "out" ? 170 : 260);
      }
      while (body.children.length > 60) body.firstChild.remove();
    }
    term.classList.add("done");
  }
  function render(kind, a, b) {
    const el = line(kind);
    if (kind === "call") el.innerHTML = `<span class="fn">${esc(a)}</span> <span class="args">${esc(b)}</span>`;
    else el.innerHTML = marks(a);
  }
  $(".term-replay", term).addEventListener("click", play);
  const io = new IntersectionObserver(([e]) => { if (e.intersectionRatio > 0.3) { io.disconnect(); play(); } }, { threshold: [0, 0.3, 0.6] });
  io.observe(term);
}

/* ---------- open-weights command tabs ---------- */
function initCodeTabs() {
  for (const card of $$(".code-card")) {
    const tabs = $$("[data-code]", card);
    for (const t of tabs) t.addEventListener("click", () => {
      for (const x of tabs) { x.setAttribute("aria-selected", String(x === t)); $("#" + x.dataset.code).hidden = x !== t; }
    });
  }
}

/* ---------- the demo video in a dialog ---------- */
function initDemo() {
  const dlg = $("#demoDialog"), v = $("#demoVideo");
  if (!dlg || !dlg.showModal) return;
  for (const a of $$("[data-demo]")) a.addEventListener("click", (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;
    e.preventDefault();
    if (!v.getAttribute("src")) v.src = "demo.mp4";
    dlg.showModal();
    v.play().catch(() => {});
  });
  const close = () => dlg.open && dlg.close();
  $(".dlg-close", dlg).addEventListener("click", close);
  dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });
  dlg.addEventListener("close", () => v.pause());
}

/* ---------- reveal on scroll ---------- */
function initReveal() {
  if (reduced || !("IntersectionObserver" in window)) return;
  const groups = [".steps-grid > .step", "#showcase .sec-head-row", ".cards > .obj", "#how .sec-head", ".pipeline > li", ".term-wrap",
    ".scan-card", ".measured-copy", "#results .sec-head", ".bento > .rc", "#models .sec-head", ".mgrid > .mc", ".ow-copy", ".code-card", ".cta",
    ".viewer-sec .sec-head-row"];
  const els = [];
  for (const sel of groups) $$(sel).forEach((el, i) => { el.classList.add("reveal"); el.style.transitionDelay = `${Math.min(i, 6) * 60}ms`; els.push(el); });
  document.documentElement.classList.add("reveal-on");
  const io = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { threshold: 0.08, rootMargin: "0px 0px -6% 0px" });
  els.forEach((el) => io.observe(el));
}

/* ---------- WebGL extras, loaded when they matter ---------- */
// No WebGL probe: a throwaway context costs as much as a real one. Without WebGL the renderer throws and the
// import's catch leaves the page as it is.
function initBackground() {
  const canvas = $("#bg");
  if (!canvas) return;
  const go = () => import("./bg.js").then((m) => m.start(canvas, { reduced, hero: $(".hero") })).catch((e) => console.warn("background", e));
  if (document.readyState === "complete") setTimeout(go, 150); else addEventListener("load", () => setTimeout(go, 150), { once: true });
}
function initScan() {
  const card = $("#scanCard");
  if (!card) return;
  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    import("./scan3d.js").then((m) => m.start({ card, reduced, finePointer })).catch((err) => console.warn("scan", err));
  }, { rootMargin: "500px 0px" });
  io.observe(card);
}

initCopy();
initCompare();
initClips();
initTerminal();
initCodeTabs();
initDemo();
initReveal();
initBackground();
initScan();
