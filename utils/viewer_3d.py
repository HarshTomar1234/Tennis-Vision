"""
utils/viewer_3d.py
──────────────────
Generates a self-contained, interactive 3-D rally viewer as a single HTML file.

Why a hand-written renderer rather than three.js
------------------------------------------------
The viewer has to open by double-clicking a file: no server, no network, no build
step, and no CDN (which a strict content policy would block anyway). Inlining a
general 3-D engine costs hundreds of kilobytes to draw a court, a net and a handful
of parabolas. A direct perspective projection onto a 2-D canvas is a few hundred
lines, has no dependencies, and keeps the file small enough to email.

Evidence, and a mistake worth recording
---------------------------------------
The first version of this module emitted the SAME three evidence strings for every
segment, including "endpoints on the floor (bounce, or hitter's feet)". A review
checked them against the data: 14 of 16 segments had a non-floor endpoint (a contact
is 0.9-2.6 m up, by construction), and 6 of 16 ended outside the court lines. The
claims were false, in the one feature whose entire purpose is that claims are earned.

Evidence is now computed per segment from the actual reconstructed values — measured
flight duration, real endpoint heights, the landing position with an in/out verdict,
and net-crossing height. A viewer that asserts things the payload contradicts is worse
than one with no evidence at all, because it teaches the reader to distrust everything
else.

The generated page references the annotated video by relative path rather than
embedding it, so the two files must travel together. Base64-embedding a 30 MB video
would produce a 40 MB HTML file that most browsers handle badly.
"""
from __future__ import annotations

import json
from pathlib import Path

# Real tennis court dimensions, metres. The viewer draws the true court rather than
# the mini-court's pixel proxy, so what you orbit around is a tennis court.
COURT_LENGTH_M = 23.77
COURT_WIDTH_DOUBLES_M = 10.97
COURT_WIDTH_SINGLES_M = 8.23
SERVICE_LINE_FROM_NET_M = 6.40
NET_HEIGHT_CENTRE_M = 0.914
NET_HEIGHT_POST_M = 1.07
NET_POST_OUTSIDE_M = 0.914   # posts sit this far outside the doubles sidelines

# Colour of the court surface per playing surface, so a clip looks like the sport.
SURFACE_COLOURS = {
    "hard": "#1d4e6b", "clay": "#8c4a2f", "grass": "#2f6b3a", "unknown": "#12303a",
}


def _segment_evidence(t, fps: float) -> list[str]:
    """
    The measured facts behind one displayed segment.

    Every string here is derived from the segment's own numbers. Nothing is asserted
    that the payload does not contain — see the module docstring for why that matters.
    """
    duration = (t.end_frame - t.start_frame) / fps if fps else 0.0
    ex, ey, ez = t.end[0], t.end[1], t.end[2]
    sz = t.start[2]

    def describe(z: float) -> str:
        return "on the floor" if z <= 0.01 else f"{z:.1f} m up (racket contact)"

    inside = (0 <= ex <= COURT_WIDTH_DOUBLES_M) and (0 <= ey <= COURT_LENGTH_M)
    evidence = [
        f"flight {duration:.2f} s — within a single-flight window",
        f"starts {describe(sz)}, ends {describe(ez)}",
        f"ends at ({ex:.1f}, {ey:.1f}) m — "
        f"{'inside the court' if inside else 'outside the lines (ball out)'}",
    ]

    # Net crossing is the question the visualisation exists to answer, so state it.
    net_y = COURT_LENGTH_M / 2
    if (t.start[1] - net_y) * (t.end[1] - net_y) < 0:
        span = t.end[1] - t.start[1]
        fraction = (net_y - t.start[1]) / span if span else 0.5
        height = t.height_at(fraction)
        evidence.append(f"crosses the net at {height:.2f} m "
                        f"({'clears' if height > NET_HEIGHT_CENTRE_M else 'below'} "
                        f"the {NET_HEIGHT_CENTRE_M:.2f} m centre)")
    else:
        evidence.append("does not cross the net")
    return evidence


def _page_template() -> str:
    """The viewer's HTML/CSS/JS. `__DATA__` is replaced with the scene JSON."""
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tennis-Vision — 3D Rally Viewer</title>
<style>
  :root { --bg:#0e1116; --panel:#161b22; --line:#30363d; --text:#e6edf3;
          --muted:#8b949e; --accent:#4ec9b0; --active:#ffa657; --idle:#6f97ad; }
  * { box-sizing:border-box; }
  /* Flex column, not a magic pixel offset: the previous calc(100vh - 59px) ignored
     the header, so the page always overflowed and the scrubber sat below the fold. */
  html,body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--text); overflow:hidden;
         display:flex; flex-direction:column;
         font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header,footer { flex:none; }
  header { padding:10px 18px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0 12px 0 0; font-weight:600; }
  .sub { font-size:12px; color:var(--muted); }
  button { background:var(--panel); color:var(--text); border:1px solid var(--line);
           padding:6px 12px; border-radius:6px; cursor:pointer; font-size:13px; }
  button:hover:not(:disabled) { border-color:var(--accent); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.active { background:var(--accent); color:#04211c; border-color:var(--accent);
                  font-weight:600; }
  .sep { width:1px; height:22px; background:var(--line); }
  main { flex:1; min-height:0; display:grid; grid-template-columns:1fr 340px; }
  @media (max-width:900px){ main { grid-template-columns:1fr; grid-template-rows:55vh 1fr; } }
  #stage { position:relative; min-height:0; overflow:hidden; }
  canvas { display:block; width:100%; height:100%; cursor:grab; touch-action:none; }
  canvas:active { cursor:grabbing; }
  video { width:100%; height:100%; object-fit:contain; background:#000; }
  .hidden { display:none !important; }
  aside { border-left:1px solid var(--line); padding:14px; overflow:auto;
          background:var(--panel); min-height:0; }
  h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
       color:var(--muted); margin:0 0 8px; font-weight:600; }
  .legend { display:flex; gap:14px; font-size:12px; color:var(--muted);
            margin-bottom:12px; flex-wrap:wrap; }
  .sw { display:inline-block; width:16px; height:3px; vertical-align:middle;
        margin-right:5px; border-radius:2px; }
  .seg { display:block; width:100%; text-align:left; padding:9px 11px;
         border:1px solid var(--line); border-radius:6px; margin-bottom:6px;
         background:transparent; font-variant-numeric:tabular-nums; }
  .seg:hover { border-color:var(--accent); }
  .seg[aria-selected="true"] { border-color:var(--accent); background:#11221f; }
  .seg .top { display:flex; justify-content:space-between; align-items:baseline; }
  .seg .name { font-weight:600; }
  .seg .kmh { font-size:18px; font-weight:600; color:var(--accent); }
  .seg .sub2 { color:var(--muted); font-size:12px; }
  details { margin-top:5px; }
  summary { font-size:12px; color:var(--muted); cursor:pointer; }
  .ev { font-size:12px; color:var(--muted); margin:3px 0 0 10px; }
  .empty { color:var(--muted); font-size:13px; border:1px dashed var(--line);
           border-radius:8px; padding:18px; text-align:center; }
  footer { padding:8px 18px; border-top:1px solid var(--line);
           display:flex; align-items:center; gap:10px; }
  #scrubwrap { flex:1; position:relative; }
  input[type=range] { width:100%; display:block; }
  #ticks { position:relative; height:5px; margin-top:2px; }
  #ticks i { position:absolute; height:100%; background:var(--idle); border-radius:2px; }
  .note { color:var(--muted); font-size:12px; margin-top:14px;
          border-top:1px solid var(--line); padding-top:10px; }
  .warn { color:var(--active); font-weight:600; }
</style>
</head>
<body>
<header>
  <h1>Tennis-Vision · 3D Rally</h1>
  <button id="btn3d" class="active">3D view</button>
  <button id="btnVideo">Annotated video</button>
  <span class="sep"></span>
  <button data-view="broadcast">Broadcast</button>
  <button data-view="side">Side</button>
  <button data-view="top">Top</button>
  <button data-view="baseline">Baseline</button>
  <button id="reset">Reset</button>
  <span class="sub" id="hint">drag orbit · scroll zoom · arrows/space/R</span>
</header>

<main>
  <div id="stage">
    <canvas id="cv" role="img" aria-label="3D reconstruction of the rally: tennis court with ball flight arcs">
      Your browser does not support canvas; the 3D view cannot be shown.
    </canvas>
    <video id="vid" class="hidden" controls preload="none"></video>
  </div>
  <aside>
    <h2>Flight segments</h2>
    <div class="legend">
      <span><i class="sw" style="background:var(--accent)"></i>selected</span>
      <span><i class="sw" style="background:var(--active)"></i>playing now</span>
      <span><i class="sw" style="background:var(--idle)"></i>other flights</span>
    </div>
    <div id="list"></div>
    <div class="note" id="note"></div>
  </aside>
</main>

<footer>
  <button id="play">▶ Play</button>
  <select id="rate" title="playback speed">
    <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
    <option value="1" selected>1×</option>
  </select>
  <div id="scrubwrap">
    <input type="range" id="scrub" min="0" max="1000" value="0" aria-label="rally timeline">
    <div id="ticks"></div>
  </div>
  <span class="sub" id="tlabel">0.0 s</span>
</footer>

<script>
const DATA = __DATA__;
const L = DATA.court.length, W = DATA.court.width, WS = DATA.court.singles_width;
const SVC = DATA.court.service_from_net, NETC = DATA.court.net_centre,
      NETP = DATA.court.net_post, POST_OUT = DATA.court.post_outside;
const CX = W / 2, CY = L / 2, NETY = L / 2;

/* ---------- court geometry ---------- */
function courtLines() {
  const xm = (W - WS) / 2, xM = W - xm, seg = [];
  const add = (a, b) => seg.push([a, b]);
  add([0,0,0],[W,0,0]); add([0,L,0],[W,L,0]);
  add([0,0,0],[0,L,0]); add([W,0,0],[W,L,0]);
  add([xm,0,0],[xm,L,0]); add([xM,0,0],[xM,L,0]);
  add([xm,NETY-SVC,0],[xM,NETY-SVC,0]); add([xm,NETY+SVC,0],[xM,NETY+SVC,0]);
  add([W/2,NETY-SVC,0],[W/2,NETY+SVC,0]);
  // Baseline centre marks
  add([W/2,0,0],[W/2,0.3,0]); add([W/2,L,0],[W/2,L-0.3,0]);
  return seg;
}
const LINES = courtLines();

// Net tape as a catenary-ish curve with zero slope at the posts, so it does not kink
// where it meets them. Posts sit outside the doubles sidelines, as on a real court.
function netCurve() {
  const x0 = -POST_OUT, x1 = W + POST_OUT, pts = [];
  for (let i = 0; i <= 28; i++) {
    const t = i / 28;
    pts.push([x0 + (x1 - x0) * t, NETY, NETP + (NETC - NETP) * (1 - Math.cos(2*Math.PI*t)) / 2]);
  }
  return pts;
}
const NET = netCurve();

/* ---------- camera ---------- */
const VIEWS = {
  broadcast:{az:0.0, polar:0.62, dist:38}, side:{az:1.57, polar:0.95, dist:34},
  top:{az:0.0, polar:0.12, dist:34},       baseline:{az:0.0, polar:1.25, dist:30},
};
let cam = {...VIEWS.broadcast};
let basis = null;             // trig hoisted out of the per-point hot path
function updateBasis() {
  const ca = Math.cos(cam.az), sa = Math.sin(cam.az);
  const cp = Math.cos(cam.polar), sp = Math.sin(cam.polar);
  basis = {ca, sa, cp, sp};
}
let VW = 0, VH = 0, FSCALE = 0;

function project(p) {
  const {ca, sa, cp, sp} = basis;
  const x = p[0] - CX, y = p[1] - CY, z = p[2];
  const X = x * ca - y * sa, Y = x * sa + y * ca;
  const Z = Y * sp + z * cp, Yc = -Y * cp + z * sp;
  const depth = Z + cam.dist;
  if (depth <= 0.6) return null;
  return [VW/2 + FSCALE * X / depth, VH/2 - FSCALE * Yc / depth, depth];
}

/* ---------- state ---------- */
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let selected = -1, playing = false, tNorm = 0, rate = 1;
const TOTAL = DATA.segments.length ? Math.max(...DATA.segments.map(s => s.end_s)) : 1;

let pending = false;
function requestDraw() {                    // rAF throttle: pointer events fire far
  if (pending) return;                      // faster than the display refreshes, and a
  pending = true;                           // full redraw per event is what made the
  requestAnimationFrame(() => { pending = false; draw(); });   // drag feel sluggish.
}

function resize() {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  VW = Math.round(r.width); VH = Math.round(r.height);
  cv.width = Math.round(VW * dpr); cv.height = Math.round(VH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  FSCALE = 0.9 * Math.min(VW, VH);
  requestDraw();
}
addEventListener('resize', resize);

/* ---------- drawing ---------- */
function strokePath(points, colour, width, dash) {
  ctx.strokeStyle = colour; ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  let started = false, drew = false;
  for (const p of points) {
    const P = project(p);
    if (!P) { started = false; continue; }
    if (!started) { ctx.moveTo(P[0], P[1]); started = true; }
    else { ctx.lineTo(P[0], P[1]); drew = true; }
  }
  if (drew) ctx.stroke();
  ctx.setLineDash([]);
}

function meanDepth(points) {
  let sum = 0, n = 0;
  for (const p of points) { const P = project(p); if (P) { sum += P[2]; n++; } }
  return n ? sum / n : Infinity;
}

function draw() {
  updateBasis();
  ctx.clearRect(0, 0, VW, VH);
  ctx.lineJoin = ctx.lineCap = 'round';

  // Court surface
  const quad = [[0,0,0],[W,0,0],[W,L,0],[0,L,0]].map(project);
  if (quad.every(Boolean)) {
    ctx.fillStyle = DATA.surface_colour;
    ctx.beginPath(); ctx.moveTo(quad[0][0], quad[0][1]);
    quad.slice(1).forEach(c => ctx.lineTo(c[0], c[1]));
    ctx.closePath(); ctx.fill();
  }
  LINES.forEach(([a,b]) => strokePath([a,b], 'rgba(220,235,240,.85)', 1.5));

  // Ground shadow of every arc: the cheapest way to make height legible without
  // orbiting, because a flat arc against a flat court reads as no height at all.
  DATA.segments.forEach(s =>
    strokePath(s.points.map(p => [p[0], p[1], 0]), 'rgba(0,0,0,.35)', 1.2));

  // Depth-sort the arcs against the net so a ball behind the net renders behind it.
  // Drawing arcs last unconditionally made every ball appear in front of the tape —
  // and "did it clear the net" is the question this view exists to answer.
  const drawables = DATA.segments.map((s, i) => ({kind:'arc', i, s, d: meanDepth(s.points)}));
  drawables.push({kind:'net', d: meanDepth(NET)});
  drawables.sort((a, b) => b.d - a.d);

  const tNow = tNorm * TOTAL;
  for (const item of drawables) {
    if (item.kind === 'net') {
      // Net as a surface, not just a tape: two posts, the curve, and vertical mesh.
      strokePath([[-POST_OUT,NETY,0],[-POST_OUT,NETY,NETP]], '#cfd8dd', 2.4);
      strokePath([[W+POST_OUT,NETY,0],[W+POST_OUT,NETY,NETP]], '#cfd8dd', 2.4);
      for (let i = 0; i <= 28; i += 2) {
        const p = NET[i];
        strokePath([[p[0],NETY,0],[p[0],NETY,p[2]]], 'rgba(200,215,225,.28)', 1);
      }
      strokePath(NET, '#eef4f7', 2.2);
      continue;
    }
    const {i, s} = item;
    const on = i === selected;
    const active = tNow >= s.start_s && tNow < s.end_s;
    // Depth fog: far arcs recede instead of every arc reading at the same weight.
    const fade = Math.max(0.25, Math.min(1, 34 / item.d));
    const colour = on ? '#4ec9b0'
                 : active ? '#ffa657'
                 : `rgba(111,151,173,${(0.5 * fade).toFixed(2)})`;
    strokePath(s.points, colour, on || active ? 3 : 1.6);

    if (active && s.points.length > 1) {
      const f = (tNow - s.start_s) / Math.max(s.end_s - s.start_s, 1e-6);
      const idx = Math.min(s.points.length - 1, Math.floor(f * (s.points.length - 1)));
      const P = project(s.points[idx]);
      if (P) {
        ctx.fillStyle = '#ffd166';
        ctx.beginPath(); ctx.arc(P[0], P[1], Math.max(3, FSCALE * 0.06 / P[2]), 0, 2*Math.PI);
        ctx.fill();
      }
    }
  }
}

/* ---------- interaction ---------- */
let drag = null;
cv.addEventListener('pointerdown', e => {
  drag = {x:e.clientX, y:e.clientY, moved:false};
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointerup', e => {
  if (drag && !drag.moved) pickAt(e);      // a click that never dragged selects an arc
  drag = null;
});
cv.addEventListener('pointermove', e => {
  if (!drag) return;
  if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) drag.moved = true;
  cam.az += (e.clientX - drag.x) * 0.006;
  cam.polar = Math.max(0.06, Math.min(1.45, cam.polar + (e.clientY - drag.y) * 0.005));
  drag.x = e.clientX; drag.y = e.clientY;
  requestDraw();
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  cam.dist = Math.max(16, Math.min(90, cam.dist * (1 + Math.sign(e.deltaY) * 0.09)));
  requestDraw();
}, {passive:false});

function pickAt(e) {
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = -1, bestD = 14;
  DATA.segments.forEach((s, i) => {
    for (const p of s.points) {
      const P = project(p);
      if (!P) continue;
      const d = Math.hypot(P[0] - mx, P[1] - my);
      if (d < bestD) { bestD = d; best = i; }
    }
  });
  if (best >= 0) select(best);
}

/* ---------- segment list ---------- */
const list = document.getElementById('list');
if (!DATA.segments.length) {
  list.innerHTML = '<div class="empty">No flight segments passed the physical gates ' +
    'for this clip.<br><br>That means the court fit, the flight timing or the event ' +
    'detection did not meet the bar — so nothing is shown rather than something ' +
    'unreliable.</div>';
}
DATA.segments.forEach((s, i) => {
  const b = document.createElement('button');
  b.className = 'seg'; b.setAttribute('aria-selected', 'false');
  b.innerHTML =
    `<span class="top"><span class="name">${s.label}</span>` +
    `<span class="kmh">${s.speed_kmh.toFixed(0)} km/h</span></span>` +
    `<span class="sub2">apex ${s.apex_m.toFixed(1)} m · ` +
    `${(s.end_s - s.start_s).toFixed(2)} s · f${s.start_frame}–${s.end_frame}</span>` +
    `<details><summary>evidence</summary>` +
    s.evidence.map(e => `<div class="ev">• ${e}</div>`).join('') + `</details>`;
  b.onclick = () => select(i);
  list.appendChild(b);
});

function select(i) {
  selected = i;
  [...list.querySelectorAll('.seg')].forEach((c, j) =>
    c.setAttribute('aria-selected', String(j === i)));
  const card = list.querySelectorAll('.seg')[i];
  if (card) card.scrollIntoView({block:'nearest'});
  const s = DATA.segments[i];
  if (s) { tNorm = s.start_s / TOTAL; scrub.value = tNorm * 1000; updateTime(); }
  requestDraw();
}

const note = document.getElementById('note');
note.innerHTML = (DATA.court_valid ? '' : '<span class="warn">COURT FIT FAILED ' +
  'VALIDATION — positions and speeds below are NOT measurements.</span><br><br>') +
  DATA.note;

/* ---------- timeline ---------- */
const scrub = document.getElementById('scrub'), tlabel = document.getElementById('tlabel');
const ticks = document.getElementById('ticks');
DATA.segments.forEach(s => {                 // segment spans, so gaps are visible
  const i = document.createElement('i');
  i.style.left = (100 * s.start_s / TOTAL) + '%';
  i.style.width = Math.max(0.6, 100 * (s.end_s - s.start_s) / TOTAL) + '%';
  ticks.appendChild(i);
});
function updateTime() { tlabel.textContent = (tNorm * TOTAL).toFixed(1) + ' s'; }
scrub.oninput = () => { setPlaying(false); tNorm = scrub.value / 1000; updateTime(); requestDraw(); };
document.getElementById('rate').onchange = e => rate = parseFloat(e.target.value);

const playBtn = document.getElementById('play');
let last = 0;
function setPlaying(on) {
  playing = on; playBtn.textContent = on ? '❚❚ Pause' : '▶ Play';
  last = 0;                                  // reset, or the first frame after a pause
  if (on) requestAnimationFrame(tick);       // advances by the whole pause duration
}
playBtn.onclick = () => setPlaying(!playing);
function tick(ts) {
  if (!playing) return;
  if (last) {
    tNorm += rate * (ts - last) / 1000 / Math.max(TOTAL, 0.001);
    if (tNorm > 1) tNorm = 0;
    scrub.value = tNorm * 1000; updateTime(); draw();
  }
  last = ts; requestAnimationFrame(tick);
}

/* ---------- views, keyboard, tabs ---------- */
document.querySelectorAll('[data-view]').forEach(b =>
  b.onclick = () => { cam = {...VIEWS[b.dataset.view]}; requestDraw(); });
document.getElementById('reset').onclick = () => { cam = {...VIEWS.broadcast}; requestDraw(); };

addEventListener('keydown', e => {
  const k = e.key;
  if (k === ' ') { e.preventDefault(); setPlaying(!playing); }
  else if (k === 'ArrowLeft')  cam.az -= 0.08;
  else if (k === 'ArrowRight') cam.az += 0.08;
  else if (k === 'ArrowUp')    cam.polar = Math.max(0.06, cam.polar - 0.06);
  else if (k === 'ArrowDown')  cam.polar = Math.min(1.45, cam.polar + 0.06);
  else if (k === '+' || k === '=') cam.dist = Math.max(16, cam.dist * 0.92);
  else if (k === '-') cam.dist = Math.min(90, cam.dist * 1.08);
  else if (k === 'r' || k === 'R') cam = {...VIEWS.broadcast};
  else return;
  requestDraw();
});

const vid = document.getElementById('vid'), b3 = document.getElementById('btn3d'),
      bv = document.getElementById('btnVideo'), hint = document.getElementById('hint');
if (!DATA.video) { bv.disabled = true; bv.title = 'no annotated video alongside this file'; }
vid.onerror = () => { bv.disabled = true; hint.textContent = 'annotated video not found next to this page'; b3.onclick(); };
b3.onclick = () => {
  vid.pause(); cv.classList.remove('hidden'); vid.classList.add('hidden');
  b3.classList.add('active'); bv.classList.remove('active');
  hint.textContent = 'drag orbit · scroll zoom · arrows/space/R'; resize();
};
bv.onclick = () => {
  if (DATA.video && !vid.src) vid.src = DATA.video;   // lazy: don't fetch on page load
  setPlaying(false);
  vid.currentTime = tNorm * TOTAL;                    // keep both clocks in step
  vid.classList.remove('hidden'); cv.classList.add('hidden');
  bv.classList.add('active'); b3.classList.remove('active');
  hint.textContent = 'video and 3D share the same timeline';
};
vid.addEventListener('timeupdate', () => {
  if (vid.classList.contains('hidden')) return;
  tNorm = Math.min(1, vid.currentTime / Math.max(TOTAL, 0.001));
  scrub.value = tNorm * 1000; updateTime();
});

resize(); updateTime();
</script>
</body>
</html>
"""


def build_viewer(
    trajectories,
    output_path: str | Path,
    fps: float,
    video_path: str | None = None,
    shot_types: dict[int, str] | None = None,
    court_valid: bool = True,
    surface: str = "unknown",
) -> Path:
    """
    Write a self-contained interactive 3-D viewer for one analysed clip.

    Args:
        trajectories: Trajectory3D segments that passed the pipeline's physical gates.
        output_path:  where to write the .html.
        fps:          frame rate, used to place segments on a seconds timeline.
        video_path:   annotated video, referenced by RELATIVE name so the two files
                      travel together rather than producing a huge embedded page.
        shot_types:   frame -> shot label.
        court_valid:  when false the page says so, matching the video's warning banner.
        surface:      hard | clay | grass, used to colour the court.

    Returns the written path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shot_types = shot_types or {}

    segments = []
    for t in trajectories:
        segments.append({
            "start_frame": t.start_frame,
            "end_frame": t.end_frame,
            "start_s": round(t.start_frame / fps, 3) if fps else 0.0,
            "end_s": round(t.end_frame / fps, 3) if fps else 0.0,
            "speed_kmh": round(t.speed_kmh, 1),
            "apex_m": round(t.apex_height_m, 2),
            "label": shot_types.get(t.start_frame) or "Flight",
            "points": [[round(c, 3) for c in p] for p in t.points],
            "evidence": _segment_evidence(t, fps),
        })

    data = {
        "court": {
            "length": COURT_LENGTH_M,
            "width": COURT_WIDTH_DOUBLES_M,
            "singles_width": COURT_WIDTH_SINGLES_M,
            "service_from_net": SERVICE_LINE_FROM_NET_M,
            "net_centre": NET_HEIGHT_CENTRE_M,
            "net_post": NET_HEIGHT_POST_M,
            "post_outside": NET_POST_OUTSIDE_M,
        },
        "segments": segments,
        # Only the file name: an absolute Windows path would be escaped into a src
        # that some browsers refuse, and the page is meant to sit beside its video.
        "video": Path(video_path).name if video_path else None,
        "court_valid": bool(court_valid),
        "surface_colour": SURFACE_COLOURS.get(surface, SURFACE_COLOURS["unknown"]),
        "note": ("Each arc is one free-flight reconstruction between detected events. "
                 "Speed is the average over the flight, so it reads at or just below a "
                 "radar gun, which measures at contact. Drag and spin are not modelled."),
    }

    # allow_nan=False so a NaN fails loudly here rather than emitting bare NaN, which
    # is invalid JSON but valid JS — the page would load and the arc would silently
    # vanish into undefined coordinates.
    payload = json.dumps(data, allow_nan=False)
    # A "</" inside any string would close the <script> block early and break the page.
    payload = payload.replace("</", "<\\/")

    output_path.write_text(_page_template().replace("__DATA__", payload), encoding="utf-8")
    return output_path
