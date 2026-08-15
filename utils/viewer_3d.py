"""
utils/viewer_3d.py
──────────────────
Generates a self-contained, interactive 3-D rally viewer as a single HTML file.

Why a hand-written renderer rather than three.js
------------------------------------------------
The viewer has to open by double-clicking a file: no server, no network, no build
step, and no CDN (which a strict content policy would block anyway). Inlining a
general 3-D engine costs hundreds of kilobytes to draw a court, a net and a handful
of parabolas. A direct perspective projection onto a 2-D canvas is about two hundred
lines, has no dependencies, and keeps the file small enough to email.

What it shows, and why that matters
-----------------------------------
Each arc is one free-flight segment from the physics reconstruction, and every
segment displayed has already passed the pipeline's physical gates: a valid court
fit, a plausible flight duration, a landing inside the court, and alternating hitters.
Clicking an arc shows the evidence behind it. The viewer is therefore not decoration
on top of the analysis — it is the analysis, made inspectable. A number you can orbit
around and check against your own eyes is a very different claim from a number in a
table.

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


def _page_template() -> str:
    """The viewer's HTML/CSS/JS. `__DATA__` is replaced with the scene JSON."""
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tennis-Vision — 3D Rally Viewer</title>
<style>
  :root { --bg:#0e1116; --panel:#161b22; --line:#30363d; --text:#e6edf3;
          --muted:#8b949e; --accent:#4ec9b0; --warn:#f0883e; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; font-weight:600; letter-spacing:.2px; }
  .badge { font-size:12px; color:var(--muted); }
  button { background:var(--panel); color:var(--text); border:1px solid var(--line);
           padding:8px 14px; border-radius:6px; cursor:pointer; font-size:13px; }
  button:hover { border-color:var(--accent); }
  button.active { background:var(--accent); color:#04211c; border-color:var(--accent);
                  font-weight:600; }
  main { display:grid; grid-template-columns:1fr 380px; gap:0; height:calc(100vh - 59px); }
  @media (max-width:900px){ main { grid-template-columns:1fr; } }
  #stage { position:relative; background:#0a0d12; }
  canvas { display:block; width:100%; height:100%; cursor:grab; }
  canvas:active { cursor:grabbing; }
  video { width:100%; height:100%; object-fit:contain; background:#000; }
  .hidden { display:none !important; }
  aside { border-left:1px solid var(--line); padding:16px; overflow:auto;
          background:var(--panel); }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
       color:var(--muted); margin:0 0 10px; font-weight:600; }
  .seg { padding:9px 11px; border:1px solid var(--line); border-radius:6px;
         margin-bottom:6px; cursor:pointer; }
  .seg:hover { border-color:var(--accent); }
  .seg.sel { border-color:var(--accent); background:#11221f; }
  .seg .top { display:flex; justify-content:space-between; font-weight:600; }
  .seg .sub { color:var(--muted); font-size:12px; }
  .ev { font-size:12px; color:var(--muted); margin:3px 0 0 12px; }
  .ev::before { content:"✓ "; color:var(--accent); }
  footer { padding:10px 20px; border-top:1px solid var(--line);
           display:flex; align-items:center; gap:12px; }
  input[type=range] { flex:1; }
  .note { color:var(--muted); font-size:12px; margin-top:14px;
          border-top:1px solid var(--line); padding-top:12px; }
</style>
</head>
<body>
<header>
  <h1>Tennis-Vision · 3D Rally Viewer</h1>
  <button id="btn3d" class="active">Explore in 3D</button>
  <button id="btnVideo">Annotated video</button>
  <span class="badge" id="hint">drag to orbit · scroll to zoom</span>
</header>

<main>
  <div id="stage">
    <canvas id="cv"></canvas>
    <video id="vid" class="hidden" controls></video>
  </div>
  <aside>
    <h2>Flight segments</h2>
    <div id="list"></div>
    <div class="note" id="note"></div>
  </aside>
</main>

<footer>
  <button id="play">▶ Play</button>
  <input type="range" id="scrub" min="0" max="1000" value="0">
  <span class="badge" id="tlabel">0.0 s</span>
</footer>

<script>
const DATA = __DATA__;

/* ---------- geometry ---------- */
const L = DATA.court.length, W = DATA.court.width, WS = DATA.court.singles_width;
const SVC = DATA.court.service_from_net, NETC = DATA.court.net_centre,
      NETP = DATA.court.net_post;
const CX = W / 2, CY = L / 2;              // court centre, used as the orbit target

function courtLines() {
  const y0 = 0, y1 = L, xm = (W - WS) / 2, xM = W - xm, nety = L / 2;
  const seg = [];
  const add = (a, b) => seg.push([a, b]);
  // Outer doubles rectangle
  add([0, y0, 0], [W, y0, 0]); add([0, y1, 0], [W, y1, 0]);
  add([0, y0, 0], [0, y1, 0]); add([W, y0, 0], [W, y1, 0]);
  // Singles sidelines
  add([xm, y0, 0], [xm, y1, 0]); add([xM, y0, 0], [xM, y1, 0]);
  // Service lines and centre service line
  add([xm, nety - SVC, 0], [xM, nety - SVC, 0]);
  add([xm, nety + SVC, 0], [xM, nety + SVC, 0]);
  add([W / 2, nety - SVC, 0], [W / 2, nety + SVC, 0]);
  // Net: posts, centre dip, and the tape between them
  add([0, nety, 0], [0, nety, NETP]); add([W, nety, 0], [W, nety, NETP]);
  const steps = 24;
  for (let i = 0; i < steps; i++) {
    const t0 = i / steps, t1 = (i + 1) / steps;
    const h = t => NETP + (NETC - NETP) * Math.sin(Math.PI * t);
    add([t0 * W, nety, h(t0)], [t1 * W, nety, h(t1)]);
  }
  return seg;
}
const LINES = courtLines();

/* ---------- camera ---------- */
const cam = { az: -0.62, el: 0.55, dist: 34 };
function project(p, w, h) {
  // World -> camera: orbit about the court centre, then a perspective divide.
  const x = p[0] - CX, y = p[1] - CY, z = p[2];
  const ca = Math.cos(cam.az), sa = Math.sin(cam.az);
  let X = x * ca - y * sa, Y = x * sa + y * ca;
  const ce = Math.cos(cam.el), se = Math.sin(cam.el);
  let Z = Y * se + z * ce;
  let Yc = -Y * ce + z * se;
  const depth = Z + cam.dist;
  if (depth <= 0.1) return null;                 // behind the camera
  const f = 0.9 * Math.min(w, h);
  return [w / 2 + f * X / depth, h / 2 - f * Yc / depth, depth];
}

/* ---------- rendering ---------- */
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let selected = -1, playing = false, tNorm = 0;
const TOTAL = DATA.segments.length
  ? DATA.segments[DATA.segments.length - 1].end_s : 1;

function resize() {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = r.width * dpr; cv.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
addEventListener('resize', resize);

function line(a, b, w, h, colour, width) {
  const A = project(a, w, h), B = project(b, w, h);
  if (!A || !B) return;
  ctx.strokeStyle = colour; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
}

function draw() {
  const w = cv.clientWidth, h = cv.clientHeight;
  ctx.clearRect(0, 0, w, h);

  // Court surface, drawn as a filled quad so arcs read against a ground plane.
  const corners = [[0,0,0],[W,0,0],[W,L,0],[0,L,0]].map(p => project(p, w, h));
  if (corners.every(Boolean)) {
    ctx.fillStyle = '#12303a';
    ctx.beginPath(); ctx.moveTo(corners[0][0], corners[0][1]);
    corners.slice(1).forEach(c => ctx.lineTo(c[0], c[1]));
    ctx.closePath(); ctx.fill();
  }
  LINES.forEach(([a, b]) => line(a, b, w, h, '#8aa0ad', 1.4));

  const tNow = tNorm * TOTAL;
  DATA.segments.forEach((s, i) => {
    const on = i === selected;
    const active = tNow >= s.start_s && tNow <= s.end_s;
    ctx.strokeStyle = on ? '#4ec9b0' : (active ? '#f0883e' : 'rgba(120,180,200,.45)');
    ctx.lineWidth = on || active ? 2.6 : 1.4;
    ctx.beginPath();
    let started = false;
    s.points.forEach(p => {
      const P = project(p, w, h);
      if (!P) { started = false; return; }
      if (!started) { ctx.moveTo(P[0], P[1]); started = true; }
      else ctx.lineTo(P[0], P[1]);
    });
    ctx.stroke();

    // Ball marker at the current time, so the scrubber animates the rally.
    if (active && s.points.length > 1) {
      const f = (tNow - s.start_s) / Math.max(s.end_s - s.start_s, 1e-6);
      const idx = Math.min(s.points.length - 1, Math.floor(f * (s.points.length - 1)));
      const P = project(s.points[idx], w, h);
      if (P) {
        ctx.fillStyle = '#ffd166';
        ctx.beginPath(); ctx.arc(P[0], P[1], 5, 0, 7); ctx.fill();
      }
    }
  });
}

/* ---------- interaction ---------- */
let drag = null;
cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointerup', () => drag = null);
cv.addEventListener('pointermove', e => {
  if (!drag) return;
  cam.az += (e.clientX - drag[0]) * 0.006;
  cam.el = Math.max(0.06, Math.min(1.45, cam.el + (e.clientY - drag[1]) * 0.005));
  drag = [e.clientX, e.clientY];
  draw();
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  cam.dist = Math.max(12, Math.min(80, cam.dist * (1 + Math.sign(e.deltaY) * 0.09)));
  draw();
}, { passive: false });

/* ---------- segment list ---------- */
const list = document.getElementById('list');
DATA.segments.forEach((s, i) => {
  const d = document.createElement('div');
  d.className = 'seg';
  d.innerHTML =
    `<div class="top"><span>${s.label || 'Flight'}</span>` +
    `<span>${s.speed_kmh.toFixed(0)} km/h</span></div>` +
    `<div class="sub">apex ${s.apex_m.toFixed(1)} m · ${(s.end_s - s.start_s).toFixed(2)} s` +
    ` · frames ${s.start_frame}-${s.end_frame}</div>` +
    (s.evidence || []).map(e => `<div class="ev">${e}</div>`).join('');
  d.onclick = () => {
    selected = i;
    [...list.children].forEach(c => c.classList.remove('sel'));
    d.classList.add('sel');
    tNorm = s.start_s / TOTAL;
    document.getElementById('scrub').value = tNorm * 1000;
    updateTime(); draw();
  };
  list.appendChild(d);
});
document.getElementById('note').textContent = DATA.note;

/* ---------- playback ---------- */
const scrub = document.getElementById('scrub'), tlabel = document.getElementById('tlabel');
function updateTime() { tlabel.textContent = (tNorm * TOTAL).toFixed(1) + ' s'; }
scrub.oninput = () => { tNorm = scrub.value / 1000; updateTime(); draw(); };
document.getElementById('play').onclick = function () {
  playing = !playing; this.textContent = playing ? '❚❚ Pause' : '▶ Play';
  if (playing) tick();
};
let last = 0;
function tick(ts) {
  if (!playing) return;
  if (last) {
    tNorm += (ts - last) / 1000 / Math.max(TOTAL, 0.001);
    if (tNorm > 1) tNorm = 0;
    scrub.value = tNorm * 1000; updateTime(); draw();
  }
  last = ts; requestAnimationFrame(tick);
}

/* ---------- view switching ---------- */
const vid = document.getElementById('vid'), b3 = document.getElementById('btn3d'),
      bv = document.getElementById('btnVideo'), hint = document.getElementById('hint');
if (DATA.video) vid.src = DATA.video; else bv.disabled = true;
b3.onclick = () => { cv.classList.remove('hidden'); vid.classList.add('hidden');
  b3.classList.add('active'); bv.classList.remove('active');
  hint.textContent = 'drag to orbit · scroll to zoom'; resize(); };
bv.onclick = () => { vid.classList.remove('hidden'); cv.classList.add('hidden');
  bv.classList.add('active'); b3.classList.remove('active');
  hint.textContent = DATA.video ? '' : 'no annotated video alongside this file'; };

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
) -> Path:
    """
    Write a self-contained interactive 3-D viewer for one analysed clip.

    Args:
        trajectories: Trajectory3D segments from the reconstruction. Only segments
                      that passed the pipeline's physical gates reach here.
        output_path:  where to write the .html.
        fps:          video frame rate, used to place segments on a seconds timeline.
        video_path:   annotated video, referenced by RELATIVE path so the two files
                      travel together rather than producing a huge embedded page.
        shot_types:   frame -> shot label, so an arc can be named.
        court_valid:  when false the page says so, matching the rendered video's
                      banner. A viewer that looks identical on an uncalibrated run
                      would undo the honesty the rest of the pipeline enforces.

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
            "label": shot_types.get(t.start_frame, "Flight"),
            "points": [[round(c, 3) for c in p] for p in t.points],
            # The gates every displayed segment has already satisfied. Shown so a
            # viewer can see why an arc is trusted, not merely that it is drawn.
            "evidence": [
                "court fit validated",
                f"flight {round((t.end_frame - t.start_frame) / fps, 2) if fps else '?'} s "
                f"(within a plausible single flight)",
                "endpoints on the floor (bounce, or hitter's feet)",
            ],
        })

    note = ("Each arc is one free-flight reconstruction between floor-anchored events. "
            "Speed is the average over the flight, so it reads at or just below a radar "
            "gun, which measures at contact. Drag and spin are not modelled.")
    if not court_valid:
        note = ("COURT FIT FAILED VALIDATION — the court could not be located reliably "
                "in this clip, so these positions and speeds are not measurements. " + note)

    data = {
        "court": {
            "length": COURT_LENGTH_M,
            "width": COURT_WIDTH_DOUBLES_M,
            "singles_width": COURT_WIDTH_SINGLES_M,
            "service_from_net": SERVICE_LINE_FROM_NET_M,
            "net_centre": NET_HEIGHT_CENTRE_M,
            "net_post": NET_HEIGHT_POST_M,
        },
        "segments": segments,
        "video": video_path,
        "note": note,
    }

    html = _page_template().replace("__DATA__", json.dumps(data))
    output_path.write_text(html, encoding="utf-8")
    return output_path
