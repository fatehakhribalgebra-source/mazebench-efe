"""Standalone HTML replay viewer.

The point is to make imagination inspectable. For every decision the page shows,
side by side:

* the real board the agent saw
* the plan it committed to
* the rollout it imagined, decoded step by step back into glyphs
* the EFE term breakdown, so you can see *why* it chose that action -- whether
  the move was driven by preference (a gem nearby), by curiosity (ensemble
  disagreement, unvisited cells), or by ambiguity avoidance

Without this you are reading scalar metrics and guessing. With it you can catch
the specific failure modes that matter: an agent that rotates the camera forever
because novelty is cheap, one that walks into walls because its `moved` head is
wrong, or imagined rollouts that diverge from reality after two steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from env.actions import ACTION_NAMES
from env.glyphs import ID_TO_GLYPH, PAD_ID


def window_to_text(window: np.ndarray) -> str:
    rows = []
    for row in np.asarray(window):
        rows.append("".join(" " if int(v) == PAD_ID else ID_TO_GLYPH.get(int(v), "?") for v in row))
    return "\n".join(rows)


def build_payload(history, title: str, summary: dict, baseline: dict | None = None) -> dict:
    steps = []
    for record in history:
        steps.append({
            "step": record.step,
            "kind": record.kind,
            "action": record.action_name,
            "room": record.room,
            "gems": record.gem_count,
            "board": record.board,
            "efe": record.efe,
            "terms": {k: round(float(v), 4) for k, v in (record.terms or {}).items()},
            "plan": [ACTION_NAMES[a] for a in (record.plan or [])],
            "imagined": [window_to_text(w) for w in (record.imagined or [])],
            "real": window_to_text(record.real_window) if record.real_window is not None else None,
            "novelty": {k: round(float(v), 4) for k, v in (record.novelty or {}).items()},
            "note": record.note,
        })
    return {"title": title, "summary": summary, "baseline": baseline, "steps": steps}


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --line:#272b35; --text:#e6e8ee;
    --muted:#9aa3b2; --accent:#6cd7ff; --warm:#ffb454; --good:#5aa95c; --bad:#ef4444;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:20px 16px 12px; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 4px; font-size:19px; font-weight:650; letter-spacing:-0.01em; }
  .sub { color:var(--muted); font-size:13px; }
  .wrap { padding:16px; max-width:1180px; margin:0 auto; }
  .cards { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 18px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:9px 12px; min-width:112px; }
  .card .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .card .v { font-size:19px; font-weight:650; font-variant-numeric:tabular-nums; }
  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
              background:var(--panel); border:1px solid var(--line);
              border-radius:8px; padding:10px 12px; margin-bottom:14px; }
  input[type=range] { flex:1; min-width:180px; accent-color:var(--accent); }
  button { background:#222733; color:var(--text); border:1px solid var(--line);
           border-radius:6px; padding:6px 12px; font-size:13px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:860px){ .grid{ grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:8px; padding:12px; min-width:0; }
  .panel h2 { margin:0 0 8px; font-size:12px; text-transform:uppercase;
              letter-spacing:.06em; color:var(--muted); font-weight:600; }
  pre { margin:0; font:12px/1.25 ui-monospace,SFMono-Regular,Consolas,monospace;
        white-space:pre; overflow-x:auto; color:#cfd6e4; }
  .imagined { display:flex; gap:10px; overflow-x:auto; padding-bottom:4px; }
  .imagined figure { margin:0; flex:0 0 auto; }
  .imagined figcaption { color:var(--muted); font-size:11px; margin-bottom:3px; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  td { padding:2px 0; }
  td.n { text-align:right; color:var(--muted); width:74px; }
  .bar { height:6px; border-radius:3px; background:#2a2f3a; overflow:hidden; }
  .bar > i { display:block; height:100%; }
  .tag { display:inline-block; padding:1px 7px; border-radius:99px; font-size:11px;
         border:1px solid var(--line); color:var(--muted); }
  .tag.plan{ color:var(--accent); border-color:#245 }
  .tag.recover{ color:var(--bad); border-color:#532 }
  .tag.jump{ color:var(--warm); border-color:#543 }
  .note { color:var(--warm); font-size:12px; margin-top:6px; }
  .legend { color:var(--muted); font-size:12px; margin-top:10px; }
  code { background:#222733; padding:1px 4px; border-radius:3px; font-size:12px; }
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub" id="sub"></div>
</header>
<div class="wrap">
  <div class="cards" id="cards"></div>
  <div class="controls">
    <button id="play">play</button>
    <button id="prev">&larr;</button>
    <button id="next">&rarr;</button>
    <input type="range" id="slider" min="0" value="0">
    <span id="pos" class="tag"></span>
  </div>
  <div class="grid">
    <div class="panel">
      <h2>Real board</h2>
      <pre id="board"></pre>
      <div class="note" id="note"></div>
    </div>
    <div class="panel">
      <h2>Why this action &mdash; expected free energy</h2>
      <table id="terms"></table>
      <div class="legend">
        <code>G = risk + ambiguity &minus; epistemic</code>, minimised.
        Risk is dispreference, epistemic is information gain.
      </div>
    </div>
  </div>
  <div class="panel" style="margin-top:14px">
    <h2>Imagined rollout &mdash; decoded from the latent world model</h2>
    <div class="imagined" id="imagined"></div>
    <div class="legend" id="planline"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const steps = DATA.steps;
const el = id => document.getElementById(id);
let i = 0, timer = null;

function card(k, v) {
  return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
}
function renderHeader() {
  const s = DATA.summary || {};
  const keys = ["gems","rooms_visited","unique_board_states","state_novelty_rate",
                "novel_block_positions","deaths","blocked_move_rate","total"];
  el("cards").innerHTML = keys.filter(k => k in s).map(k => {
    let v = s[k];
    if (typeof v === "number" && !Number.isInteger(v)) v = v.toFixed(3);
    return card(k.replace(/_/g," "), v);
  }).join("");
  let sub = `${steps.length} decisions`;
  if (DATA.baseline && DATA.baseline.gems !== undefined) {
    sub += ` &middot; random baseline: ${DATA.baseline.gems} gems, ` +
           `${DATA.baseline.rooms_visited} rooms, ` +
           `${(DATA.baseline.state_novelty_rate ?? 0).toFixed(3)} novelty rate`;
  }
  el("sub").innerHTML = sub;
}

const POS = {epistemic:1, utility:1, gem_proximity:1, cell_novelty:1,
             pattern_novelty:1, disagreement:1, e_gems:1, p_room_change:1};
function renderTerms(s) {
  const t = s.terms || {};
  const keys = Object.keys(t);
  if (!keys.length) { el("terms").innerHTML =
    '<tr><td style="color:var(--muted)">no plan this step &mdash; supervisor acted</td></tr>'; return; }
  const max = Math.max(...keys.map(k => Math.abs(t[k])), 1e-6);
  const order = ["risk","ambiguity","epistemic","utility","e_gems","gem_proximity",
                 "p_room_change","p_death","disagreement","cell_novelty","pattern_novelty"];
  const sorted = order.filter(k => k in t).concat(keys.filter(k => !order.includes(k)));
  el("terms").innerHTML = sorted.map(k => {
    const v = t[k], w = Math.min(100, Math.abs(v) / max * 100);
    const col = (k in POS) ? "var(--good)" : "var(--warm)";
    return `<tr><td style="width:40%">${k.replace(/_/g," ")}</td>
      <td><div class="bar"><i style="width:${w}%;background:${col}"></i></div></td>
      <td class="n">${v.toFixed(3)}</td></tr>`;
  }).join("") + (s.efe != null
      ? `<tr><td colspan="2" style="padding-top:8px"><b>total G</b></td>
         <td class="n" style="padding-top:8px"><b>${s.efe.toFixed(3)}</b></td></tr>` : "");
}

function render() {
  const s = steps[i];
  el("pos").textContent = `step ${s.step} / ${steps.length - 1}`;
  el("board").textContent = s.board || "(no board)";
  el("note").textContent = s.note || "";
  renderTerms(s);
  const imagined = s.imagined || [];
  el("imagined").innerHTML = imagined.length
    ? (s.real ? `<figure><figcaption>actually seen (t)</figcaption><pre>${s.real}</pre></figure>` : "")
      + imagined.map((w, k) =>
        `<figure><figcaption>imagined t+${k + 1}</figcaption><pre>${w}</pre></figure>`).join("")
    : '<span style="color:var(--muted)">no imagination recorded for this step</span>';
  el("planline").innerHTML = (s.plan && s.plan.length)
    ? `<span class="tag ${s.kind}">${s.kind}</span> chose <b>${s.action}</b> &mdash; plan: ${s.plan.join(" &rarr; ")}`
    : `<span class="tag ${s.kind}">${s.kind}</span> ${s.action}`;
  el("slider").value = i;
}
el("slider").max = steps.length - 1;
el("slider").oninput = e => { i = +e.target.value; render(); };
el("prev").onclick = () => { i = Math.max(0, i - 1); render(); };
el("next").onclick = () => { i = Math.min(steps.length - 1, i + 1); render(); };
el("play").onclick = () => {
  if (timer) { clearInterval(timer); timer = null; el("play").textContent = "play"; return; }
  el("play").textContent = "pause";
  timer = setInterval(() => {
    i = (i + 1) % steps.length; render();
  }, 450);
};
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") el("prev").click();
  if (e.key === "ArrowRight") el("next").click();
});
renderHeader(); render();
</script></body></html>
"""


def write_replay(
    history,
    path: str | Path,
    title: str = "MazeBench EFE agent",
    summary: dict | None = None,
    baseline: dict | None = None,
) -> Path:
    payload = build_payload(history, title, summary or {}, baseline)
    html = (
        _HTML.replace("__DATA__", json.dumps(payload))
        .replace("__TITLE__", title)
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf8")
    return path
