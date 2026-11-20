"""Local web replay of a single update.

``export_replay`` runs one plan through the simulator and writes a
self-contained JSON document; ``serve`` serves a small page that animates it.
The page shows, for every tick:

* which switch changed (highlighted on the topology),
* where every flow is (its ingress trail),
* whether any invariant is broken at that tick (a red banner),
* the round that is being delivered / acknowledged,
* the current table rules of any switch you click.

Nothing is uploaded anywhere: the page reads the JSON from the same local
server that serves it.
"""

from __future__ import annotations

import json
import os
import random
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Flow, Topology
from .planner import Plan, Planner
from .sim import RunningPlan, Timing, simulate_baseline


# ---------------------------------------------------------------------------
# replay document
# ---------------------------------------------------------------------------


def export_replay(
    topo: Topology,
    flows: Sequence[Flow],
    plan: Optional[Plan] = None,
    seed: int = 1,
    fail_switch: Optional[int] = None,
    fail_tick: Optional[int] = None,
    title: str = "table update",
) -> dict:
    """Build the JSON the replay page consumes."""
    if plan is None:
        plan = Planner(topo, flows, drain_ticks=2).plan()
    rng = random.Random(seed)
    sim = RunningPlan(topo, flows, plan.rounds, Timing(), rng)
    res = sim.run(fail_switch=fail_switch, fail_tick=fail_tick)
    return _assemble(topo, flows, plan, res, title)


def export_baseline_replay(
    topo: Topology, flows: Sequence[Flow], seed: int = 1, title: str = "baseline"
) -> dict:
    res = simulate_baseline(topo, flows, random.Random(seed))
    # fabricate an empty plan container for the assembler
    plan = Planner(topo, flows).plan()
    return _assemble(topo, flows, plan, res, title, baseline=True)


def _assemble(
    topo: Topology,
    flows: Sequence[Flow],
    plan: Plan,
    res,
    title: str,
    baseline: bool = False,
) -> dict:
    links = [{"a": a, "b": b, "cap": c} for (a, b), c in sorted(topo.links.items())]
    flows_doc = []
    for f in flows:
        flows_doc.append(
            {
                "fid": f.fid,
                "src": f.src,
                "dst": f.dst,
                "old": list(f.old),
                "new": list(f.new),
                "demand": f.demand,
            }
        )
    hosts = [
        {"sw": sw, "host": h} for sw, hs in sorted(topo.host_at.items()) for h in hs
    ]
    rounds = []
    for r in plan.rounds:
        rounds.append(
            {
                "index": r.index,
                "kind": r.kind,
                "switches": r.switches,
                "ops": [str(o) for o in r.ops],
                "commands": [
                    {"sw": c.sw, "ops": [o.snapshot() for o in c.ops]} for c in r.commands
                ],
            }
        )
    return {
        "title": title,
        "baseline": baseline,
        "topology": {
            "switches": sorted(topo.switches),
            "links": links,
            "hosts": hosts,
        },
        "flows": flows_doc,
        "rounds": rounds,
        "stats": plan.stats.snapshot(),
        "sim": res.snapshot(),
        "frames": res.trail_trace,
        "events": res.per_tick,
        "violations": [str(v) for v in res.violations],
    }


# ---------------------------------------------------------------------------
# built-in demos
# ---------------------------------------------------------------------------


def demo(kind: str = "swap", seed: int = 1) -> dict:
    from .scenarios import (
        chain_scenario,
        detour_scenario,
        merge_scenario,
        random_scenario,
        swap_scenario,
    )

    if kind == "swap":
        topo, flows = swap_scenario()
        return export_replay(topo, flows, seed=seed, title="adjacent-hop swap (needs version tags)")
    if kind == "merge":
        topo, flows = merge_scenario()
        return export_replay(topo, flows, seed=seed, title="two flows merging (needs rounds)")
    if kind == "detour":
        topo, flows = detour_scenario()
        return export_replay(topo, flows, seed=seed, title="capacity deadlock broken by a transit path")
    if kind == "chain":
        topo, flows = chain_scenario()
        return export_replay(topo, flows, seed=seed, title="dependency chain")
    if kind == "fail":
        # a multi-flow task so the retreat actually has work to do
        from .planner import Planner
        sc = random_scenario(0, switches=10, degree=3, flows=6, cap=10, swap_bias=0.5)
        plan = Planner(sc.topo, sc.flows, drain_ticks=2).plan()
        return export_replay(
            sc.topo, sc.flows, plan, seed=seed, fail_switch=0, fail_tick=3,
            title="switch failure mid-update and safe retreat",
        )
    sc = random_scenario(seed, switches=10, degree=3, flows=6, cap=10, swap_bias=0.6)
    return export_replay(sc.topo, sc.flows, seed=seed, title=f"random task #{seed}")


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>netorch replay</title>
<style>
:root{--bg:#0f1117;--panel:#171a23;--line:#2a2f3d;--txt:#e6e9f0;--dim:#8b93a7;
--ok:#3ddc97;--bad:#ff5c7a;--warn:#ffcc66;--hi:#5aa9ff;--new:#b985ff;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--txt);display:grid;grid-template-columns:1fr 380px;height:100vh}
#left{position:relative;display:flex;flex-direction:column}
#right{background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:14px}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.2px}
header{display:flex;align-items:center;gap:12px;padding:10px 14px;border-bottom:1px solid var(--line)}
#topo{flex:1;min-height:0}
#bar{display:flex;align-items:center;gap:10px;padding:10px 14px;border-top:1px solid var(--line);background:var(--panel)}
#scrub{flex:1}
button{background:#222838;color:var(--txt);border:1px solid var(--line);border-radius:7px;
padding:6px 11px;font:inherit;cursor:pointer}
button:hover{border-color:var(--hi)}
button.on{border-color:var(--hi);color:var(--hi)}
#tick{font-variant-numeric:tabular-nums;color:var(--dim);min-width:74px;text-align:right}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px}
.badge.ok{background:rgba(61,220,151,.13);color:var(--ok)}
.badge.bad{background:rgba(255,92,122,.15);color:var(--bad)}
.badge.dim{background:rgba(139,147,167,.14);color:var(--dim)}
#banner{padding:8px 14px;display:none;font-size:13px;border-bottom:1px solid var(--line)}
#banner.bad{display:block;background:rgba(255,92,122,.1);color:var(--bad)}
.flow-row{display:flex;gap:8px;align-items:baseline;padding:5px 0;border-bottom:1px dashed #232838}
.fid{font-weight:600;color:var(--hi);min-width:34px}
.path{color:var(--dim);font-size:12px;word-break:break-all}
.path b{color:var(--txt);font-weight:500}
.sw{fill:#202634;stroke:#3a4256;stroke-width:1.5}
.sw.changed{fill:#33405e;stroke:var(--hi);stroke-width:2.5}
.sw.dead{fill:#3a1f28;stroke:var(--bad);stroke-width:2.5}
.lbl{fill:#c7cede;font-size:16px;text-anchor:middle;dominant-baseline:middle;pointer-events:none}
.link{stroke:#384053;stroke-width:2}
.link.hot{stroke:var(--warn);stroke-opacity:.85}
.flowline{fill:none;stroke-width:2.5;stroke-opacity:.9}
.host{fill:#2b3142;stroke:#4a5570;stroke-width:1.5}
.rk{font-size:12px;color:var(--dim);margin:10px 0 4px;text-transform:uppercase;letter-spacing:.6px}
.op{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;color:#aab3c8;
padding:1px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.op.cur{color:var(--warn)}
#legend{position:absolute;left:14px;bottom:64px;font-size:11.5px;color:var(--dim);
background:rgba(15,17,23,.75);padding:6px 9px;border-radius:8px;border:1px solid var(--line)}
</style>
</head>
<body>
<script type="application/json" id="replay-data">__DATA__</script>
<div id="left">
  <header>
    <h1 id="title">replay</h1>
    <span id="state" class="badge ok">safe</span>
    <span id="phase" class="badge dim"></span>
  </header>
  <div id="banner"></div>
  <div id="legend">blue ring = switch changing &nbsp;·&nbsp; purple = new path &nbsp;·&nbsp; grey = old path &nbsp;·&nbsp; red = failed</div>
  <svg id="topo" viewBox="0 0 900 620" preserveAspectRatio="xMidYMid meet"></svg>
  <div id="bar">
    <button id="play">▶ play</button>
    <button id="stepb">◀</button>
    <button id="stepf">▶</button>
    <input id="scrub" type="range" min="0" max="0" value="0">
    <span id="tick">t=0</span>
  </div>
</div>
<div id="right">
  <div class="rk">rounds</div>
  <div id="rounds"></div>
  <div class="rk">flows <span class="badge dim" id="fcount"></span></div>
  <div id="flows"></div>
  <div class="rk">switch detail</div>
  <div id="switchbox" class="path">click a switch</div>
</div>
<script>
const D = window.REPLAY || JSON.parse(document.getElementById('replay-data').textContent);
const S = D.topology.switches, L = D.topology.links, H = D.topology.hosts;
const F = D.flows; const frames = D.frames; const events = D.events;
const COL = ['#5aa9ff','#b985ff','#3ddc97','#ffcc66','#ff8f6b','#6be0e0','#ff5c7a','#c7ff6b'];
function flowColor(fid){ return COL[fid % COL.length]; }
const svg = document.getElementById('topo');
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs){ const e=document.createElementNS(NS,tag); for(const k in attrs) e.setAttribute(k,attrs[k]); return e; }

// ---- layout: force-directed on a big circle, deterministic ----
let pos = {};
function layout(){
  const n=S.length, idx={}; S.forEach((s,i)=>idx[s]=i);
  const adj = S.map(()=>[]);
  L.forEach(l=>{ adj[idx[l.a]].push(idx[l.b]); adj[idx[l.b]].push(idx[l.a]); });
  // node ids must be 0..n-1 for this to index correctly
  let p = S.map((s,i)=>({x:450+300*Math.cos(2*Math.PI*i/n), y:310+240*Math.sin(2*Math.PI*i/n)}));
  let temp = 200;
  for(let it=0; it<700; it++){
    const disp = p.map(()=>({x:0,y:0}));
    for(let i=0;i<n;i++) for(let j=i+1;j<n;j++){
      let dx=p[i].x-p[j].x, dy=p[i].y-p[j].y; let d=Math.max(Math.hypot(dx,dy),1);
      const f = 16000/(d*d); dx/=d; dy/=d;
      disp[i].x+=dx*f; disp[i].y+=dy*f; disp[j].x-=dx*f; disp[j].y-=dy*f;
    }
    for(let i=0;i<n;i++) for(const j of adj[i]){
      if(i>=j) continue;
      let dx=p[j].x-p[i].x, dy=p[j].y-p[i].y; let d=Math.max(Math.hypot(dx,dy),1);
      const f = (d-170)*0.09; dx/=d; dy/=d;
      disp[i].x+=dx*f; disp[i].y+=dy*f; disp[j].x-=dx*f; disp[j].y-=dy*f;
    }
    for(let i=0;i<n;i++){
      let dx=disp[i].x, dy=disp[i].y; let d=Math.max(Math.hypot(dx,dy),0.01);
      const lim = Math.min(d, temp);
      p[i].x += dx/d*lim; p[i].y += dy/d*lim;
      p[i].x = Math.max(70, Math.min(830, p[i].x));
      p[i].y = Math.max(70, Math.min(550, p[i].y));
    }
    temp *= 0.99;
  }
  S.forEach((s,i)=>{ pos[s] = p[i]; });
}
layout();

const gLinks = el('g',{}); const gFlows=el('g',{}); const gNodes=el('g',{}); const gHosts=el('g',{});
svg.appendChild(gLinks); svg.appendChild(gFlows); svg.appendChild(gHosts);
const linkEls = {};
L.forEach(l=>{
  const e = el('line',{class:'link', x1:pos[l.a].x,y1:pos[l.a].y,x2:pos[l.b].x,y2:pos[l.b].y});
  e.dataset.a=l.a; e.dataset.b=l.b; gLinks.appendChild(e); linkEls[l.a+'-'+l.b]=e; linkEls[l.b+'-'+l.a]=e;
});
const hostEls = {};
const hostPos = {};
const hostsBySwitch = {};
H.forEach(h=>{ (hostsBySwitch[h.sw] = hostsBySwitch[h.sw]||[]).push(h); });
H.forEach((h,hi)=>{
  const sp = pos[h.sw];
  const grp = hostsBySwitch[h.sw];
  const gi = grp.indexOf(h), gn = grp.length;
  // fan hosts out downward, away from the main body of the graph
  const ang = Math.PI/2 + (gi - (gn-1)/2) * 0.6;
  const hp = {x: sp.x + 52*Math.cos(ang), y: sp.y + 52*Math.sin(ang)};
  hostPos[h.host]=hp;
  const e=el('circle',{class:'host',cx:hp.x,cy:hp.y,r:6}); gHosts.appendChild(e);
  const t=el('text',{class:'lbl',x:hp.x,y:hp.y,fill:'#9aa4bb','font-size':'12'}); t.textContent='host '+h.host;
  t.setAttribute('y', hp.y-11); gHosts.appendChild(t);
  hostEls[h.host]={e,p:hp,sw:h.sw};
});
const swEls = {};
S.forEach(s=>{
  const g = el('g',{});
  const c = el('circle',{class:'sw',cx:pos[s].x,cy:pos[s].y,r:24}); c.dataset.sw=s;
  const t = el('text',{class:'lbl',x:pos[s].x,y:pos[s].y}); t.textContent=s;
  g.appendChild(c); g.appendChild(t); gNodes.appendChild(g); swEls[s]={c,t,pos:pos[s]};
  c.addEventListener('click',()=>showSwitch(s));
});
svg.appendChild(gNodes);
const flowLines = {};
F.forEach(f=>{
  const e = el('path',{class:'flowline',stroke:flowColor(f.fid),opacity:0.9}); gFlows.appendChild(e);
  flowLines[f.fid]=e;
});

function off(p1,p2,r){
  // point at distance r from p1 towards p2, so lines start at the node edge
  const dx=p2.x-p1.x, dy=p2.y-p1.y, d=Math.hypot(dx,dy)||1;
  return {x:p1.x+dx/d*r, y:p1.y+dy/d*r};
}
function hostFor(last, fid){
  // the host owned by `last` that this flow targets
  const spec = F.find(f=>f.fid===fid);
  const cand = H.filter(h=>h.sw===last && (!spec || h.host===spec.dst));
  return (cand[0]||H.find(h=>h.sw===last)||null);
}
const LINK = new Set();
L.forEach(l=>{ LINK.add(l.a+'-'+l.b); LINK.add(l.b+'-'+l.a); });
function bowOff(fid){ return ((fid % 5) - 2) * 3.2; }
function trailToPath(trail, fid){
  if(!trail || trail.length<2) return '';
  let d = '';
  for(let i=0;i<trail.length-1;i++){
    const a=trail[i], b=trail[i+1];
    if(a===-1||b===-1) continue;
    const pa = pos[a], pb = pos[b];
    if(!pa||!pb) continue;
    const s0 = off(pa,pb,24), s1 = off(pb,pa,24);
    if(LINK.has(a+'-'+b)){
      // tiny perpendicular bow per flow so shared links stay visible
      const bo = bowOff(fid);
      const ux=s1.x-s0.x, uy=s1.y-s0.y, nl=Math.hypot(ux,uy)||1;
      const ox=-uy/nl*bo, oy=ux/nl*bo, mx=(s0.x+s1.x)/2, my=(s0.y+s1.y)/2;
      if(bo===0) d += ` M ${s0.x} ${s0.y} L ${s1.x} ${s1.y}`;
      else d += ` M ${s0.x} ${s0.y} Q ${mx+2*ox} ${my+2*oy} ${s1.x} ${s1.y}`;
    } else {
      // not a physical link: bow the line out so it cannot be mistaken for one
      const mx=(s0.x+s1.x)/2, my=(s0.y+s1.y)/2;
      const ux=s1.x-s0.x, uy=s1.y-s0.y, nl=Math.hypot(ux,uy)||1;
      const ox=-uy/nl*34, oy=ux/nl*34;
      d += ` M ${s0.x} ${s0.y} Q ${mx+ox} ${my+oy} ${s1.x} ${s1.y}`;
    }
  }
  const last = trail[trail.length-1];
  if(last!==-1 && pos[last]){
    const hh = hostFor(last, fid);
    if(hh){ const hp=hostPos[hh.host], pp=pos[last];
      const s0=off(pp,hp,24), s1=off(hp,pp,6);
      d += ` M ${s0.x} ${s0.y} L ${s1.x} ${s1.y}`;
    }
  }
  return d;
}

// ---- state ----
let frame = 0; let playing = false; let timer = null;
const scrub = document.getElementById('scrub');
scrub.max = Math.max(0, frames.length-1);
const fcount = document.getElementById('fcount');
fcount.textContent = F.length;

function eventsAt(t){
  return events.filter(e=>e.t===t);
}
function roundAt(t){
  // the latest send event at or before t
  let r = null;
  for(const e of events){ if(e.kind==='send' && e.t<=t) r=e.round; }
  return r;
}
function render(){
  const fr = frames[Math.min(frame, frames.length-1)];
  const t = fr.t;
  document.getElementById('tick').textContent = 't='+t;
  scrub.value = frame;

  // changed switches at this tick
  const changed = new Set();
  let failing = null;
  for(const e of eventsAt(t)){
    if(e.kind==='apply') changed.add(e.sw);
    if(e.kind==='fail') failing=e.sw;
  }
  const dead = new Set();
  for(const e of events) if(e.kind==='fail' && e.t<=t) dead.add(e.sw);

  // links hot = used by any current trail
  const hot = new Set();
  for(const fl of fr.flows){
    for(const tg of (fl.tags&&fl.tags.length?fl.tags:[{trail:fl.trail}])){
      const tr = tg.trail||[];
      for(let i=0;i<tr.length-1;i++){
        if(tr[i]===-1||tr[i+1]===-1) continue;
        hot.add(tr[i]+'-'+tr[i+1]); hot.add(tr[i+1]+'-'+tr[i]);
      }
    }
  }
  for(const k in linkEls) linkEls[k].classList.toggle('hot', hot.has(k));
  for(const s in swEls){
    swEls[s].c.classList.toggle('changed', changed.has(+s));
    swEls[s].c.classList.toggle('dead', dead.has(+s));
  }
  // flows
  let anyBad=false;
  for(const fl of fr.flows){
    let tr;
    if(fl.tags && fl.tags.length){ tr = fl.tags[0].trail; anyBad = anyBad || fl.tags.some(x=>!x.ok); }
    else { tr = fl.trail; anyBad = anyBad || !fl.ok; }
    flowLines[fl.fid].setAttribute('d', trailToPath(tr, fl.fid));
  }
  const badge = document.getElementById('state');
  const banner = document.getElementById('banner');
  const vs = eventsAt(t).filter(e=>e.kind==='violation');
  if(vs.length){ anyBad = true; }
  badge.className = 'badge ' + (anyBad?'bad':'ok');
  badge.textContent = anyBad? 'violation' : 'safe';
  if(vs.length){
    banner.className='bad';
    banner.textContent = 't='+t+': '+vs[0].items.join('  |  ');
  } else banner.className='';

  const ph = document.getElementById('phase');
  const r = roundAt(t);
  ph.textContent = r===null? 'not started' : 'round '+(r+1)+' / '+D.rounds.length;
  ph.className = 'badge ' + (r===null?'dim':'ok');

  renderFlows(fr);
  renderRounds(t, changed);
}

function renderFlows(fr){
  const box = document.getElementById('flows');
  let html='';
  for(const fl of fr.flows){
    const spec = F.find(f=>f.fid===fl.fid);
    const tr = (fl.tags && fl.tags.length)? fl.tags[fl.tags.length-1].trail : fl.trail;
    const ok = (fl.tags && fl.tags.length)? fl.tags.every(x=>x.ok) : fl.ok;
    html += `<div class="flow-row">
      <span class="fid" style="color:${flowColor(fl.fid)}">f${fl.fid}</span>
      <div><div class="path">${pretty(tr)}</div>
      <div class="path" style="opacity:.7">old ${spec.old.map(x=>'s'+x).join('→')} · new ${spec.new.map(x=>'s'+x).join('→')}
      ${ok?'':'<span style="color:var(--bad)"> ⚠ broken</span>'}</div></div></div>`;
  }
  box.innerHTML = html;
}
function pretty(tr){
  if(!tr) return '';
  return tr.map(x=>x===-1?'<b>host</b>':'<b>s'+x+'</b>').join(' → ');
}
function renderRounds(t, changed){
  const box=document.getElementById('rounds');
  const cur = roundAt(t);
  let html='';
  for(const rd of D.rounds){
    const active = rd.index===cur;
    html += `<div style="margin:7px 0;opacity:${active?1:.62}">
      <div style="font-weight:600;color:${active?'var(--warn)':'var(--txt)'}">round ${rd.index+1}
      <span class="badge dim">${rd.kind}</span>
      <span class="path"> · s${(rd.switches||[]).join(', s')}</span></div>`;
    for(const op of (rd.ops||[]).slice(0,8)) html += `<div class="op ${active?'cur':''}">${op}</div>`;
    if((rd.ops||[]).length>8) html += `<div class="op">… ${rd.ops.length-8} more</div>`;
    html += `</div>`;
  }
  box.innerHTML = html;
}

function showSwitch(s){
  const fr = frames[Math.min(frame, frames.length-1)];
  const rows=[];
  for(const fl of fr.flows){
    const rules = fl.rules;
    if(!rules) continue;
    if(Object.prototype.hasOwnProperty.call(rules, s)){
      const r = rules[s];
      rows.push(`f${fl.fid}: next → ${r===null?'(none)':(r===-1?'host':'s'+r)}`);
    }
  }
  const box=document.getElementById('switchbox');
  box.innerHTML = `<b>s${s}</b><br>` + (rows.length?rows.map(r=>`<div class="op">${r}</div>`).join(''):'<div class="op">no rule for any flow</div>');
}

document.getElementById('play').onclick = ()=>{
  playing = !playing;
  document.getElementById('play').classList.toggle('on', playing);
  document.getElementById('play').textContent = playing? '❚❚ pause':'▶ play';
  if(playing){
    timer = setInterval(()=>{ if(frame>=frames.length-1){ frame=0; } else frame++; render(); }, 650);
  } else clearInterval(timer);
};
document.getElementById('stepf').onclick = ()=>{ frame=Math.min(frames.length-1, frame+1); render(); };
document.getElementById('stepb').onclick = ()=>{ frame=Math.max(0, frame-1); render(); };
scrub.oninput = ()=>{ frame=+scrub.value; render(); };

document.getElementById('title').textContent = D.title;
// allow ?t=12 or #t=12 to open on a specific tick
(function(){
  try{
    const ls = (typeof location!=='undefined') ? (location.hash+location.search) : '';
    const m = ls.match(/[?#&]t=(\d+)/);
    if(m){ frame = Math.max(0, Math.min(frames.length-1, parseInt(m[1],10))); }
  }catch(e){}
})();
render();
</script>
</body></html>
"""


def render_page(data: dict) -> str:
    """Single-file HTML with the replay JSON embedded (no fetch, no CORS)."""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("__DATA__", payload)


@dataclass
class ReplayServer:
    data: dict
    port: int = 0

    def __post_init__(self) -> None:
        self._httpd: Optional[ThreadingHTTPServer] = None

    def _make_handler(self):
        data = self.data

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = render_page(data).encode("utf-8")
                    ctype = "text/html; charset=utf-8"
                elif self.path == "/replay.json":
                    body = json.dumps({"__DATA__": False, **data}).encode("utf-8")
                    ctype = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def start(self, open_browser: bool = True) -> str:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._make_handler())
        port = self._httpd.server_address[1]
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{port}/"
        if open_browser:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        return url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def serve(data: dict, port: int = 0, open_browser: bool = True) -> ReplayServer:
    srv = ReplayServer(data, port)
    url = srv.start(open_browser)
    print(f"replay: {url}")
    return srv


if __name__ == "__main__":  # pragma: no cover
    import sys

    kind = sys.argv[1] if len(sys.argv) > 1 else "swap"
    data = demo(kind)
    page = render_page(data)
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"replay-{kind}.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote", os.path.abspath(out))
