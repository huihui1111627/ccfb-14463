"""Discrete-tick simulator with randomised, uncertain switch apply times.

What is modelled
----------------
* **Ack-gated rounds.**  The controller sends round ``r+1`` only after every
  command of round ``r`` has been acknowledged, plus any drain wait.  Within a
  round, each switch draws its own delivery + apply delay, so commands may take
  effect in **any order** — the property the planner is built around.
* **Atomic per-switch commands.**  When a command fires, all its ops land in
  one table update.
* **Failures.**  A switch can die at a chosen tick.  It then applies nothing
  more and drops what is addressed to it.  The controller computes a verified
  safe retreat from the live state (``retreat.plan_retreat``).
* **Traffic.**  Every flow injects ``demand`` packets per tick at its ingress;
  each packet advances one hop per tick through the rule the switch holds that
  tick.  A packet reaching a switch with no rule is *lost*; one revisiting a
  switch is *looped*.
* **Per-tick audit** with the planner's own invariants (INFLIGHT, HOLDERS,
  bandwidth, table budget).

The baseline (``simulate_baseline``) pushes every command at once with no
ack gating — "一股脑下发" — and is audited identically, so violation-tick counts
are directly comparable.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .model import (
    HOST,
    TAG_OLD,
    nexthop,
    Flow,
    FState,
    Topology,
    Violation,
    audit,
    make_states,
    nm,
    walk,
)
from .planner import Command, Round, apply_op
from .retreat import plan_retreat


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    ticks: int = 0
    finished: bool = False
    make_span: int = 0                 # first send -> last ack (ticks)
    violation_ticks: int = 0
    violations: List[Violation] = field(default_factory=list)
    rounds_sent: int = 0
    failed_switches: List[int] = field(default_factory=list)
    rollback_used: bool = False
    rollback_rounds: int = 0
    rollback_violation_ticks: int = 0
    rollback_exempt: List[int] = field(default_factory=list)
    rollback_unrecoverable: List[int] = field(default_factory=list)
    packets_injected: int = 0
    packets_lost: int = 0
    packets_looped: int = 0
    per_tick: List[dict] = field(default_factory=list)
    trail_trace: List[dict] = field(default_factory=list)

    def kinds(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in self.violations:
            out[v.kind] = out.get(v.kind, 0) + 1
        return out

    def snapshot(self) -> dict:
        return {
            "ticks": self.ticks,
            "finished": self.finished,
            "make_span": self.make_span,
            "violation_ticks": self.violation_ticks,
            "violation_count": len(self.violations),
            "violation_by_kind": self.kinds(),
            "rounds_sent": self.rounds_sent,
            "failed_switches": list(self.failed_switches),
            "rollback_used": self.rollback_used,
            "rollback_rounds": self.rollback_rounds,
            "rollback_violation_ticks": self.rollback_violation_ticks,
            "rollback_exempt": list(self.rollback_exempt),
            "rollback_unrecoverable": list(self.rollback_unrecoverable),
            "packets_injected": self.packets_injected,
            "packets_lost": self.packets_lost,
            "packets_looped": self.packets_looped,
            "violations": [str(v) for v in self.violations[:40]],
        }


# ---------------------------------------------------------------------------
# traffic
# ---------------------------------------------------------------------------


@dataclass
class _Packet:
    pid: int
    fid: int
    at: int
    trail: List[int] = field(default_factory=list)
    tag: Optional[int] = None   # version tag stamped at the ingress


class Traffic:
    def __init__(self, flows: Iterable[Flow]) -> None:
        self.flows = {f.fid: f for f in flows}
        self.packets: List[_Packet] = []
        self.next_pid = 0
        self.injected = 0
        self.lost = 0
        self.looped = 0
        self.arrived = 0

    def step(self, states: Mapping[int, FState], dead: Set[int]) -> None:
        survivors: List[_Packet] = []
        for p in self.packets:
            if p.at in dead:
                self.lost += 1
                continue
            # A packet follows the rule for the tag it was stamped with at
            # injection.  A tag-0 packet that is mid-flight when the ingress
            # flips to tag 1 keeps using the tag-0 rules until it leaves.
            nx = states[p.fid].rule(p.at, p.tag)
            if nx is None:
                self.lost += 1
                continue
            if nx == HOST:
                self.arrived += 1
                continue
            if nx in p.trail or nx in dead:
                self.looped += 1
                continue
            p.trail.append(nx)
            p.at = nx
            if len(p.trail) > 256:
                self.lost += 1
                continue
            survivors.append(p)
        self.packets = survivors
        for fid, f in self.flows.items():
            st = states[fid]
            # Always stamp a concrete version tag.  A plain flow's packets are
            # tag 0 (the "old" version); if the flow is later upgraded to a
            # two-phase move, those packets still follow the old rules, which
            # is exactly what tag 0 means.  A ``None`` tag would be silently
            # reinterpreted as the *current* stamp later, which is wrong.
            tag = st.stamp if st.mode == "tagged" else TAG_OLD
            for _ in range(f.demand):
                self.next_pid += 1
                self.injected += 1
                if f.src in dead:
                    self.lost += 1
                    continue
                self.packets.append(_Packet(self.next_pid, fid, f.src, [f.src], tag))


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    link: Tuple[int, int] = (0, 2)          # controller <-> switch one way
    apply: Tuple[int, int] = (1, 4)         # receive -> table live
    ack: Tuple[int, int] = (0, 2)           # switch -> controller
    drain: int = 5

    def d(self, rng: random.Random, span: Tuple[int, int]) -> int:
        return rng.randint(span[0], span[1])


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------


class RunningPlan:
    """Execute a sequence of rounds ack-gated under uncertain apply order."""

    def __init__(
        self,
        topo: Topology,
        flows: Sequence[Flow],
        rounds: Sequence[Round],
        timing: Timing,
        rng: random.Random,
        drain: int = 5,
    ) -> None:
        self.topo = topo
        self.flows = {f.fid: f for f in flows}
        self.rounds = list(rounds)
        self.timing = timing
        self.rng = rng
        self.drain = drain
        self.states = make_states(list(flows))
        self.dead: Set[int] = set()

    # -- helpers -------------------------------------------------------
    def _apply(self, cmd: Command) -> None:
        for op in cmd.ops:
            apply_op(self.states[op.fid], op)

    def _finalize(self) -> None:
        """Nothing to normalise in the explicit-rule model."""

    # -- main ----------------------------------------------------------
    def run(self, fail_switch: Optional[int] = None, fail_tick: Optional[int] = None) -> SimResult:
        res = SimResult()
        rng = self.rng
        tm = self.timing
        # event heap: (tick, seq, kind, payload)
        heap: List[Tuple[int, int, str, object]] = []
        seq = [0]

        def push(t: int, kind: str, payload) -> None:
            seq[0] += 1
            heapq.heappush(heap, (t, seq[0], kind, payload))

        # -- draw per-command timing for the main plan -----------------
        # each round's commands: apply delay and ack delay, drawn independently
        row_apply: List[List[Tuple[int, int, Command]]] = []  # (apply_off, ack_off, cmd)
        for r in self.rounds:
            row = []
            for cmd in r.commands:
                apply_off = tm.d(rng, tm.link) + tm.d(rng, tm.apply)
                ack_off = apply_off + tm.d(rng, tm.ack)
                row.append((apply_off, ack_off, cmd))
            row_apply.append(row)

        def round_duration(i: int) -> int:
            if not self.rounds[i].commands:
                return 0
            return max(a for _, a, _ in row_apply[i])

        # -- controller gate: send round i at send_tick[i] --------------
        # we schedule sends lazily via events so that drain waits are honoured.
        send_tick = [0] * len(self.rounds)
        gate = 0
        for i, r in enumerate(self.rounds):
            send_tick[i] = gate + int(r.start_wait)
            gate = send_tick[i] + round_duration(i)

        # -- schedule main-plan events ---------------------------------
        for i, row in enumerate(row_apply):
            for (apply_off, ack_off, cmd) in row:
                push(send_tick[i] + apply_off, "apply", (i, cmd))
                push(send_tick[i] + ack_off, "ack", (i, cmd))
        for i, st in enumerate(send_tick):
            push(st, "send", i)

        plan_end = max(
            (send_tick[i] + round_duration(i) for i in range(len(self.rounds))),
            default=0,
        )

        if fail_switch is not None:
            ft = fail_tick if fail_tick is not None else rng.randint(1, max(1, plan_end))
            push(ft, "fail", fail_switch)

        traffic = Traffic(self.flows.values())
        exempt: Set[int] = set()
        retreat_rounds: List[Round] = []
        retreat_scheduled = False
        retreat_last_tick = 0
        last_event_tick = 0
        pending_retreat = [False]
        horizon = max(plan_end + 50, 100) + 600
        t = 0
        finished = False

        def schedule_retreat(rb_rounds: Sequence[Round], now: int) -> int:
            """Chain retreat rounds with ack gating; returns the end tick."""
            gate_t = now
            for r in rb_rounds:
                if not r.commands:
                    continue
                last = gate_t
                for cmd in r.commands:
                    apply_off = tm.d(rng, tm.link) + tm.d(rng, tm.apply)
                    ack_off = apply_off + tm.d(rng, tm.ack)
                    push(gate_t + apply_off, "apply", (1000 + r.index, cmd))
                    push(gate_t + ack_off, "ack", (1000 + r.index, cmd))
                    last = max(last, gate_t + ack_off)
                gate_t = last + 1
            return gate_t

        while t <= horizon:
            # 1. fire everything scheduled for this tick, order: send, fail,
            #    apply, ack (a switch applies before it acks at the same tick)
            fired = []
            while heap and heap[0][0] == t:
                _, _, kind, payload = heapq.heappop(heap)
                fired.append((kind, payload))
            order = {"send": 0, "fail": 1, "apply": 2, "ack": 3}
            fired.sort(key=lambda kp: order.get(kp[0], 9))
            for kind, payload in fired:
                last_event_tick = t
                if kind == "send":
                    res.rounds_sent += 1
                    res.per_tick.append({"t": t, "kind": "send", "round": payload})
                elif kind == "apply":
                    i, cmd = payload
                    if cmd.sw in self.dead:
                        continue
                    self._apply(cmd)
                    res.per_tick.append(
                        {"t": t, "kind": "apply", "round": i, "sw": cmd.sw,
                         "ops": [str(o) for o in cmd.ops]}
                    )
                elif kind == "ack":
                    res.per_tick.append(
                        {"t": t, "kind": "ack", "round": payload[0], "sw": payload[1].sw}
                    )
                elif kind == "fail":
                    sw = payload
                    if sw in self.dead:
                        continue
                    self.dead.add(sw)
                    res.failed_switches = [sw]
                    res.per_tick.append({"t": t, "kind": "fail", "sw": sw})
                    # Commands already delivered will still land even after the
                    # failure is noticed (the switch that has them may be a
                    # different one).  Cancel only the *unsent* rounds, then let
                    # every already-scheduled apply fire, and only afterwards
                    # compute a retreat from the settled state.  Computing the
                    # retreat now would plan against a state that is about to
                    # change under us.
                    kept = []
                    for ev in heap:
                        if ev[2] == "apply":
                            kept.append(ev)
                        elif ev[2] == "ack":
                            kept.append(ev)
                    heap[:] = kept
                    heapq.heapify(heap)
                    pending_retreat[0] = True

            # 1b. once the failure's in-flight commands have drained, compute
            # the retreat from the *settled* state and schedule it.
            if pending_retreat[0]:
                pending_retreat[0] = False
                already_delivered = any(e[2] == "apply" for e in heap)
                if already_delivered:
                    # wait: still commands landing that were delivered before
                    # the failure; check again next tick
                    pending_retreat[0] = True
                else:
                    rb = plan_retreat(
                        self.topo, list(self.flows.values()), self.states, self.dead
                    )
                    exempt = set(rb.exempt) | set(rb.unrecoverable)
                    res.rollback_used = bool(rb.rounds)
                    res.rollback_rounds = len(rb.rounds)
                    res.rollback_exempt = rb.exempt
                    res.rollback_unrecoverable = rb.unrecoverable
                    res.per_tick.append(
                        {"t": t, "kind": "retreat_plan", "rounds": len(rb.rounds),
                         "exempt": rb.exempt, "unrecoverable": rb.unrecoverable,
                         "notes": rb.notes}
                    )
                    retreat_rounds = list(rb.rounds)
                    retreat_last_tick = schedule_retreat(retreat_rounds, t)
                    retreat_scheduled = True

            # 2. traffic
            traffic.step(self.states, self.dead)

            # 3. audit
            vs = audit(
                self.states,
                self.topo,
                strict_hold=True,
                check_capacity=True,
                tick=float(t),
                exempt=exempt,
                failed_switches=self.dead,
            )
            if vs:
                res.violation_ticks += 1
                res.violations.extend(vs)
                if retreat_scheduled:
                    res.rollback_violation_ticks += 1
                res.per_tick.append(
                    {"t": t, "kind": "violation", "items": [str(v) for v in vs[:5]]}
                )
            res.trail_trace.append(self._trace(t))

            # 4. termination
            if not retreat_scheduled:
                if t > plan_end + 3 and not heap:
                    finished = True
                    break
            else:
                if not heap and t >= retreat_last_tick:
                    finished = True
                    break
            t += 1

        res.ticks = t
        res.finished = finished
        res.make_span = last_event_tick
        res.packets_injected = traffic.injected
        res.packets_lost = traffic.lost
        res.packets_looped = traffic.looped
        return res

    def _trace(self, t: int) -> dict:
        out = []
        for fid in sorted(self.states):
            s = self.states[fid]
            tags = []
            for tag in s.live_tags():
                ok, trail, loop_at = walk(self.states, fid, tag)
                tags.append({"tag": tag, "ok": ok, "trail": trail, "loop_at": loop_at})
            primary = tags[0] if tags else {"tag": None, "ok": False, "trail": [], "loop_at": None}
            out.append(
                {
                    "fid": fid,
                    "ok": all(x["ok"] for x in tags),
                    "trail": primary["trail"],
                    "mode": s.mode,
                    "rules": {x: r for x, r in sorted(s.rules.items())},
                    "committed": list(s.committed_path()),
                    "stamp": s.stamp,
                    "tags": tags,
                    "old_inst": sorted(s.old_inst),
                    "new_inst": sorted(s.new_inst),
                }
            )
        return {"t": t, "flows": out}

    @staticmethod
    def _total_span(send_tick: Sequence[int], round_done: Sequence[int]) -> int:
        if not send_tick:
            return 0
        return max(s + d for s, d in zip(send_tick, round_done))


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


def simulate_baseline(
    topo: Topology,
    flows: Sequence[Flow],
    rng: Optional[random.Random] = None,
    max_apply: int = 6,
    traffic: bool = True,
    horizon: int = 60,
) -> SimResult:
    """Push every changed switch's new rule at once, no ack gating."""
    rng = rng or random.Random(0)
    res = SimResult()
    states = make_states(flows)
    tr = Traffic(flows)

    applies: List[Tuple[int, int, Flow]] = []
    for f in flows:
        for x in f.changed_switches:
            applies.append((rng.randint(1, max_apply), x, f))
    applies.sort(key=lambda a: (a[0], a[1]))
    i = 0
    t = 0
    while t <= horizon:
        while i < len(applies) and applies[i][0] <= t:
            _, sw, f = applies[i]
            st = states[f.fid]
            st.set_rule(sw, nexthop(f.new, sw))
            res.per_tick.append({"t": t, "kind": "apply", "sw": sw, "ops": [f"f{f.fid}"]})
            i += 1
        if traffic:
            tr.step(states, set())
        vs = audit(states, topo, strict_hold=True, check_capacity=True, tick=float(t))
        if vs:
            res.violation_ticks += 1
            res.violations.extend(vs)
        frame_flows = []
        for fid, s in sorted(states.items()):
            ok, trail, loop_at = walk(states, fid)
            frame_flows.append(
                {"fid": fid, "ok": ok, "trail": trail, "mode": s.mode,
                 "rules": {x: r for x, r in sorted(s.rules.items())},
                 "tags": [{"tag": None, "ok": ok, "trail": trail, "loop_at": loop_at}]}
            )
        res.trail_trace.append({"t": t, "flows": frame_flows})
        if i >= len(applies) and t > max_apply + 3:
            break
        t += 1
    res.ticks = t
    res.finished = i >= len(applies)
    res.packets_injected = tr.injected
    res.packets_lost = tr.lost
    res.packets_looped = tr.looped
    return res
