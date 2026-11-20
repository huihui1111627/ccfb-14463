"""Planner: flow path changes -> safe, ack-separated rounds.

Execution model
---------------
* A *command* is ``(flow, switch)``: one switch installing / flipping one
  flow's rule.  FlowMods are independent, so two flows' commands on the same
  switch are independent.
* A *round* is a set of commands delivered together.  The controller waits for
  every acknowledgement before starting the next round.  Hence within a round
  the commands may take effect in **any order**.
* Safety therefore requires that *every subset* of a round's commands leaves
  the network safe, not merely the full round.  Equivalently, a round must be
  an antichain of the precedence relation whose prerequisites were all
  committed in earlier rounds (a layer of that DAG).

Given the dependency rules in :mod:`netorch.deps`, every *down-set* of the
precedence DAG is safe; layer ``L`` of a topological rank assignment is exactly
"all prerequisites committed, and no op depends on another op in the same
layer", so its every subset is a down-set.  That is why the planner only has
to build the DAG and colour it.

Pipeline
--------
``per-flow prereqs`` -> ``break cycles (version tags)`` -> ``capacity staging``
-> ``rank layering`` -> ``table budget`` -> ``checkpoint audit``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import deps
from .model import (
    HOST,
    nm,
    TAG_NEW,
    TAG_OLD,
    Flow,
    FState,
    Topology,
    audit,
    make_states,
    nexthop,
    path_links,
)

# ---------------------------------------------------------------------------
# plan description
# ---------------------------------------------------------------------------


def _p(path: Sequence[int]) -> str:
    return "-".join("H" if x == HOST else str(x) for x in path)


@dataclass(frozen=True)
class Op:
    """One table operation for one flow on one switch."""

    kind: str                     # flip | install | stamp | drop
    fid: int
    sw: int
    target: Tuple[int, ...] = ()
    rule: Optional[int] = None    # for flip: the exact next hop to install

    def __str__(self) -> str:
        if self.kind == "flip":
            txt = nm(self.rule) if self.rule is not None else "del"
            return f"f{self.fid}@s{self.sw} rule->{txt}"
        if self.kind == "install":
            return f"f{self.fid}@s{self.sw} install(tag1)->{_p(self.target)}"
        if self.kind == "stamp":
            return f"f{self.fid}@s{self.sw} stamp=tag1"
        return f"f{self.fid}@s{self.sw} drop(tag0)"

    def snapshot(self) -> dict:
        return {
            "kind": self.kind,
            "fid": self.fid,
            "sw": self.sw,
            "target": list(self.target),
            "rule": self.rule,
            "text": str(self),
        }


@dataclass
class Command:
    """Everything one switch applies atomically in one round."""

    sw: int
    ops: List[Op]

    def snapshot(self) -> dict:
        return {"sw": self.sw, "ops": [o.snapshot() for o in self.ops]}


@dataclass
class Round:
    index: int
    commands: List[Command]
    kind: str = "update"          # update | stamp | drop | noop
    label: str = ""
    # Extra controller-side wait (in ticks) *before* sending this round, on top
    # of the acknowledgement gate of the previous round.  This is the drain
    # window that lets packets already inside the network leave before any
    # rule they depend on is deleted.
    start_wait: float = 0.0

    @property
    def switches(self) -> List[int]:
        return sorted({c.sw for c in self.commands})

    @property
    def ops(self) -> List[Op]:
        return [o for c in self.commands for o in c.ops]

    def deletes(self) -> bool:
        return any(
            o.kind == "drop" or (o.kind == "flip" and o.rule is None)
            for o in self.ops
        )

    def snapshot(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "label": self.label,
            "start_wait": self.start_wait,
            "deletes": self.deletes(),
            "switches": self.switches,
            "commands": [c.snapshot() for c in self.commands],
        }


@dataclass
class PlanStats:
    rounds: int = 0
    rounds_with_writes: int = 0
    # theoretical lower bounds
    dag_height: int = 0            # precedence chain length (ordering LB)
    capacity_lb: int = 0           # clique/serialisation LB
    tagged_lb: int = 0             # 3 rounds per tagged flow
    lower_bound: int = 0           # max of the above
    # contents
    switches_touched: int = 0
    ops: int = 0
    tagged_flows: List[int] = field(default_factory=list)
    detoured_flows: List[int] = field(default_factory=list)
    capacity_stage_rounds: int = 0
    # tables
    peak_table: int = 0
    peak_table_switch: int = -1
    table_cap: int = 0
    table_deficit: int = 0
    table_detail: Dict[int, int] = field(default_factory=dict)
    infeasible: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def gap(self) -> int:
        return self.rounds - self.lower_bound

    def snapshot(self) -> dict:
        return {
            "rounds": self.rounds,
            "rounds_with_writes": self.rounds_with_writes,
            "gap": self.gap,
            "dag_height": self.dag_height,
            "capacity_lb": self.capacity_lb,
            "tagged_lb": self.tagged_lb,
            "lower_bound": self.lower_bound,
            "switches_touched": self.switches_touched,
            "ops": self.ops,
            "tagged_flows": list(self.tagged_flows),
            "detoured_flows": list(self.detoured_flows),
            "capacity_stage_rounds": self.capacity_stage_rounds,
            "peak_table": self.peak_table,
            "peak_table_switch": self.peak_table_switch,
            "table_cap": self.table_cap,
            "table_deficit": self.table_deficit,
            "table_detail": {str(k): v for k, v in self.table_detail.items()},
            "infeasible": self.infeasible,
            "notes": list(self.notes),
        }


@dataclass
class Plan:
    rounds: List[Round]
    stats: PlanStats
    flows: Dict[int, Flow]
    initial: Dict[int, FState]
    final: Dict[int, Tuple[int, ...]]
    checkpoints: List[Dict[int, FState]] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "stats": self.stats.snapshot(),
            "rounds": [r.snapshot() for r in self.rounds],
            "flows": {
                fid: {
                    "src": f.src,
                    "dst": f.dst,
                    "old": list(f.old),
                    "new": list(f.new),
                    "demand": f.demand,
                }
                for fid, f in self.flows.items()
            },
            "final": {fid: list(p) for fid, p in self.final.items()},
            "checkpoints": [
                {fid: s.snapshot() for fid, s in cp.items()} for cp in self.checkpoints
            ],
        }


# ---------------------------------------------------------------------------
# per-flow operation programs
# ---------------------------------------------------------------------------


@dataclass
class FlowProgram:
    fid: int
    tagged: bool
    depth: int                        # number of sequential rounds this flow needs
    stages: Dict[int, List[Op]]       # stage -> ops (one stage per round)

    def max_stage(self) -> int:
        return max(self.stages) if self.stages else -1


def plain_program(f: Flow) -> FlowProgram:
    pre = deps.flow_prereqs(f)
    if not pre:
        return FlowProgram(f.fid, False, 0, {})
    edges = [(b, t) for t, bs in pre.items() for b in bs]
    stages: Dict[int, List[Op]] = {}

    def emit(sw: int, stage: int) -> None:
        stages.setdefault(stage, []).append(Op("flip", f.fid, sw, f.new, nexthop(f.new, sw)))

    if deps.has_cycle(list(pre), edges):
        for k, sw in enumerate(sorted(pre)):
            emit(sw, k)
        return FlowProgram(f.fid, False, len(stages), stages)
    r = deps.ranks(list(pre), edges)
    for sw, rank in r.items():
        emit(sw, rank)
    return FlowProgram(f.fid, False, max(r.values()) + 1, stages)


def tagged_program(f: Flow) -> FlowProgram:
    return FlowProgram(
        f.fid,
        True,
        3,
        {
            0: [Op("install", f.fid, x, f.new) for x in f.new],
            1: [Op("stamp", f.fid, f.src, f.new)],
            2: [Op("drop", f.fid, x, f.new) for x in f.old],
        },
    )


def flow_program(f: Flow, tagged: bool) -> FlowProgram:
    return tagged_program(f) if tagged else plain_program(f)


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


class Planner:
    """Compute a safe, round-minimal plan.  See the module docstring."""

    def __init__(
        self,
        topo: Topology,
        flows: Sequence[Flow],
        drain_ticks: float = 5.0,
        strategy: str = "auto",          # auto | tag | detour
        allow_tagging: bool = True,
        allow_detour: bool = True,
        table_cap_override: Optional[int] = None,
    ) -> None:
        self.topo = topo
        self.flows = list(flows)
        self.by_id = {f.fid: f for f in self.flows}
        self.drain_ticks = drain_ticks
        self.strategy = strategy
        self.allow_tagging = allow_tagging and strategy != "detour"
        self.allow_detour = allow_detour and strategy != "tag"
        self.table_cap_override = table_cap_override

    # ==================================================================
    def plan(self) -> Plan:
        stats = PlanStats()
        states = make_states(self.flows)
        final = {f.fid: f.new for f in self.flows}
        active = [f for f in self.flows if f.old != f.new]
        if not active:
            stats.notes.append("nothing to update: every flow is already on its new path")
            return Plan([], stats, self.by_id, states, final, [states])

        ok, why = self._target_feasible(active)
        if not ok:
            stats.infeasible = why
            stats.notes.append("target configuration is infeasible: " + why)
            return Plan([], stats, self.by_id, states, final, [states])

        # -- 1. per-flow precedence, then break cycles with version tags --
        per_flow = {f.fid: deps.flow_prereqs(f) for f in active}
        tagged = self._break_cycles(per_flow, active, stats)
        programs = {f.fid: flow_program(f, f.fid in tagged) for f in active}

        # -- 2. capacity staging -----------------------------------------
        start, blocked = self._stage(active, programs, stats)
        if start is None:
            if not self.allow_detour:
                stats.notes.extend(blocked)
                stats.infeasible = "capacity deadlock and detours are disabled"
                return Plan([], stats, self.by_id, states, final, [states])
            detoured = self._plan_with_detour(active, programs, tagged, stats, blocked)
            if detoured is None:
                stats.notes.extend(blocked)
                stats.infeasible = "capacity deadlock could not be resolved"
                return Plan([], stats, self.by_id, states, final, [states])
            return detoured

        # -- 3. layering --------------------------------------------------
        rounds = self._rounds(active, programs, start, tagged)
        stats.rounds = len(rounds)
        stats.rounds_with_writes = sum(1 for r in rounds if r.commands)
        stats.ops = sum(len(r.ops) for r in rounds)
        stats.switches_touched = len({c.sw for r in rounds for c in r.commands})
        stats.tagged_flows = sorted(tagged)
        stats.capacity_stage_rounds = max((start[f.fid] for f in active), default=0)

        # -- 4. lower bounds ---------------------------------------------
        self._lower_bounds(active, programs, tagged, per_flow, stats)

        # -- 5. table budget ---------------------------------------------
        self._table_budget(tagged, stats)

        # -- 6. checkpoints + audit --------------------------------------
        cps = self._checkpoints(rounds)
        self._verify(cps, stats)
        if stats.table_deficit and tagged:
            # Two-phase rules do not fit.  A version-tagged plan is the only
            # one-phase-order-free way to break the cycle, so there is no safe
            # plan that fits the budget; say so instead of silently exceeding it.
            stats.infeasible = (
                f"two-phase rules need {stats.peak_table} entries on switch "
                f"s{stats.peak_table_switch} but only {stats.table_cap} are available"
            )
            stats.notes.append(
                "table budget exceeded: the dependency cycle cannot be broken "
                "without two-phase rules, which do not fit"
            )
            return Plan([], stats, self.by_id, states, final, [states])
        return Plan(rounds, stats, self.by_id, states, final, cps)

    # ==================================================================
    # feasibility
    # ==================================================================
    def _target_feasible(self, active: Sequence[Flow]) -> Tuple[bool, str]:
        moving = {f.fid for f in active}
        load: Dict[Tuple[int, int], int] = {}
        for f in self.flows:
            path = f.new if f.fid in moving else f.old
            for l in path_links(path):
                load[l] = load.get(l, 0) + f.demand
        for (a, b), v in sorted(load.items()):
            if not self.topo.has_link(a, b):
                return False, f"link s{a}-s{b} is not in the topology"
            if v > self.topo.cap(a, b):
                return False, f"link s{a}-s{b} would carry {v} > capacity {self.topo.cap(a, b)}"
        return True, ""

    # ==================================================================
    # cycle breaking
    # ==================================================================
    def _break_cycles(
        self,
        per_flow: Mapping[int, Mapping[int, Set[int]]],
        active: Sequence[Flow],
        stats: PlanStats,
    ) -> Set[int]:
        tagged: Set[int] = set()

        def components(tag_set):
            pre = self._union(per_flow, tag_set)
            edges = [(b, t) for t, bs in pre.items() for b in bs]
            return [c for c in deps.sccs(list(pre), edges) if len(c) > 1]

        if not self.allow_tagging:
            if components(tagged):
                stats.notes.append(
                    "dependency cycles exist but tagging is disabled; no safe "
                    "one-phase plan exists for this task"
                )
            return tagged

        while True:
            comps = components(tagged)
            if not comps:
                return tagged
            comp_of = {n: i for i, c in enumerate(comps) for n in c}
            # pick the flow that owns the most intra-cycle edges per unit cost
            best, best_score = None, 0.0
            for f in active:
                if f.fid in tagged:
                    continue
                pf = per_flow[f.fid]
                intra = sum(
                    1
                    for t, bs in pf.items()
                    if t in comp_of
                    for b in bs
                    if comp_of.get(b) == comp_of[t]
                )
                if intra == 0:
                    continue
                cost = 1 + len(f.new) + len(f.old)   # table entries paid
                score = intra / cost
                if score > best_score:
                    best, best_score = f.fid, score
            if best is None:
                inside = {n for c in comps for n in c}
                extra = {
                    f.fid for f in active
                    if f.fid not in tagged and set(per_flow[f.fid]) & inside
                }
                if not extra:
                    stats.notes.append(
                        "WARNING: a cycle could not be attributed to any flow"
                    )
                    return tagged
                tagged |= extra
                stats.notes.append(
                    f"flows {sorted(extra)} promoted to two-phase: they share a "
                    "cycle spread across flows"
                )
                continue
            tagged.add(best)
            stats.notes.append(
                f"flow {best} promoted to two-phase (version tags): it closes a "
                "dependency cycle, so no one-phase order exists"
            )

    @staticmethod
    def _union(
        per_flow: Mapping[int, Mapping[int, Set[int]]], tagged: Set[int]
    ) -> Dict[int, Set[int]]:
        out: Dict[int, Set[int]] = {}
        for fid, pre in per_flow.items():
            if fid in tagged:
                continue
            for t, bs in pre.items():
                out.setdefault(t, set()).update(bs)
                out.setdefault(t, set())
        return out

    # ==================================================================
    # capacity staging
    # ==================================================================
    def _fixed_load(self, moving: Set[int]) -> Dict[Tuple[int, int], int]:
        fixed: Dict[Tuple[int, int], int] = {}
        for f in self.flows:
            if f.fid in moving:
                continue
            for l in path_links(f.old):
                fixed[l] = fixed.get(l, 0) + f.demand
        return fixed

    def _stage(
        self,
        active: Sequence[Flow],
        programs: Mapping[int, FlowProgram],
        stats: PlanStats,
    ) -> Tuple[Optional[Dict[int, int]], List[str]]:
        """Greedy scheduler: when may each flow start its move?

        A moving flow reserves its **worst-case footprint** ``old ∪ new`` for
        every round it is in flight (a held old rule and a new rule may both be
        live).  A flow may start only if adding that footprint stays within
        every link's capacity.  If nothing can start and nothing is running,
        the instance has a capacity deadlock.
        """
        by_id = {f.fid: f for f in active}
        fixed = self._fixed_load(set(by_id))
        # Start the most contended flows first: a flow whose footprint crosses
        # the busiest links is hardest to place, so scheduling it while the
        # fabric is emptiest avoids leaving it for a late wave.
        footprint: Dict[Tuple[int, int], int] = {}
        for f in active:
            for l in path_links(f.old) | path_links(f.new):
                footprint[l] = footprint.get(l, 0) + f.demand
        contention = {
            f.fid: -sum(
                min(footprint[l], self.topo.cap(*l))
                for l in path_links(f.old) | path_links(f.new)
            )
            for f in active
        }
        order = sorted(active, key=lambda f: (contention[f.fid], f.fid))
        start: Dict[int, int] = {}
        finish: Dict[int, int] = {}
        pending = set(by_id)
        reasons: List[str] = []
        t = 0
        guard = 0

        while pending:
            guard += 1
            if guard > 10000:
                return None, ["capacity staging did not converge"]
            running = {fid: d for fid, d in finish.items() if d > t}
            # load from running flows (post-move reservations are their new path)
            load = dict(fixed)
            for fid, f in by_id.items():
                if fid in running:
                    for l in path_links(f.old) | path_links(f.new):
                        load[l] = load.get(l, 0) + f.demand
                elif fid in finish:
                    for l in path_links(f.new):
                        load[l] = load.get(l, 0) + f.demand
                elif fid in pending:
                    for l in path_links(f.old):
                        load[l] = load.get(l, 0) + f.demand
            started_any = False
            for f in order:
                fid = f.fid
                if fid not in pending:
                    continue
                trial = dict(load)
                # replace this flow's old-only reservation with old ∪ new
                for l in path_links(f.old):
                    trial[l] = trial[l] - f.demand
                for l in path_links(f.old) | path_links(f.new):
                    trial[l] = trial.get(l, 0) + f.demand
                bad = [(l, v) for l, v in trial.items() if v > self.topo.cap(*l)]
                if bad:
                    if t == 0:
                        l, v = bad[0]
                        reasons.append(
                            f"flow {fid}: starting now would put {v} > "
                            f"{self.topo.cap(*l)} on link s{l[0]}-s{l[1]}"
                        )
                    continue
                load = trial
                start[fid] = t
                depth = max(1, programs[fid].depth)
                finish[fid] = t + depth
                pending.discard(fid)
                started_any = True
            if not pending:
                break
            if started_any:
                continue
            upcoming = [d for d in finish.values() if d > t]
            if upcoming:
                t = min(upcoming)
                continue
            return None, reasons or [
                f"capacity deadlock among flows {sorted(pending)}"
            ]
        return start, reasons

    # ==================================================================
    # detour (borrow a transit path to break a capacity deadlock)
    # ==================================================================
    def _plan_with_detour(
        self,
        active: Sequence[Flow],
        programs: Mapping[int, FlowProgram],
        tagged: Set[int],
        stats: PlanStats,
        blocked: Sequence[str],
    ) -> Optional[Plan]:
        """Break a capacity deadlock by parking one flow on a transit path.

        A deadlock means every flow that could free a contended link is itself
        waiting for another.  We break it by moving one flow onto a spare
        transit path *first*; once it has left the contended link, the rest of
        the flows stage normally, and finally the detoured flow takes its new
        path.  The plan is concatenated and re-audited checkpoint by
        checkpoint, so an unhelpful detour is rejected rather than shipped.
        """
        by_id = {f.fid: f for f in active}
        fixed = self._fixed_load(set(by_id))
        candidates = sorted(by_id, key=lambda k: (by_id[k].demand, k))
        all_ids = {f.fid for f in self.flows}
        for fid in candidates:
            f = by_id[fid]
            for transit in self._transit_candidates(f, fixed):
                if transit == f.old or transit == f.new or len(transit) < 2:
                    continue
                # stage A: move this flow old -> transit, everyone else parked
                park_flow = Flow(f.fid, f.src, f.dst, f.old, transit, f.demand)
                rest = [g for g in self.flows if g.fid != fid]
                parked_rest = [
                    Flow(g.fid, g.src, g.dst, g.old, g.old, g.demand) for g in rest
                ]
                pA = Planner(
                    self.topo, [park_flow] + parked_rest,
                    drain_ticks=self.drain_ticks, allow_tagging=self.allow_tagging,
                    allow_detour=False, table_cap_override=self.table_cap_override,
                ).plan()
                if pA.stats.infeasible:
                    continue
                # stage B: everyone else moves, this flow stays on transit
                moved = [
                    Flow(g.fid, g.src, g.dst, g.old, g.new, g.demand) for g in rest
                ]
                stay = Flow(f.fid, f.src, f.dst, transit, transit, f.demand)
                pB = Planner(
                    self.topo, [stay] + moved,
                    drain_ticks=self.drain_ticks, allow_tagging=self.allow_tagging,
                    allow_detour=True, table_cap_override=self.table_cap_override,
                ).plan()
                if pB.stats.infeasible:
                    continue
                # stage C: the detoured flow transit -> new, alone
                pC = Planner(
                    self.topo, [Flow(f.fid, f.src, f.dst, transit, f.new, f.demand)],
                    drain_ticks=self.drain_ticks, allow_tagging=self.allow_tagging,
                    allow_detour=False, table_cap_override=self.table_cap_override,
                ).plan()
                if pC.stats.infeasible:
                    continue
                rounds = self._concat(pA.rounds, pB.rounds, pC.rounds)
                cps = self._checkpoints(rounds)
                if self._has_violation(cps):
                    continue
                self._fill_detour_stats(stats, pA, pB, pC, rounds, fid, transit)
                stats.notes.append(
                    f"capacity deadlock broken: flow {fid} was parked on a temporary "
                    f"transit path via {_p(transit)} while the others moved, then "
                    "took its new path"
                )
                stats.notes.extend(blocked)
                return Plan(
                    rounds, stats, self.by_id, make_states(self.flows),
                    {fl.fid: fl.new for fl in self.flows}, cps,
                )
        return None

    def _fill_detour_stats(self, stats: PlanStats, pA, pB, pC, rounds, fid, transit) -> None:
        stats.rounds = len(rounds)
        stats.rounds_with_writes = sum(1 for r in rounds if r.commands)
        stats.ops = sum(len(r.ops) for r in rounds)
        stats.switches_touched = len({c.sw for r in rounds for c in r.commands})
        stats.tagged_flows = sorted(
            set(pA.stats.tagged_flows) | set(pB.stats.tagged_flows) | set(pC.stats.tagged_flows)
        )
        stats.detoured_flows = [fid]
        # A detour costs at least: the detoured flow's own two legs plus the
        # best-case time for the rest.  Each leg is at least one round.
        stats.dag_height = pA.stats.dag_height + pC.stats.dag_height
        stats.capacity_lb = pB.stats.capacity_lb
        stats.tagged_lb = pB.stats.tagged_lb
        stats.lower_bound = max(
            2 + pB.stats.lower_bound,
            pA.stats.lower_bound + pB.stats.lower_bound + pC.stats.lower_bound,
        )
        stats.capacity_stage_rounds = (
            pA.stats.capacity_stage_rounds
            + pB.stats.capacity_stage_rounds
            + pC.stats.capacity_stage_rounds
        )
        peak, peak_sw, cap, detail = (0, -1, 0, {})
        for p in (pA, pB, pC):
            if p.stats.peak_table > peak:
                peak, peak_sw = p.stats.peak_table, p.stats.peak_table_switch
                cap, detail = p.stats.table_cap, p.stats.table_detail
        stats.peak_table, stats.peak_table_switch = peak, peak_sw
        stats.table_cap, stats.table_detail = cap, detail
        stats.table_deficit = max(0, peak - cap) if cap else 0

    def _transit_candidates(
        self, f: Flow, fixed: Mapping[Tuple[int, int], int], limit: int = 8
    ) -> List[Tuple[int, ...]]:
        """Spare-capacity src..last-hop paths other than the new path, best first."""
        src, last = f.src, f.new[-1]
        found: List[Tuple[Tuple[int, ...], int]] = []
        pq: List[Tuple[int, int, Tuple[int, ...], int]] = [(0, 0, (src,), 1 << 30)]
        seen: Dict[int, int] = {}
        guard = 0
        while pq and guard < 5000:
            guard += 1
            hops, _, path, bottleneck = heapq.heappop(pq)
            node = path[-1]
            if node == last and len(path) >= 2:
                if path != f.new:
                    found.append((path, bottleneck))
                    if len(found) >= limit:
                        break
                continue
            prev = seen.get(node)
            if prev is not None and prev >= bottleneck:
                continue
            seen[node] = bottleneck
            for w in self.topo.neighbours(node):
                if w in path:
                    continue
                l = (min(node, w), max(node, w))
                spare = self.topo.cap(*l) - fixed.get(l, 0) - f.demand
                if spare < 0:
                    continue
                nb = min(bottleneck, spare)
                heapq.heappush(pq, (hops + 1, -nb, path + (w,), nb))
        found.sort(key=lambda p: (-p[1], len(p[0])))
        return [p for p, _ in found]

    @staticmethod
    def _concat(*groups: Sequence[Round]) -> List[Round]:
        out: List[Round] = []
        for group in groups:
            for r in group:
                out.append(
                    Round(
                        index=len(out),
                        commands=r.commands,
                        kind=r.kind,
                        label=r.label,
                        start_wait=r.start_wait,
                    )
                )
        return out

    def _has_violation(self, cps: Sequence[Mapping[int, FState]]) -> bool:
        for i, cp in enumerate(cps):
            if audit(cp, self.topo, strict_hold=True, check_capacity=True, tick=float(i)):
                return True
        return False

    # ==================================================================
    # layering
    # ==================================================================
    def _rounds(
        self,
        active: Sequence[Flow],
        programs: Mapping[int, FlowProgram],
        start: Mapping[int, int],
        tagged: Set[int],
    ) -> List[Round]:
        """Place every op at ``start[flow] + stage``; collect per-round commands.

        A plain flow may skip a stage whose ops would be empty, but stages must
        stay strictly ordered, so we compact each flow's non-empty stages to
        consecutive offsets.  This keeps the round count tight when the
        capacity scheduler delayed a flow.
        """
        sched: Dict[int, Dict[int, int]] = {}
        for f in active:
            prog = programs[f.fid]
            base = start.get(f.fid, 0)
            offsets: Dict[int, int] = {}
            used = 0
            for stage in sorted(prog.stages):
                if not prog.stages[stage]:
                    continue
                offsets[stage] = base + used
                used += 1
            sched[f.fid] = offsets

        last = max((max(v.values()) for v in sched.values() if v), default=-1)
        rounds: List[Round] = []
        for t in range(last + 1):
            cmds: Dict[int, List[Op]] = {}
            for fid, st in sched.items():
                prog = programs[fid]
                for stage, rnd in st.items():
                    if rnd != t:
                        continue
                    for op in prog.stages.get(stage, ()):
                        cmds.setdefault(op.sw, []).append(op)
            if not cmds:
                continue
            kinds = {op.kind for ops in cmds.values() for op in ops}
            kind = (
                "stamp" if kinds == {"stamp"}
                else "drop" if kinds == {"drop"}
                else "update"
            )
            # A round that *deletes* rules must be preceded by a drain: any
            # packet already inside the network needs one hop per switch to
            # leave, and the dependency ordering guarantees no new packet will
            # arrive at a deleting switch, so waiting is sufficient.
            drain = 0.0
            if kinds & {"drop"} or any(
                op.kind == "flip" and op.rule is None
                for ops in cmds.values() for op in ops
            ):
                longest = max(
                    (len(self.by_id[op.fid].old) + len(self.by_id[op.fid].new)
                     for ops in cmds.values() for op in ops),
                    default=2,
                )
                drain = float(max(self.drain_ticks, longest + 2))
            rounds.append(
                Round(
                    index=len(rounds),
                    commands=[Command(sw, ops) for sw, ops in sorted(cmds.items())],
                    kind=kind,
                    label=f"round {len(rounds) + 1}",
                    start_wait=drain,
                )
            )
        return rounds

    # ==================================================================
    # lower bounds
    # ==================================================================
    def _lower_bounds(
        self,
        active: Sequence[Flow],
        programs: Mapping[int, FlowProgram],
        tagged: Set[int],
        per_flow: Mapping[int, Mapping[int, Set[int]]],
        stats: PlanStats,
    ) -> None:
        """Sound lower bounds on the number of rounds.

        The plan's operations are per **(flow, switch)** and two flows' ops on
        the same switch are independent FlowMods, so the only edges in the
        operation DAG are (a) a flow's own ordering constraints and (b) the
        capacity staging that serialises whole flows.  Therefore:

        * every flow needs at least its own chain depth (``flow_lb``);
        * a set of flows whose worst-case footprints cannot coexist on some
          link must move strictly one after another, so at least the sum of
          their depths (``capacity_lb``, the weight of the heaviest clique).

        The lower bound is the larger of the two.  The plan itself may need
        more rounds because of how the staging interleaves, and the difference
        is the reported gap.
        """
        fids = [f.fid for f in active]
        # per-flow depth
        own: Dict[int, int] = {}
        for fid in fids:
            prog = programs[fid]
            own[fid] = max(1, prog.depth) if prog.stages else 1
        flow_lb = max(own.values(), default=0)

        # capacity clique (mutual exclusion) using each flow's own depth
        by_id = {f.fid: f for f in active}
        fixed = self._fixed_load(set(by_id))
        conflict: Dict[int, Set[int]] = {fid: set() for fid in fids}
        for i, a in enumerate(fids):
            fa = by_id[a]
            fa_links = path_links(fa.old) | path_links(fa.new)
            for b in fids[i + 1:]:
                fb = by_id[b]
                fb_links = path_links(fb.old) | path_links(fb.new)
                for l in fa_links & fb_links:
                    if fixed.get(l, 0) + fa.demand + fb.demand > self.topo.cap(*l):
                        conflict[a].add(b)
                        conflict[b].add(a)
                        break
        capacity_lb = _weighted_clique_weight(fids, conflict, own)
        # Link-round cut bound: a flow whose old-or-new path crosses link l is
        # in flight for at least ``own[fid]`` rounds, and its old ∪ new
        # footprint reserves l for every one of those rounds.  So l must carry
        # sum(demand · rounds) "link-rounds" of work, of which it can do ``cap``
        # per round.  This is sound and much tighter than a clique bound on
        # dense fabrics (many flows, few tight links).
        cut_lb = 0
        per_link: Dict[Tuple[int, int], int] = {}
        for f in active:
            rounds_needed = own[f.fid] * f.demand
            for l in path_links(f.old) | path_links(f.new):
                per_link[l] = per_link.get(l, 0) + rounds_needed
        for l, work in per_link.items():
            cap = self.topo.cap(*l)
            if cap > 0:
                cut_lb = max(cut_lb, -(-work // cap))
        capacity_lb = max(capacity_lb, cut_lb)

        stats.dag_height = flow_lb
        stats.capacity_lb = capacity_lb
        stats.tagged_lb = 3 if tagged else 0
        stats.lower_bound = max(flow_lb, capacity_lb, 1)
        if stats.rounds > stats.lower_bound:
            stats.notes.append(
                f"{stats.rounds} rounds vs lower bound {stats.lower_bound}: the "
                "extra rounds are where flow moves had to be interleaved by capacity"
            )

    # ==================================================================
    # table budget
    # ==================================================================
    def _table_budget(self, tagged: Set[int], stats: PlanStats) -> None:
        """Peak table entries per switch.

        A flow on its final plain path occupies one entry per switch.  A flow
        being updated **two-phase** occupies two entries on every switch of
        ``old ∪ new`` while both tags are live (the old rule and the tag-1
        rule).  This peak is what decides whether the two-phase rule set fits;
        it is the number the user asked to "算清楚".
        """
        per_switch: Dict[int, int] = {}
        dual_per_switch: Dict[int, int] = {}
        for f in self.flows:
            if f.fid in tagged:
                for x in set(f.old) | set(f.new):
                    per_switch[x] = per_switch.get(x, 0) + 1
                    if x in f.old and x in f.new:
                        # both rules live at the same switch during the update
                        per_switch[x] += 1
                        dual_per_switch[x] = dual_per_switch.get(x, 0) + 1
            else:
                for x in set(f.old) | set(f.new):
                    per_switch[x] = per_switch.get(x, 0) + 1
        stats.table_detail = dict(sorted(per_switch.items()))
        if not per_switch:
            return
        sw = max(per_switch, key=lambda x: (per_switch[x], -x))
        stats.peak_table = per_switch[sw]
        stats.peak_table_switch = sw
        stats.table_cap = (
            self.table_cap_override
            if self.table_cap_override is not None
            else self.topo.table_cap.get(sw, 1 << 20)
        )
        stats.table_deficit = max(0, stats.peak_table - stats.table_cap)
        if stats.table_deficit:
            stats.notes.append(
                f"table budget: switch s{sw} needs {stats.peak_table} entries but has "
                f"{stats.table_cap}; the two-phase rules do not fit there"
            )
            if dual_per_switch.get(sw):
                stats.notes.append(
                    f"  {dual_per_switch[sw]} of those are the second (tag-1) rule "
                    "of a flow that is being updated two-phase"
                )

    # ==================================================================
    # checkpoints
    # ==================================================================
    def _checkpoints(self, rounds: Sequence[Round]) -> List[Dict[int, FState]]:
        states = make_states(self.flows)
        out = [self._clone(states)]
        for r in rounds:
            for cmd in r.commands:
                for op in cmd.ops:
                    apply_op(states[op.fid], op)
            out.append(self._clone(states))
        return out

    @staticmethod
    def _clone(states: Mapping[int, FState]) -> Dict[int, FState]:
        return {fid: s.clone() for fid, s in states.items()}

    def _verify(self, cps: Sequence[Mapping[int, FState]], stats: PlanStats) -> None:
        for i, cp in enumerate(cps):
            vs = audit(cp, self.topo, strict_hold=True, check_capacity=True, tick=float(i))
            if vs:
                stats.notes.append(
                    f"INTERNAL: checkpoint {i} is unsafe: "
                    + "; ".join(str(v) for v in vs[:4])
                )


def apply_op(st: FState, op: Op) -> None:
    """Apply one table op to a flow's state."""
    if op.kind == "flip":
        if st.mode != "plain":
            raise RuntimeError(f"flip on a tagged flow {op.fid}")
        rule = op.rule if op.rule is not None else nexthop(op.target, op.sw)
        st.set_rule(op.sw, rule)
    elif op.kind == "install":
        if st.mode != "tagged":
            st.begin_tagged(op.target)
        st.install_tagged(op.sw)
    elif op.kind == "stamp":
        if st.mode != "tagged":
            st.begin_tagged(op.target)
        # A switch rejects a stamp change that references a tag-1 path which is
        # not fully installed (e.g. because an install was lost with a failed
        # switch).  Model that as a no-op rather than a crash: the flow simply
        # stays on the old tag, which is safe.
        if st.new_inst >= set(st.t_to):
            st.flip_stamp()
    elif op.kind == "drop":
        if st.mode == "tagged":
            st.drop_old()
    else:  # pragma: no cover
        raise RuntimeError(f"unknown op {op.kind}")


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


@dataclass
class RollbackResult:
    plan: Optional[Plan]
    exempt_flows: List[int]
    unrecoverable: List[int]
    notes: List[str]

    def snapshot(self) -> dict:
        return {
            "plan": self.plan.snapshot() if self.plan else None,
            "exempt_flows": list(self.exempt_flows),
            "unrecoverable": list(self.unrecoverable),
            "notes": list(self.notes),
        }


def rollback_plan(
    topo: Topology,
    flows: Sequence[Flow],
    current: Mapping[int, FState],
    failed: Iterable[int],
    drain_ticks: float = 5.0,
) -> RollbackResult:
    """Plan a safe retreat to every flow's old path, avoiding failed switches.

    * The failed switches are removed from the topology: no retreat may use them.
    * A flow whose *current* committed path crosses a failed switch is already
      dropping packets; it is reported as exempt rather than as the retreat's
      fault.
    * A flow whose old path crosses a failed switch cannot be restored; it is
      reported as unrecoverable with the reason.
    """
    failed = set(failed)
    notes: List[str] = []
    exempt: List[int] = []
    unrecoverable: List[int] = []
    survivors = topo.without(failed)

    retreat: List[Flow] = []
    for f in flows:
        st = current[f.fid]
        here = st.committed_path()
        if set(here) & failed:
            exempt.append(f.fid)
            notes.append(
                f"flow {f.fid}: committed path {_p(here)} crosses the failed switch; "
                "packets are already dropping, so it is exempt from the retreat audit"
            )
            continue
        if set(f.old) & failed:
            unrecoverable.append(f.fid)
            notes.append(
                f"flow {f.fid}: its old path {_p(f.old)} crosses the failed switch, "
                "so it cannot be restored"
            )
            continue
        if here != f.old:
            retreat.append(Flow(f.fid, f.src, f.dst, here, f.old, f.demand))

    if not retreat:
        return RollbackResult(None, exempt, unrecoverable, notes)

    p = Planner(
        survivors,
        retreat,
        drain_ticks=drain_ticks,
        strategy="auto",
        allow_tagging=True,
        allow_detour=True,
    ).plan()
    if p.stats.infeasible:
        notes.append("retreat is infeasible: " + p.stats.infeasible)
        return RollbackResult(None, exempt, unrecoverable, notes)
    notes.append(
        f"retreat: {p.stats.rounds} rounds to get every surviving flow back onto "
        "its old path"
    )
    return RollbackResult(p, exempt, unrecoverable, notes)


def _weighted_clique_weight(
    fids: Sequence[int], conflict: Mapping[int, Set[int]], weight: Mapping[int, int]
) -> int:
    """Maximum total weight of a clique in the exclusion graph."""
    best = 0
    if len(fids) <= 22:
        def bk(R: Set[int], P: Set[int], X: Set[int]) -> None:
            nonlocal best
            if not P and not X:
                best = max(best, sum(weight[f] for f in R))
                return
            pivot = max(P | X, key=lambda u: len(conflict[u] & P), default=None)
            cand = P - (conflict[pivot] if pivot is not None else set())
            for v in list(cand):
                bk(R | {v}, P & conflict[v], X & conflict[v])
                P = P - {v}
                X = X | {v}
        bk(set(), set(fids), set())
    else:
        order = sorted(fids, key=lambda u: -len(conflict[u]))
        clique: Set[int] = set()
        for v in order:
            if all(v in conflict[u] for u in clique):
                clique.add(v)
        best = sum(weight[f] for f in clique)
    return best
