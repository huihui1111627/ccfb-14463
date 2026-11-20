"""Topology, flows, per-flow forwarding state and the instantaneous audit.

Forwarding model
----------------
A *flow* has an old path and a new path.  Each path is a tuple of switch ids
starting at the ingress switch and ending at the *last-hop* switch (the switch
owning the port towards the destination host).  Paths are simple (no switch
repeats).  Hosts are not switches: ``HOST`` is a sentinel meaning "delivered".

Every switch holds at most one live rule per (flow, tag).  A packet enters at
the ingress and at each switch follows that switch's rule for the flow.  The
invariants audited at every instant are:

INFLIGHT   a packet in the network never loops and never reaches a switch
           with no rule for it (blackhole).
HOLDERS    **every** switch that still holds a live rule for the flow must be
           able to reach the destination along that rule.  A switch holding a
           rule that leads into a switch without one is a *latent* blackhole:
           it is harmless only if no packet ever arrives there, which an
           orchestrator cannot assume (buffer drains, retransmits, ack timing).
           Enforcing HOLDERS is exactly what makes "freeze a switch, update
           it, resume" sound, and it is what the simulator's packet replay
           confirms at packet granularity.

Bandwidth is audited on the *worst-case* one-tick load: every holder may be
holding a buffered packet and may send along its own path.  A flow contributes
its demand to a link at most once (its offered rate is split between old and
new paths, never doubled), so the load of a flow is the union of the links
reachable from its holders.

Two-stage (tagged) updates
--------------------------
A flow can be updated with version tags: every switch on the new path first
gets the new rule under tag 1 while the ingress still stamps packets with tag
0; the ingress then flips its stamp; after a drain window the old rules are
dropped.  Only *live* tags are audited — a tag no ingress emits cannot appear
in the network, so an incomplete set of tag-1 rules during installation is
invisible and therefore safe.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

HOST = -1
TAG_OLD = 0
TAG_NEW = 1


def nm(x: int) -> str:
    return "HOST" if x == HOST else f"s{x}"


def nexthop(path: Sequence[int], x: int) -> Optional[int]:
    """Next hop at switch ``x`` on ``path``; HOST at the last hop; None if absent."""
    if x not in path:
        return None
    i = path.index(x)
    return HOST if i == len(path) - 1 else path[i + 1]


def path_links(path: Sequence[int]) -> Set[Tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in zip(path, path[1:])}


def path_str(path: Sequence[int]) -> str:
    if not path:
        return "-"
    return "->".join(nm(x) for x in path) + "->H"


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------


class Topology:
    """Undirected switch graph; ``cap`` is packets per tick on a link."""

    def __init__(self, name: str = "topo") -> None:
        self.name = name
        self.switches: List[int] = []
        self.links: Dict[Tuple[int, int], int] = {}
        self.adj: Dict[int, List[int]] = {}
        self.host_at: Dict[int, List[int]] = {}
        self.table_cap: Dict[int, int] = {}

    # -- construction ---------------------------------------------------
    def add_switch(self, sw: int, table_cap: int = 1 << 20) -> None:
        if sw not in self.adj:
            self.adj[sw] = []
            self.switches.append(sw)
            self.host_at.setdefault(sw, [])
            self.table_cap[sw] = table_cap

    def add_link(self, a: int, b: int, cap: int) -> None:
        if a == b:
            raise ValueError("self loop")
        self.add_switch(a)
        self.add_switch(b)
        key = (min(a, b), max(a, b))
        if key in self.links:
            raise ValueError(f"duplicate link {key}")
        self.links[key] = cap
        self.adj[a].append(b)
        self.adj[b].append(a)

    def add_host(self, sw: int, host: int) -> None:
        self.add_switch(sw)
        self.host_at[sw].append(host)

    # -- queries --------------------------------------------------------
    def cap(self, a: int, b: int) -> int:
        return self.links[(min(a, b), max(a, b))]

    def has_link(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.links

    def neighbours(self, sw: int) -> List[int]:
        return sorted(self.adj.get(sw, []))

    def copy(self) -> "Topology":
        return copy.deepcopy(self)

    def link_list(self) -> List[Tuple[int, int]]:
        return sorted(self.links)

    def without(self, dead: Iterable[int]) -> "Topology":
        """Copy with the given switches (and their links) removed."""
        dead = set(dead)
        t = Topology(self.name + "-minus" + "-".join(str(d) for d in sorted(dead)))
        for sw in self.switches:
            if sw in dead:
                continue
            t.add_switch(sw, self.table_cap.get(sw, 1 << 20))
        for (a, b), cap in self.links.items():
            if a in dead or b in dead:
                continue
            t.links[(a, b)] = cap
            t.adj[a].append(b)
            t.adj[b].append(a)
        for sw, hosts in self.host_at.items():
            if sw in dead:
                continue
            t.host_at[sw] = list(hosts)
        return t

    def egress_switches(self, dst: int) -> Set[int]:
        """Switches that have a port to host ``dst``."""
        return {sw for sw, hs in self.host_at.items() if dst in hs}

    def __repr__(self) -> str:
        return (
            f"Topology({self.name}: {len(self.switches)} switches, "
            f"{len(self.links)} links)"
        )


# ---------------------------------------------------------------------------
# flows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Flow:
    fid: int
    src: int           # ingress switch
    dst: int           # destination host id
    old: Tuple[int, ...]
    new: Tuple[int, ...]
    demand: int = 1    # packets per tick

    def __post_init__(self) -> None:
        if not self.old or not self.new:
            raise ValueError(f"flow {self.fid}: empty path")
        for name, p in (("old", self.old), ("new", self.new)):
            if len(set(p)) != len(p):
                raise ValueError(f"flow {self.fid}: {name} path is not simple: {p}")
            if p[0] != self.src:
                raise ValueError(
                    f"flow {self.fid}: {name} path starts at {p[0]}, not src {self.src}"
                )
            if p[-1] == self.src:
                raise ValueError(f"flow {self.fid}: src is also the last hop")

    @property
    def switches(self) -> frozenset:
        return frozenset(self.old) | frozenset(self.new)

    @property
    def changed_switches(self) -> frozenset:
        out = set()
        for x in self.switches:
            if nexthop(self.old, x) != nexthop(self.new, x):
                out.add(x)
        return frozenset(out)

    def rule_old(self, x: int) -> Optional[int]:
        return nexthop(self.old, x)

    def rule_new(self, x: int) -> Optional[int]:
        return nexthop(self.new, x)

    def __repr__(self) -> str:
        return f"Flow({self.fid}: {path_str(self.old)} => {path_str(self.new)} d={self.demand})"


# ---------------------------------------------------------------------------
# per-flow execution state
# ---------------------------------------------------------------------------


@dataclass
class FState:
    """Rules currently installed for one flow.

    Plain (untagged) mode
        switches in ``upd`` hold the ``target`` rule, the others ``base``.

    Tagged (two-stage) mode
        ``t_from`` / ``t_to``  the two paths
        ``old_inst``           switches holding the tag-0 (from) rule
        ``new_inst``           switches holding the tag-1 (to) rule
        ``stamp``              tag the ingress currently emits
        ``draining``           both tags are live (old packets still in flight)
    """

    flow: Flow
    mode: str = "plain"
    # Plain mode: the exact rule held at every switch.  This is deliberately
    # explicit rather than derived from a "target path + flipped set", because a
    # retreat may have to leave switches on several different paths (some
    # already on the new path, some back on the old one).  A derived form would
    # silently snap those back to a single path.
    rules: Dict[int, Optional[int]] = field(default_factory=dict)
    t_from: Tuple[int, ...] = ()
    t_to: Tuple[int, ...] = ()
    old_inst: frozenset = frozenset()
    new_inst: frozenset = frozenset()
    stamp: int = TAG_OLD
    draining: bool = False
    note: str = ""

    @classmethod
    def fresh(cls, f: Flow) -> "FState":
        st = cls(flow=f, mode="plain")
        st._reset_plain(f.old)
        return st

    # -- plain helpers --------------------------------------------------
    def _reset_plain(self, path: Tuple[int, ...]) -> None:
        st = self.flow
        self.rules = {}
        for x in set(st.old) | set(st.new) | set(path):
            self.rules[x] = nexthop(path, x)

    def _path_rules(self, path: Tuple[int, ...]) -> Dict[int, Optional[int]]:
        return {x: nexthop(path, x) for x in set(path) | set(self.flow.old) | set(self.flow.new)}

    # -- live tags ------------------------------------------------------
    def live_tags(self) -> Tuple[Optional[int], ...]:
        if self.mode == "plain":
            return (None,)
        if self.stamp == TAG_OLD:
            return (TAG_OLD,)
        return (TAG_OLD, TAG_NEW) if self.draining else (TAG_NEW,)

    # -- rules ----------------------------------------------------------
    def rule(self, x: int, tag: Optional[int] = None) -> Optional[int]:
        if self.mode == "plain":
            return self.rules.get(x)
        if tag is None:
            tag = self.stamp
        if tag == TAG_NEW:
            return nexthop(self.t_to, x) if x in self.new_inst else None
        return nexthop(self.t_from, x) if x in self.old_inst else None

    def holder_switches(self) -> Set[int]:
        """Switches holding any live rule for this flow."""
        if self.mode == "plain":
            return {x for x, r in self.rules.items() if r is not None}
        return set(self.old_inst) | set(self.new_inst)

    def holders_for(self, tag: Optional[int]) -> Set[int]:
        """Switches holding a rule a packet *carrying this tag* could meet.

        This is the set the HOLDERS invariant must be checked over: a switch
        that has only the tag-1 rule installed is invisible to a tag-0 packet,
        and vice versa.  Checking all holders against every tag would wrongly
        demand that a new-path-only switch also forward the old tag.
        """
        if self.mode == "plain":
            return self.holder_switches()
        if tag == TAG_NEW:
            return set(self.new_inst)
        if tag == TAG_OLD:
            return set(self.old_inst)
        return self.holder_switches()

    def pending(self) -> Set[int]:
        """In plain mode nothing is intrinsically pending: every rule is explicit.

        A "move" is just a plan over ``rules``; the planner decides which
        switches to change next.  ``pending`` remains for the tagged path only.
        """
        return set()

    def settled(self) -> bool:
        return self.mode == "plain"

    def current_path(self) -> Tuple[int, ...]:
        if self.mode == "plain":
            return self._frames_committed()
        return self.t_from if self.stamp == TAG_OLD else self.t_to

    def committed_path(self) -> Tuple[int, ...]:
        """A representative path for this flow's live rules.

        If the ingress rule matches the new path, report the new path; else the
        old path.  Used by the retreat's classification, which only needs to
        know whether the flow is "on old" or "on new".
        """
        if self.mode == "plain":
            return self._frames_committed()
        return self.t_to if self.stamp == TAG_NEW else self.t_from

    def _frames_committed(self) -> Tuple[int, ...]:
        """Which path the live rules currently describe.

        The ingress next hop alone is not enough: old and new paths often share
        their first hop while diverging later.  A flow counts as "on the new
        path" only when every switch it touches holds the new path's rule.
        """
        f = self.flow
        if self._matches(f.new):
            return f.new
        return f.old

    def _matches(self, path: Tuple[int, ...]) -> bool:
        return all(
            self.rules.get(x) == nexthop(path, x)
            for x in set(path) | set(self.flow.old) | set(self.flow.new)
        )

    def fully_on(self, path: Tuple[int, ...]) -> bool:
        return all(
            self.rules.get(x) == nexthop(path, x)
            for x in set(path) | set(self.flow.old) | set(self.flow.new)
        )

    # -- mutation -------------------------------------------------------
    def begin_plain(self, path: Tuple[int, ...]) -> None:
        """Reset every switch to ``path`` (only legal when no switch is mid-set)."""
        if self.mode != "plain":
            raise RuntimeError("begin_plain on a tagged flow")
        self._reset_plain(path)

    def flip(self, x: int) -> None:
        raise RuntimeError("flip() needs an explicit target rule; use set_rule()")

    def set_rule(self, x: int, rule: Optional[int]) -> None:
        """Install ``rule`` (a next hop or HOST, or None to remove) at switch x."""
        if self.mode != "plain":
            raise RuntimeError("set_rule on a tagged flow")
        self.rules[x] = rule

    def settle(self) -> None:
        return

    def begin_tagged(self, to_path: Tuple[int, ...]) -> None:
        if self.mode == "tagged":
            if self.pending():
                raise RuntimeError("begin_tagged while tagged move in flight")
            if self.t_to == to_path:
                return
        if self.mode == "plain":
            frm = self._frames_committed()
        else:
            frm = self.t_to
        self.mode = "tagged"
        self.t_from = frm
        self.t_to = to_path
        self.old_inst = frozenset(frm)
        self.new_inst = frozenset()
        self.stamp = TAG_OLD
        self.draining = False

    def install_tagged(self, x: int) -> None:
        if self.mode != "tagged":
            raise RuntimeError("install_tagged on a plain flow")
        self.new_inst = self.new_inst | {x}

    def flip_stamp(self) -> None:
        if self.mode != "tagged":
            raise RuntimeError("flip_stamp on a plain flow")
        if not self.new_inst >= set(self.t_to):
            raise RuntimeError(
                f"flow {self.flow.fid}: stamp flipped before new path installed"
            )
        self.stamp = TAG_NEW
        self.draining = True

    def maybe_finish(self) -> None:
        """If a tagged flow has finished draining, collapse to the plain new path.

        ``drop_old`` may only be invoked when the old rules are removed; this
        helper also finalises the state so the flow reports a plain new path
        once no tag-0 packets can still be in flight.
        """
        if self.mode == "tagged" and self.draining and not self.old_inst:
            self.drop_old()

    def drop_old(self) -> None:
        if self.mode != "tagged":
            raise RuntimeError("drop_old on a plain flow")
        self.old_inst = frozenset()
        self.draining = False
        # the flow is now simply on t_to
        newpath = self.t_to
        self.mode = "plain"
        self._reset_plain(newpath)
        self.t_from = ()
        self.t_to = ()
        self.stamp = TAG_OLD

    def clone(self) -> "FState":
        return FState(
            flow=self.flow,
            mode=self.mode,
            rules=dict(self.rules),
            t_from=self.t_from,
            t_to=self.t_to,
            old_inst=self.old_inst,
            new_inst=self.new_inst,
            stamp=self.stamp,
            draining=self.draining,
            note=self.note,
        )

    def snapshot(self) -> dict:
        return {
            "fid": self.flow.fid,
            "src": self.flow.src,
            "dst": self.flow.dst,
            "old": list(self.flow.old),
            "new": list(self.flow.new),
            "demand": self.flow.demand,
            "mode": self.mode,
            "rules": {x: r for x, r in sorted(self.rules.items())},
            "committed": list(self._frames_committed()),
            "pending": sorted(self.pending()),
            "t_from": list(self.t_from),
            "t_to": list(self.t_to),
            "old_inst": sorted(self.old_inst),
            "new_inst": sorted(self.new_inst),
            "stamp": self.stamp,
            "draining": self.draining,
            "live_tags": [t for t in self.live_tags()],
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    kind: str  # loop | blackhole | overload | table
    tick: float
    detail: str
    fid: Optional[int] = None

    def key(self) -> Tuple[str, int]:
        return (self.kind, -1 if self.fid is None else self.fid)

    def __str__(self) -> str:
        who = "" if self.fid is None else f" flow={self.fid}"
        return f"[t={self.tick:9.3f}] {self.kind.upper():9s}{who}: {self.detail}"


def walk(
    states: Mapping[int, FState], fid: int, tag: Optional[int] = None
) -> Tuple[bool, List[int], Optional[int]]:
    """Follow one packet from the ingress.  Returns ``(ok, trail, loop_at)``."""
    s = states[fid]
    x = s.flow.src
    seen: Set[int] = set()
    trail = [x]
    while True:
        if x == HOST:
            return True, trail, None
        if x in seen:
            return False, trail, x
        seen.add(x)
        nx = s.rule(x, tag)
        if nx is None:
            return False, trail, None
        x = nx
        trail.append(x)


def walk_from(
    states: Mapping[int, FState], fid: int, start: int, tag: Optional[int] = None
) -> Tuple[bool, List[int], Optional[int]]:
    s = states[fid]
    x = start
    seen: Set[int] = set()
    trail = [x]
    while True:
        if x == HOST:
            return True, trail, None
        if x in seen:
            return False, trail, x
        seen.add(x)
        nx = s.rule(x, tag)
        if nx is None:
            return False, trail, None
        x = nx
        trail.append(x)


def audit(
    states: Mapping[int, FState],
    topo: Optional[Topology] = None,
    strict_hold: bool = True,
    check_capacity: bool = True,
    tick: float = 0.0,
    exempt: Iterable[int] = (),
    failed_switches: Iterable[int] = (),
) -> List[Violation]:
    """Audit all invariants at one instant.

    ``exempt`` flows cannot possibly be safe (their path crosses a failed
    switch); the caller reports those as failure fallout, not as violations.
    """
    out: List[Violation] = []
    exempt = set(exempt)
    failed = set(failed_switches)
    for fid in sorted(states):
        if fid in exempt:
            continue
        s = states[fid]
        for tag in s.live_tags():
            ok, trail, loop_at = walk(states, fid, tag)
            if not ok:
                out.append(
                    Violation(
                        "loop" if loop_at is not None else "blackhole",
                        tick,
                        f"ingress(tag={tag}) " + "->".join(nm(x) for x in trail),
                        fid,
                    )
                )
                break
            if not strict_hold:
                continue
            bad = set()
            for h in sorted(s.holders_for(tag)):
                if h in bad or h in failed:
                    continue
                hok, htrail, hloop = walk_from(states, fid, h, tag)
                if not hok:
                    bad.add(h)
                    out.append(
                        Violation(
                            "loop" if hloop is not None else "blackhole",
                            tick,
                            f"holder {nm(h)} cannot reach dst (tag={tag}): "
                            + "->".join(nm(x) for x in htrail),
                            fid,
                        )
                    )
    if topo is not None and check_capacity:
        for (a, b), load in link_load(states, topo).items():
            if load > topo.cap(a, b):
                out.append(
                    Violation(
                        "overload",
                        tick,
                        f"link s{a}-s{b}: {load} > cap {topo.cap(a, b)}",
                    )
                )
    if topo is not None:
        for sw, used in sorted(table_usage(states).items()):
            cap = topo.table_cap.get(sw, 1 << 20)
            if used > cap:
                out.append(
                    Violation("table", tick, f"switch {nm(sw)}: {used} rules > cap {cap}")
                )
    return out


def reachable_links(
    states: Mapping[int, FState], fid: int, tag: Optional[int]
) -> Set[Tuple[int, int]]:
    """All links flow ``fid`` can possibly put a packet on, given its holders."""
    s = states[fid]
    links: Set[Tuple[int, int]] = set()
    for h in s.holders_for(tag):
        ok, trail, _ = walk_from(states, fid, h, tag)
        if not ok:
            continue
        for a, b in zip(trail, trail[1:]):
            if a == HOST or b == HOST:
                continue
            links.add((min(a, b), max(a, b)))
    return links


def link_load(
    states: Mapping[int, FState], topo: Topology
) -> Dict[Tuple[int, int], int]:
    """Worst-case one-tick load per link (see module docstring)."""
    load: Dict[Tuple[int, int], int] = {}
    for fid, s in states.items():
        links: Set[Tuple[int, int]] = set()
        for tag in s.live_tags():
            links |= reachable_links(states, fid, tag)
        for key in links:
            load[key] = load.get(key, 0) + s.flow.demand
    return load


def committed_load(
    states: Mapping[int, FState], topo: Topology
) -> Dict[Tuple[int, int], int]:
    """Steady-state load if every flow follows its committed path."""
    load: Dict[Tuple[int, int], int] = {}
    for s in states.values():
        for key in path_links(s.committed_path()):
            load[key] = load.get(key, 0) + s.flow.demand
    return load


def table_entries(s: FState) -> Dict[int, int]:
    """Rules (entries, not switches) each switch holds for one flow.

    In tagged mode a switch on both paths holds the old *and* the new rule, so
    it occupies two entries.  This is what two-phase updates must budget for.
    """
    out: Dict[int, int] = {}
    if s.mode == "plain":
        for x in s.holder_switches():
            out[x] = 1
        return out
    for x in s.old_inst:
        out[x] = out.get(x, 0) + 1
    for x in s.new_inst:
        out[x] = out.get(x, 0) + 1
    return out


def table_usage(states: Mapping[int, FState]) -> Dict[int, int]:
    """Number of table entries each switch holds, across all flows."""
    used: Dict[int, int] = {}
    for s in states.values():
        for x, n in table_entries(s).items():
            used[x] = used.get(x, 0) + n
    return used


def make_states(flows: Iterable[Flow]) -> Dict[int, FState]:
    return {f.fid: FState.fresh(f) for f in flows}


def clone_states(states: Mapping[int, FState]) -> Dict[int, FState]:
    return {fid: s.clone() for fid, s in states.items()}


def forwarding_view(states: Mapping[int, FState]) -> List[dict]:
    """Serialisable description of every flow's current end-to-end behaviour."""
    out = []
    for fid in sorted(states):
        s = states[fid]
        tags = []
        for tag in s.live_tags():
            ok, trail, loop_at = walk(states, fid, tag)
            tags.append({"tag": tag, "ok": ok, "trail": trail, "loop_at": loop_at})
        out.append(
            {
                "fid": fid,
                "src": s.flow.src,
                "dst": s.flow.dst,
                "mode": s.mode,
                "rules": {x: r for x, r in sorted(s.rules.items())},
                "committed": list(s.committed_path()),
                "pending": sorted(s.pending()),
                "stamp": s.stamp,
                "tags": tags,
                "holders": sorted(s.holder_switches()),
            }
        )
    return out
