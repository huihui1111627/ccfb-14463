"""Randomised scenario generation.

A scenario is a topology, a set of flows each with an old and a new path, and
the task of moving every flow onto its new path.  Generation is parameterised so
the acceptance harness can sweep:

* ``switches`` / ``degree``  fabric size and connectivity
* ``flows``                  how many flows change
* ``tight``                  choose new paths under load, so some link ends up
                             near capacity and capacity staging or a detour is
                             needed
* ``swap_bias``              how often the new path is a small perturbation of
                             the old one — exactly what produces dependency
                             cycles and therefore version-tagged updates
* ``table_cap``              per-switch rule budget, so two-phase updates can be
                             made not to fit

Everything is driven by an explicit ``random.Random``, so a scenario is fully
reproducible from its seed.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .model import Flow, Topology, path_links


@dataclass
class Scenario:
    name: str
    topo: Topology
    flows: List[Flow]
    seed: int = 0
    tight: bool = False
    table_cap: int = 1 << 20

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "switches": len(self.topo.switches),
            "links": len(self.topo.links),
            "hosts": sum(len(h) for h in self.topo.host_at.values()),
            "flows": len(self.flows),
            "changing": sum(1 for f in self.flows if f.old != f.new),
            "tight": self.tight,
            "table_cap": self.table_cap,
        }


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------


def random_fabric(
    rng: random.Random,
    switches: int,
    degree: int = 3,
    cap: int = 10,
    table_cap: int = 256,
    hosts: int = 4,
    name: str = "fabric",
) -> Topology:
    """Connected random graph plus ``hosts`` hosts spread over the switches."""
    t = Topology(name)
    for s in range(switches):
        t.add_switch(s, table_cap)
    order = list(range(switches))
    rng.shuffle(order)
    for i in range(1, switches):
        a, b = order[i], order[rng.randrange(i)]
        if not t.has_link(a, b):
            t.add_link(a, b, cap)
    target = min(switches * (switches - 1) // 2, max(switches - 1, switches * degree // 2))
    guard = 0
    while len(t.links) < target and guard < switches * switches * 6:
        guard += 1
        a, b = rng.randrange(switches), rng.randrange(switches)
        if a != b and not t.has_link(a, b):
            t.add_link(a, b, cap)
    # hosts: prefer leaves for shorter paths, but keep a few on the core
    deg = {s: len(t.adj[s]) for s in t.switches}
    leaves = [s for s in t.switches if deg[s] <= 1]
    pool = leaves + [s for s in t.switches if s not in leaves]
    rng.shuffle(pool)
    for k in range(hosts):
        t.add_host(pool[k % len(pool)], 10_000 + k)
    return t


def _rand_path(
    topo: Topology,
    src: int,
    last: int,
    rng: random.Random,
    banned: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[Tuple[int, ...]]:
    """A randomised simple path src..last (BFS over a shuffled frontier)."""
    banned = banned or set()
    prev: Dict[int, int] = {src: -1}
    frontier = [src]
    while frontier:
        rng.shuffle(frontier)
        nxt: List[int] = []
        for u in frontier:
            for w in topo.neighbours(u):
                if w in prev or w in prev:
                    continue
                if (min(u, w), max(u, w)) in banned:
                    continue
                prev[w] = u
                if w == last:
                    path = [w]
                    while prev[path[-1]] != -1:
                        path.append(prev[path[-1]])
                    return tuple(reversed(path))
                nxt.append(w)
        frontier = nxt
    return None


def _spare_path(
    topo: Topology, src: int, last: int, load: Dict[Tuple[int, int], int], demand: int
) -> Optional[Tuple[int, ...]]:
    """Shortest path whose every link has ``demand`` spare after ``load``."""
    pq = [(0, src, (src,))]
    seen: Set[int] = set()
    while pq:
        hops, node, path = heapq.heappop(pq)
        if node == last:
            return path
        if node in seen:
            continue
        seen.add(node)
        for w in topo.neighbours(node):
            if w in seen or w in path:
                continue
            l = (min(node, w), max(node, w))
            if topo.cap(*l) - load.get(l, 0) < demand:
                continue
            heapq.heappush(pq, (hops + 1, w, path + (w,)))
    return None


def _perturb(topo: Topology, path: Tuple[int, ...], rng: random.Random,
             last: int) -> Optional[Tuple[int, ...]]:
    """Small perturbation of a simple path: swap hops or hop sideways.

    These are the perturbations that create dependency cycles (a swap) or long
    chains (a shift by one hop), which is what we want to stress.
    """
    if len(path) < 3:
        return None
    mode = rng.random()
    if mode < 0.55:
        # swap two adjacent internal hops
        i = rng.randrange(1, len(path) - 1)
        cand = list(path)
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        t = tuple(cand)
        if len(set(t)) == len(t) and topo.has_link(t[i], t[i + 1]) and topo.has_link(t[i - 1], t[i]):
            return t
        return None
    # replace one internal hop with a neighbour not already on the path
    i = rng.randrange(1, len(path) - 1)
    a = path[i - 1]
    b = path[i + 1] if i + 1 < len(path) else last
    options = [
        w for w in topo.neighbours(a)
        if w not in path and w != b and topo.has_link(w, b)
    ]
    if not options:
        return None
    w = rng.choice(options)
    cand = path[:i] + (w,) + path[i + 1:]
    if len(set(cand)) != len(cand):
        return None
    return cand


# ---------------------------------------------------------------------------
# scenario
# ---------------------------------------------------------------------------


def random_scenario(
    seed: int,
    switches: int = 10,
    degree: int = 3,
    flows: int = 6,
    cap: int = 10,
    demand: int = 1,
    tight: bool = False,
    swap_bias: float = 0.5,
    table_cap: int = 256,
    hosts: int = 4,
    name: str = "scenario",
) -> Scenario:
    rng = random.Random(seed)
    topo = random_fabric(rng, switches, degree, cap, table_cap, hosts, name)
    dst_hosts = [h for hs in topo.host_at.values() for h in hs]
    host_sw = {h: s for s, hs in topo.host_at.items() for h in hs}
    out: List[Flow] = []
    # Two separate load maps: the *current* (old) configuration must be feasible
    # — it is the live network — and the *target* (new) configuration must be
    # feasible.  The per-flow old∪new footprint may be tight; that is what
    # forces capacity staging and, in the worst case, a transit detour.
    old_load: Dict[Tuple[int, int], int] = {}
    new_load: Dict[Tuple[int, int], int] = {}
    for fid in range(flows):
        for _attempt in range(200):
            src = rng.choice(topo.switches)
            dst = rng.choice(dst_hosts)
            last = host_sw[dst]
            if src == last:
                continue
            # the LIVE configuration must always be feasible: choose an old
            # path with spare capacity whatever the tightness mode
            old = _spare_path(topo, src, last, old_load, demand)
            if old is None and not tight:
                old = _rand_path(topo, src, last, rng)
            if not old or len(old) < 2:
                continue
            new: Optional[Tuple[int, ...]] = None
            if rng.random() < swap_bias:
                new = _perturb(topo, old, rng, last)
            if new is None:
                if tight:
                    new = _spare_path(topo, src, last, new_load, demand)
                else:
                    new = _rand_path(topo, src, last, rng)
                for _ in range(8):
                    if new is not None and new != old:
                        break
                    new = _rand_path(topo, src, last, rng)
            if new is None or new == old or len(new) < 2:
                continue
            if len(set(new)) != len(new) or new[0] != src:
                continue
            if any(f.old == old and f.new == new for f in out):
                continue
            # enforce target feasibility; in tight mode also keep the footprint
            # within reach of capacity (so staging, not infeasibility, is needed)
            # the target configuration must always be feasible too
            if not all(
                new_load.get(l, 0) + demand <= topo.cap(*l)
                for l in path_links(new)
            ):
                continue
            for l in path_links(old):
                old_load[l] = old_load.get(l, 0) + demand
            for l in path_links(new):
                new_load[l] = new_load.get(l, 0) + demand
            out.append(Flow(fid, src, dst, old, new, demand))
            break
    return Scenario(name, topo, out, seed, tight, table_cap)


# ---------------------------------------------------------------------------
# hand-built cases for the tests
# ---------------------------------------------------------------------------


def merge_scenario(name: str = "merge", cap: int = 10) -> Tuple[Topology, List[Flow]]:
    """Two flows whose new paths collide on one link (capacity staging)."""
    t = Topology(name)
    for s in range(6):
        t.add_switch(s, 256)
    edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (0, 4), (1, 5)]
    for a, b in edges:
        t.add_link(a, b, cap)
    t.add_host(5, 500)
    t.add_host(4, 400)
    flows = [
        Flow(0, 0, 500, (0, 2, 3, 4, 5), (0, 4, 5), 4),
        Flow(1, 1, 500, (1, 2, 3, 4, 5), (1, 5), 4),
    ]
    return t, flows


def swap_scenario(name: str = "swap") -> Tuple[Topology, List[Flow]]:
    """A flow that swaps two adjacent hops: needs a version-tagged update."""
    t = Topology(name)
    for s in range(5):
        t.add_switch(s, 256)
    for a, b in [(0, 1), (1, 3), (0, 2), (2, 3), (1, 2)]:
        t.add_link(a, b, 10)
    t.add_host(3, 300)
    return t, [Flow(0, 0, 300, (0, 1, 2, 3), (0, 2, 1, 3), 1)]


def detour_scenario(name: str = "detour", cap: int = 10) -> Tuple[Topology, List[Flow]]:
    r"""A capacity deadlock that can only be broken by a temporary transit path.

    Layout::

        s0 -- s1 -- s2 -- s3 -- s4 -- s5
         \          |          |         /
          +--- s6 --+----------+--------+

    Flow A: s0 -> s1 -> s2 -> s3 -> s4 -> s5  (big flow, 6 pkts/tick)
            new path s0 -> s6 -> s4 -> s5
    Flow B: s1 -> s2 -> s3 -> s4 -> s5        (4 pkts/tick)
            new path s1 -> s2 -> s5  (via the s2--s5 shortcut)

    Flow A's worst-case footprint is its whole old path (its old rule is held
    until it flips), and flow A cannot start while flow B is on the shared
    segment.  Flow B cannot start while flow A holds the segment.  Neither can
    free the segment first without the other moving, hence a deadlock.  The
    spare path s0-s6-s4-s5 lets A vacate the segment, after which B can move,
    and finally A can take its new path.
    """
    t = Topology(name)
    for s in range(7):
        t.add_switch(s, 256)
    for a, b in [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 6), (6, 4), (1, 6), (2, 5),
    ]:
        t.add_link(a, b, cap)
    t.add_host(5, 500)
    flows = [
        Flow(0, 0, 500, (0, 1, 2, 3, 4, 5), (0, 6, 4, 5), 6),
        Flow(1, 1, 500, (1, 2, 3, 4, 5), (1, 2, 5), 4),
    ]
    return t, flows


def chain_scenario(name: str = "chain") -> Tuple[Topology, List[Flow]]:
    """A ladder where every flow shifts one hop: a long dependency chain."""
    t = Topology(name)
    n = 8
    for s in range(n):
        t.add_switch(s, 256)
    for i in range(n - 1):
        t.add_link(i, i + 1, 20)
    for i in range(n - 2):
        t.add_link(i, i + 2, 20)
    t.add_host(n - 1, 900)
    host_last = n - 1
    flows = []
    for k in range(5):
        a = k
        b = host_last
        if a >= b - 1:
            continue
        old = tuple(range(a, b + 1))
        # take the same route but replace one hop with a ladder shortcut;
        # the shortcut exists by construction (i -- i+2 links)
        i = a + 1
        cand = old[:i] + (i + 1,) + old[i + 2:] if i + 2 <= b else old
        if len(set(cand)) != len(cand) or cand[-1] != b or cand[0] != a:
            cand = old
        if cand == old:
            continue
        flows.append(Flow(k, a, b, old, cand, 1))
    return t, flows


def detour_family(seed: int, name: str = "detourfam") -> Tuple[Topology, List[Flow]]:
    """Generate a capacity-deadlock family by construction.

    Two flows each need a link that the other currently occupies, both at a
    load that cannot coexist, and both have a spare transit route.  The
    topology is a two-lane ring with shortcuts, so a spare path exists.
    """
    rng = random.Random(seed)
    n = 7
    t = Topology(f"{name}{seed}")
    for s in range(n):
        t.add_switch(s, 256)
    # lanes: 0-1-4 and 0-2-4 are the two "contended" paths; 0-5-6-4 is the
    # spare transit lane used to break the deadlock.
    base = [(0, 1), (1, 4), (0, 2), (2, 4)]
    upper = [(0, 5), (5, 6), (6, 4)]
    for a, b in base + upper:
        t.add_link(a, b, 10)
    t.add_host(4, 400)
    # two flows with identical endpoints that swap lanes; demand chosen so
    # their worst-case footprints cannot coexist on the shared lane
    # each flow alone fits on either lane (<= 10) but together their
    # worst-case footprints (old ∪ new) exceed a lane, so one must vacate
    # that lane first; since both need the other's lane, this is a deadlock
    # that only a spare lane (0-5-6-4) can break.
    d0 = rng.choice([6, 7])
    d1 = rng.choice([6, 7])
    flows = [
        Flow(0, 0, 400, (0, 1, 4), (0, 2, 4), d0),
        Flow(1, 0, 400, (0, 2, 4), (0, 1, 4), d1),
    ]
    return t, flows
