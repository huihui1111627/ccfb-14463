"""Dependency extraction between switch table updates.

Definitions
-----------
For a flow with old path ``A`` and new path ``B`` and a switch ``x``::

    old(x) = next hop on A  (HOST at the last hop, None if x not on A)
    new(x) = next hop on B
    changed = { x : old(x) != new(x) }

A switch can only flip its whole table at once, so "flip x" is an atomic action.
A partly-updated network is described by the set ``C`` of switches that have
already flipped.  Two rules force an order:

1. **New path, downstream first.**
   If ``x`` flips and its new successor ``y`` (the next *changed* switch further
   down ``B``; ``y`` itself when ``x`` is changed) has not flipped yet, ``y``
   is still forwarding along its old rule, and a packet following the new edge
   ``x -> y`` enters a stale path and is likely to loop or drop.  Therefore
   ``y`` must flip **before** ``x``.  This applies to every pair of changed
   switches on ``B``: the more downstream one goes first.

2. **Old path, upstream first.**
   If ``x`` is changed and its old successor ``y`` is also changed, but ``x``'s
   new rule no longer points at ``y``, then a packet can still arrive at ``y``
   through the old edge ``x -> y`` until ``x`` flips.  If ``y`` flips first it
   removes the entry the incoming packet needs.  Therefore ``x`` must flip
   **before** ``y``: old-path changed switches go upstream first.

The resulting graph can contain cycles even for a single flow (e.g. swapping
two adjacent hops), in which case no one-phase order exists and a two-stage
tagged update must be used.

Why down-sets are safe
----------------------
Write ``prereq[x]`` for the set of switches that must have flipped before
``x``.  A state ``C`` is a *down-set* if ``prereq[x] <= C`` for every ``x in C``.
Two facts, both checked exhaustively by brute force over ~130k random
single- and multi-flow instances (see ``tests/test_deps.py``):

* every down-set of the graph above satisfies the INFLIGHT and HOLDERS
  invariants (see ``model``), and
* every safe one-phase state is *some* down-set of a graph in the transitive
  closure of the rules above, so the rules are not merely sufficient but
  capture the whole safe region's shape.

Consequently: a set of switches may be flipped **in the same batch** (in any
order, because a batch's switches may apply in arbitrary order) iff every one
of them has all its prerequisites **already committed**, i.e. iff the batch is
an antichain of the prerequisite relation sitting on top of a down-set.  This
is exactly the condition the planner uses, and it is why "wait for acks between
batches" is the right execution model.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .model import Flow, nexthop

Node = int


# ---------------------------------------------------------------------------
# per-flow prerequisite extraction
# ---------------------------------------------------------------------------


def flow_prereqs(f: Flow) -> Dict[int, Set[int]]:
    """``x -> switches that must flip before x``, for one flow."""
    ch = f.changed_switches
    pre: Dict[int, Set[int]] = {x: set() for x in ch}

    # (1) new path: changed downstream switches first
    newseq = [x for x in f.new if x in ch]
    for i, upstream in enumerate(newseq):
        for downstream in newseq[i + 1 :]:
            pre[upstream].add(downstream)

    # (2) old path: changed upstream switches first, when x stops pointing at y
    for x, y in zip(f.old, f.old[1:]):
        if x in ch and y in ch and nexthop(f.new, x) != y:
            pre[y].add(x)

    return pre


def flow_edges(f: Flow) -> List[Tuple[int, int]]:
    """Dependency edges ``(blocker, target)``: blocker flips before target."""
    return sorted((b, x) for x, bs in flow_prereqs(f).items() for b in bs)


def edges_to_prereq(edges: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
    pre: Dict[int, Set[int]] = {}
    for b, t in edges:
        pre.setdefault(t, set()).add(b)
        pre.setdefault(b, set())
    return pre


def merge_prereqs(maps: Iterable[Mapping[int, Iterable[int]]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for m in maps:
        for t, bs in m.items():
            out.setdefault(t, set()).update(bs)
            out.setdefault(t, set())
    return out


# ---------------------------------------------------------------------------
# graph algorithms
# ---------------------------------------------------------------------------


def has_cycle(nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]) -> bool:
    adj: Dict[Node, List[Node]] = {n: [] for n in nodes}
    for b, t in edges:
        adj.setdefault(b, []).append(t)
        adj.setdefault(t, [])
    color: Dict[Node, int] = {}
    for start in adj:
        if color.get(start, 0):
            continue
        stack = [(start, 0)]
        color[start] = 1
        while stack:
            node, i = stack[-1]
            neigh = adj[node]
            if i < len(neigh):
                stack[-1] = (node, i + 1)
                w = neigh[i]
                c = color.get(w, 0)
                if c == 1:
                    return True
                if c == 0:
                    color[w] = 1
                    stack.append((w, 0))
            else:
                color[node] = 2
                stack.pop()
    return False


def sccs(nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]) -> List[List[Node]]:
    """Tarjan's algorithm (iterative), deterministic ordering."""
    node_list = list(dict.fromkeys(nodes))
    adj: Dict[Node, List[Node]] = {n: [] for n in node_list}
    for b, t in edges:
        adj.setdefault(b, []).append(t)
        adj.setdefault(t, [])
    for n in adj:
        adj[n] = sorted(set(adj[n]))
    index: Dict[Node, int] = {}
    low: Dict[Node, int] = {}
    on_stack: Dict[Node, bool] = {}
    stack: List[Node] = []
    result: List[List[Node]] = []
    counter = 0

    for root in sorted(adj):
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            recurred = False
            neighbours = adj[node]
            while pi < len(neighbours):
                w = neighbours[pi]
                if w not in index:
                    work[-1] = (node, pi + 1)
                    work.append((w, 0))
                    recurred = True
                    break
                if on_stack.get(w):
                    low[node] = min(low[node], index[w])
                pi += 1
            if recurred:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == node:
                        break
                result.append(sorted(comp))
    return result


def cyclic_switches(
    nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]
) -> Set[Node]:
    node_list = list(nodes)
    edge_set = list(edges)
    out: Set[Node] = set()
    for comp in sccs(node_list, edge_set):
        if len(comp) > 1:
            out.update(comp)
        else:
            x = comp[0]
            if any(b == x and t == x for b, t in edge_set):
                out.add(x)
    return out


def ranks(
    nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]
) -> Dict[Node, int]:
    """Longest-path rank (0 = no prerequisites).  Requires a DAG."""
    node_list = list(dict.fromkeys(nodes))
    pre: Dict[Node, Set[Node]] = {n: set() for n in node_list}
    succ: Dict[Node, Set[Node]] = {n: set() for n in node_list}
    for b, t in edges:
        pre.setdefault(t, set()).add(b)
        succ.setdefault(b, set()).add(t)
        pre.setdefault(b, set())
        succ.setdefault(t, set())
    rank: Dict[Node, int] = {}
    # Kahn
    indeg = {n: len(pre[n]) for n in pre}
    ready = sorted(n for n in pre if indeg[n] == 0)
    order: List[Node] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for w in sorted(succ[n]):
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
                ready.sort()
    if len(order) != len(pre):
        raise ValueError("ranks() needs an acyclic graph")
    for n in order:
        rank[n] = max((rank[b] + 1 for b in pre[n]), default=0)
    return rank


def layers(nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]) -> List[List[Node]]:
    """Group switches into rounds: layer i = rank i.  Requires a DAG."""
    r = ranks(nodes, edges)
    if not r:
        return []
    out: List[List[Node]] = [[] for _ in range(max(r.values()) + 1)]
    for n, k in sorted(r.items()):
        out[k].append(n)
    return out


def longest_chain(nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]) -> int:
    """Height of the prerequisite DAG (0 for no nodes)."""
    r = ranks(nodes, edges)
    return (max(r.values()) + 1) if r else 0


def downsets(
    nodes: Sequence[Node], prereq: Mapping[Node, Iterable[Node]]
) -> List[Set[Node]]:
    """All down-sets, by brute force.  Test helper only (exponential)."""
    ns = list(nodes)
    out = []
    for mask in range(1 << len(ns)):
        chosen = {ns[i] for i in range(len(ns)) if mask >> i & 1}
        if all(set(prereq.get(x, ())) <= chosen for x in chosen):
            out.append(chosen)
    return out


def topological_order(
    nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]
) -> List[Node]:
    return [n for layer in layers(nodes, edges) for n in layer]
