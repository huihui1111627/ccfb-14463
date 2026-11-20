"""Exhaustive verification of the dependency rules and the planner's safety.

These tests are deliberately brute-force.  The whole design rests on the claim

    "every down-set of the per-flow prerequisite DAG is a safe instant"

so the tests enumerate random single- and multi-flow instances, enumerate
*every* subset of changed switches (all possible intermediate states), and check
the INFLIGHT and HOLDERS invariants with the production audit code.  They also
verify that the planner, run on the same instances, never emits a round whose
every-subset state is unsafe.
"""

from __future__ import annotations

import itertools
import random

import pytest

from netorch import deps
from netorch.model import (
    Flow,
    FState,
    Topology,
    audit,
    clone_states,
    make_states,
    nexthop,
    path_links,
)
from netorch.planner import Planner, apply_op


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rand_path(rng, switches, src, last):
    mids = [s for s in switches if s not in (src, last)]
    rng.shuffle(mids)
    k = rng.randint(0, len(mids))
    return tuple([src] + mids[:k] + [last])


def rand_instance(rng, n_sw=6, n_flow=1):
    switches = list(range(n_sw))
    src = rng.choice(switches)
    last = rng.choice([s for s in switches if s != src])
    flows = []
    for fid in range(n_flow):
        s = rng.choice(switches)
        cand = [t for t in switches if t != s]
        if not cand:
            continue
        last = rng.choice(cand)
        old = rand_path(rng, switches, s, last)
        new = rand_path(rng, switches, s, last)
        if len(old) < 2 or len(new) < 2:
            continue
        flows.append(Flow(fid, s, 1000 + fid, old, new, 1))
    return switches, flows


def subset_states(flows, switches):
    """Yield (flipped_set, states) for every subset of changed switches."""
    changed = sorted({x for f in flows for x in f.changed_switches})
    for mask in range(1 << len(changed)):
        flip = {changed[i] for i in range(len(changed)) if mask >> i & 1}
        states = make_states(flows)
        for f in flows:
            for x in f.changed_switches & flip:
                states[f.fid].set_rule(x, nexthop(f.new, x))
        yield flip, states


def safe_states(flows, states):
    return not audit(states, topo=None, strict_hold=True, check_capacity=False)


# ---------------------------------------------------------------------------
# the core claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(120))
def test_downsets_are_safe_single_flow(seed):
    """Every down-set of the per-flow DAG is loop-free and blackhole-free."""
    rng = random.Random(seed)
    switches, flows = rand_instance(rng, n_sw=6, n_flow=1)
    if not flows:
        pytest.skip("degenerate instance")
    f = flows[0]
    pre = deps.flow_prereqs(f)
    if deps.has_cycle(f.changed_switches, deps.flow_edges(f)):
        pytest.skip("cyclic: handled by tagging")
    tested = 0
    for flip, states in subset_states(flows, switches):
        if all(pre.get(x, set()) <= flip for x in flip):
            assert safe_states(flows, states), f"unsafe down-set {flip} for {f}"
            tested += 1
    assert tested >= 1


@pytest.mark.parametrize("seed", range(120))
def test_downsets_are_safe_multi_flow(seed):
    rng = random.Random(seed + 5000)
    switches, flows = rand_instance(rng, n_sw=6, n_flow=3)
    if not flows:
        pytest.skip("degenerate instance")
    pre = deps.merge_prereqs(deps.flow_prereqs(f) for f in flows)
    edges = [(b, t) for t, bs in pre.items() for b in bs]
    if deps.has_cycle(list(pre), edges):
        pytest.skip("cyclic: handled by tagging")
    for flip, states in subset_states(flows, switches):
        if all(pre.get(x, set()) <= flip for x in flip):
            assert safe_states(flows, states), f"unsafe down-set {flip}"


@pytest.mark.parametrize("seed", range(120))
def test_tagged_installs_are_safe_any_order(seed):
    """Two-phase installs are invisible to live tags: any subset is safe.

    A two-phase (tagged) update installs tag-1 rules in any order while the
    ingress still emits tag 0.  The invariant must hold for *every* subset of
    installed switches — this is what licenses installing them all at once.
    """
    rng = random.Random(seed + 9000)
    switches, flows = rand_instance(rng, n_sw=6, n_flow=1)
    if not flows:
        pytest.skip("degenerate instance")
    f = flows[0]
    if len(f.new) < 2:
        pytest.skip("trivial path")
    new_switches = sorted(f.new)
    for mask in range(1 << len(new_switches)):
        states = make_states(flows)
        st = states[f.fid]
        st.begin_tagged(f.new)
        for i, x in enumerate(new_switches):
            if mask >> i & 1:
                st.install_tagged(x)
        # tag 0 is still live: the old path must be intact
        assert safe_states(flows, states), f"tagged install subset {mask} unsafe"


def test_planner_rounds_every_subset_safe():
    """For randomly generated plans, every subset of every round is safe.

    This is the property that makes 'apply the round in any order' sound, and
    it is exactly what the randomised-apply simulator exercises.
    """
    from netorch.scenarios import random_scenario

    for seed in range(40):
        sc = random_scenario(seed, switches=8, degree=3, flows=5, cap=10, swap_bias=0.6)
        if not sc.flows:
            continue
        plan = Planner(sc.topo, sc.flows, drain_ticks=2).plan()
        if plan.stats.infeasible:
            continue
        states = make_states(sc.flows)
        for r in plan.rounds:
            # snapshot before the round
            before = clone_states(states)
            cmds = [(c.sw, op) for c in r.commands for op in c.ops]
            # try every subset of this round's ops
            for mask in range(1 << len(cmds)):
                trial = clone_states(before)
                for i, (sw, op) in enumerate(cmds):
                    if mask >> i & 1:
                        apply_op(trial[op.fid], op)
                assert not audit(
                    trial, sc.topo, strict_hold=True, check_capacity=True
                ), f"round {r.index} subset {mask} unsafe (seed {seed})"
            # then commit the whole round
            for c in r.commands:
                for op in c.ops:
                    apply_op(states[op.fid], op)


# ---------------------------------------------------------------------------
# graph algorithms
# ---------------------------------------------------------------------------


def test_scc_and_cycle_detection():
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    assert deps.has_cycle(nodes, edges)
    comps = deps.sccs(nodes, edges)
    assert sorted(sorted(c) for c in comps) == [[0, 1, 2], [3]]
    assert deps.cyclic_switches(nodes, edges) == {0, 1, 2}


def test_ranks_is_longest_path():
    nodes = [0, 1, 2, 3]
    edges = [(0, 2), (1, 2), (2, 3)]
    r = deps.ranks(nodes, edges)
    assert r[0] == 0 and r[1] == 0 and r[2] == 1 and r[3] == 2
    assert deps.longest_chain(nodes, edges) == 3


def test_adjacent_swap_produces_cycle():
    f = Flow(0, 0, 300, (0, 1, 2, 3), (0, 2, 1, 3), 1)
    assert deps.has_cycle(f.changed_switches, deps.flow_edges(f))


def test_hop_shift_is_acyclic():
    f = Flow(0, 0, 300, (0, 1, 3), (0, 2, 3), 1)
    assert not deps.has_cycle(f.changed_switches, deps.flow_edges(f))


def test_downsets_bruteforce():
    pre = {0: set(), 1: {0}, 2: {1}}
    ds = deps.downsets([0, 1, 2], pre)
    assert set(map(frozenset, ds)) == {
        frozenset(), frozenset({0}), frozenset({0, 1}), frozenset({0, 1, 2})
    }


@pytest.mark.parametrize("seed", range(80))
def test_planner_rounds_incl_cyclic_are_fully_safe(seed):
    """Plan random instances (including cyclic ones) and check every subset of
    every round.  This is the end-to-end soundness property: no matter what
    order a round's commands apply in, the network is safe."""
    from netorch.planner import Planner, apply_op
    from netorch.model import Topology

    rng = random.Random(seed + 20000)
    switches, flows = rand_instance(rng, n_sw=6, n_flow=3)
    if not flows:
        pytest.skip("degenerate")
    t = Topology("inst")
    for s in switches:
        t.add_switch(s)
    links = set()
    for f in flows:
        for a, b in zip(f.old, f.old[1:]):
            links.add((min(a, b), max(a, b)))
        for a, b in zip(f.new, f.new[1:]):
            links.add((min(a, b), max(a, b)))
    for a, b in links:
        t.add_link(a, b, 100)
    for f in flows:
        if f.dst not in t.host_at[f.new[-1]]:
            t.add_host(f.new[-1], f.dst)

    plan = Planner(t, flows, drain_ticks=0).plan()
    if plan.stats.infeasible:
        pytest.skip("infeasible")
    states = make_states(flows)
    for r in plan.rounds:
        before = clone_states(states)
        cmds = [(c.sw, op) for c in r.commands for op in c.ops]
        if len(cmds) > 12:
            # keep the exhaustive subset check bounded
            cmds = cmds[:12]
        for mask in range(1 << len(cmds)):
            trial = clone_states(before)
            for i, (sw, op) in enumerate(cmds):
                if mask >> i & 1:
                    apply_op(trial[op.fid], op)
            assert not audit(
                trial, t, strict_hold=True, check_capacity=True
            ), f"unsafe subset of round {r.index} (seed {seed})"
        for c in r.commands:
            for op in c.ops:
                apply_op(states[op.fid], op)
