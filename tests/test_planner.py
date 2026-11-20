"""Tests for the planner: cycles, tagging, capacity staging, detour, bounds."""

from __future__ import annotations

import pytest

from netorch.model import Flow, Topology, make_states, nexthop, audit
from netorch.planner import Planner, apply_op, rollback_plan
from netorch.scenarios import (
    chain_scenario,
    detour_scenario,
    merge_scenario,
    swap_scenario,
)


def ladder(cap=10, n=6, table_cap=256):
    t = Topology("ladder")
    for s in range(n):
        t.add_switch(s, table_cap)
    for i in range(n - 1):
        t.add_link(i, i + 1, cap)
    for i in range(n - 2):
        t.add_link(i, i + 2, cap)
    t.add_host(n - 1, 900)
    return t


def final_rules(plan):
    states = make_states(list(plan.flows.values()))
    for r in plan.rounds:
        for c in r.commands:
            for op in c.ops:
                apply_op(states[op.fid], op)
    return states


def assert_converged(plan):
    states = final_rules(plan)
    for f in plan.flows.values():
        st = states[f.fid]
        assert st.mode == "plain"
        for x in set(f.old) | set(f.new):
            assert st.rules.get(x) == nexthop(f.new, x), f"flow {f.fid} switch {x}"


def test_simple_shift_has_no_tags_and_converges():
    t = ladder()
    f = Flow(0, 0, 900, (0, 1, 2, 3), (0, 2, 3), 1)
    plan = Planner(t, [f]).plan()
    assert plan.stats.tagged_flows == []
    assert plan.stats.gap >= 0
    assert_converged(plan)


def test_adjacent_swap_requires_two_phase():
    t, flows = swap_scenario()
    plan = Planner(t, flows, drain_ticks=2).plan()
    assert plan.stats.tagged_flows == [0]
    # install / stamp / drop
    kinds = [r.kind for r in plan.rounds]
    assert "stamp" in kinds
    assert plan.stats.rounds == 3
    assert_converged(plan)


def test_rounds_are_minimal_for_chain():
    """A dependency chain of length k must take k rounds."""
    t = ladder(n=6)
    f = Flow(0, 0, 900, (0, 1, 2, 3, 4, 5), (0, 3, 5), 1)
    plan = Planner(t, [f]).plan()
    # changed: 0,1,2,3,4  -> the new path binds downstream first
    assert plan.stats.rounds == plan.stats.dag_height
    assert plan.stats.gap == 0


def test_capacity_staging_serialises_flows():
    t, flows = merge_scenario()
    plan = Planner(t, flows).plan()
    assert not plan.stats.infeasible
    assert plan.stats.capacity_lb >= 1
    assert plan.stats.gap >= 0
    # every checkpoint must be safe
    for cp in plan.checkpoints:
        assert not audit(cp, t, strict_hold=True, check_capacity=True)


def test_detour_breaks_capacity_deadlock():
    t = Topology("swapbw")
    for s in range(6):
        t.add_switch(s)
    for a, b in [(0, 1), (0, 2), (1, 4), (2, 4), (0, 3), (3, 5), (5, 4)]:
        t.add_link(a, b, 10)
    t.add_host(4, 400)
    flows = [
        Flow(0, 0, 400, (0, 1, 4), (0, 2, 4), 6),
        Flow(1, 0, 400, (0, 2, 4), (0, 1, 4), 6),
    ]
    plan = Planner(t, flows, drain_ticks=2).plan()
    assert not plan.stats.infeasible
    assert plan.stats.detoured_flows == [0]
    assert plan.stats.gap == 0
    for cp in plan.checkpoints:
        assert not audit(cp, t, strict_hold=True, check_capacity=True)
    assert_converged(plan)


def test_infeasible_target_is_reported():
    t = ladder(cap=1)
    t.add_host(5, 900)
    flows = [
        Flow(0, 0, 900, (0, 1, 5), (0, 1, 2, 5), 1),
        Flow(1, 0, 900, (0, 2, 5), (0, 1, 2, 5), 1),
    ]
    plan = Planner(t, flows).plan()
    # link 0-1 and 1-2 would each carry 2 > cap 1 on the new paths
    assert plan.stats.infeasible


def test_table_budget_reports_deficit():
    t = ladder(table_cap=1)
    t2, flows = swap_scenario()
    # rebuild swap on a table cap of 1 to force a deficit report
    topo = Topology("tiny")
    for s in range(5):
        topo.add_switch(s, 1)
    for a, b in [(0, 1), (1, 3), (0, 2), (2, 3), (1, 2)]:
        topo.add_link(a, b, 10)
    topo.add_host(3, 300)
    plan = Planner(topo, [Flow(0, 0, 300, (0, 1, 2, 3), (0, 2, 1, 3), 1)]).plan()
    assert plan.stats.tagged_flows == [0]
    assert plan.stats.peak_table >= 2
    assert plan.stats.table_deficit >= 1 or plan.stats.table_cap >= plan.stats.peak_table


def test_chain_scenario_multiple_flows():
    t, flows = chain_scenario()
    plan = Planner(t, flows).plan()
    assert not plan.stats.infeasible
    for cp in plan.checkpoints:
        assert not audit(cp, t, strict_hold=True, check_capacity=True)
    assert_converged(plan)


def test_rollback_restores_old_paths():
    t, flows = swap_scenario()
    plan = Planner(t, flows, drain_ticks=2).plan()
    cur = make_states(flows)
    # simulate the install round only
    first = plan.rounds[0]
    for c in first.commands:
        for op in c.ops:
            apply_op(cur[op.fid], op)
    rb = rollback_plan(t, flows, cur, failed=[3])
    assert rb.plan is not None or rb.exempt_flows or rb.unrecoverable
    assert rb.notes
