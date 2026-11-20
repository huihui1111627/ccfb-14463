"""Tests for the simulator: randomised apply order, violations, failures.

These are the acceptance properties in miniature: with apply order shuffled,
the planner's plan must never violate an invariant, while the "push everything
at once" baseline must violate them on at least some seeds.
"""

from __future__ import annotations

import random

import pytest

from netorch.model import audit, make_states
from netorch.planner import Planner
from netorch.sim import RunningPlan, Timing, simulate_baseline
from netorch.scenarios import (
    detour_scenario,
    merge_scenario,
    random_scenario,
    swap_scenario,
)


@pytest.mark.parametrize("seed", range(30))
def test_plan_is_safe_under_random_apply_order(seed):
    sc = random_scenario(seed, switches=9, degree=3, flows=5, cap=10, swap_bias=0.6)
    if not sc.flows:
        pytest.skip("empty scenario")
    plan = Planner(sc.topo, sc.flows, drain_ticks=2).plan()
    if plan.stats.infeasible:
        pytest.skip("infeasible target")
    for k in range(3):
        res = RunningPlan(
            sc.topo, sc.flows, plan.rounds, Timing(), random.Random(seed * 31 + k)
        ).run()
        assert res.violation_ticks == 0, res.violations[:3]


@pytest.mark.parametrize("name", ["swap", "merge", "detour"])
def test_hand_scenarios_are_safe(name):
    topo, flows = {
        "swap": swap_scenario,
        "merge": merge_scenario,
        "detour": detour_scenario,
    }[name]()
    plan = Planner(topo, flows, drain_ticks=2).plan()
    assert not plan.stats.infeasible
    for k in range(5):
        res = RunningPlan(topo, flows, plan.rounds, Timing(), random.Random(k)).run()
        assert res.violation_ticks == 0


def test_baseline_has_violations_somewhere():
    """The whole point: uncontrolled pushes do violate the invariants."""
    hits = 0
    for seed in range(40):
        sc = random_scenario(seed, switches=9, degree=3, flows=5, cap=10, swap_bias=0.6)
        if not sc.flows:
            continue
        b = simulate_baseline(sc.topo, sc.flows, random.Random(seed))
        if b.violation_ticks > 0:
            hits += 1
    assert hits > 0


def test_baseline_loses_packets():
    topo, flows = swap_scenario()
    b = simulate_baseline(topo, flows, random.Random(0))
    assert b.violation_ticks + b.packets_lost + b.packets_looped > 0


def test_failure_triggers_retreat_and_never_violates():
    topo, flows = swap_scenario()
    plan = Planner(topo, flows, drain_ticks=2).plan()
    sws = sorted({c.sw for r in plan.rounds for c in r.commands})
    for sw in sws:
        for ft in (1, 2, 3, 4):
            res = RunningPlan(topo, flows, plan.rounds, Timing(), random.Random(1)).run(
                fail_switch=sw, fail_tick=ft
            )
            # After the failure the retreat must be safe for every non-exempt flow
            assert res.rollback_violation_ticks == 0 or set(
                f.fid for f in flows
            ) <= set(res.rollback_exempt)


def test_simulation_finishes_and_converges():
    topo, flows = merge_scenario()
    plan = Planner(topo, flows, drain_ticks=2).plan()
    res = RunningPlan(topo, flows, plan.rounds, Timing(), random.Random(0)).run()
    assert res.finished
    assert res.packets_injected > 0
    assert res.packets_lost == 0


def test_trailing_events_reach_destination():
    topo, flows = swap_scenario()
    plan = Planner(topo, flows, drain_ticks=2).plan()
    res = RunningPlan(topo, flows, plan.rounds, Timing(), random.Random(2)).run()
    # the very last frame must be fully on the new path
    last = res.trail_trace[-1]
    for fl in last["flows"]:
        assert fl["ok"]
        assert fl["trail"][-1] == -1
