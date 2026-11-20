"""Batch acceptance runs and metric aggregation.

The acceptance harness does what the task asks for:

* generates thousands of *different* update tasks (``scenarios.random_scenario``
  plus the hand-built stress cases);
* runs each plan in the simulator with **randomised switch apply times** and
  audits at **every tick** for loops, blackholes and link overloads;
* runs the same task through the "push everything at once" baseline and counts
  its violation ticks the same way;
* records round counts, the gap to the theoretical lower bound, failure
  injection and rollback outcomes;
* aggregates everything into one JSON report the replay page and the README
  quotes.

Everything is seeded, so a report is reproducible.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Flow, Topology
from .planner import Plan, Planner
from .scenarios import (
    chain_scenario,
    detour_family,
    detour_scenario,
    merge_scenario,
    random_scenario,
    swap_scenario,
)
from .sim import RunningPlan, Timing, simulate_baseline


# ---------------------------------------------------------------------------
# per-task record
# ---------------------------------------------------------------------------


@dataclass
class TaskRecord:
    task: str
    seed: int
    flows: int
    changing: int
    switches: int
    links: int
    # planning
    infeasible: str = ""
    rounds: int = 0
    lower_bound: int = 0
    gap: int = 0
    tagged: int = 0
    detoured: int = 0
    peak_table: int = 0
    table_deficit: int = 0
    # simulation, main
    plan_runs: int = 0
    plan_violation_ticks: int = 0
    plan_runs_with_violation: int = 0
    plan_packets_lost: int = 0
    plan_packets_looped: int = 0
    plan_packets_injected: int = 0
    plan_make_span: float = 0.0
    # simulation, baseline (one run per task)
    base_violation_ticks: int = 0
    base_packets_lost: int = 0
    base_packets_looped: int = 0
    # failure injection
    fail_runs: int = 0
    fail_rollback_runs: int = 0
    fail_unrecoverable: int = 0
    fail_exempt: int = 0

    def snapshot(self) -> dict:
        d = dict(self.__dict__)
        return d


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def _plan_for(topo: Topology, flows: Sequence[Flow], drain: int = 2) -> Plan:
    return Planner(topo, flows, drain_ticks=drain).plan()


def run_task(
    name: str,
    topo: Topology,
    flows: Sequence[Flow],
    seed: int,
    plan_runs: int = 4,
    fail_runs: int = 2,
    timing: Optional[Timing] = None,
    baseline: bool = True,
) -> TaskRecord:
    rec = TaskRecord(
        task=name,
        seed=seed,
        flows=len(flows),
        changing=sum(1 for f in flows if f.old != f.new),
        switches=len(topo.switches),
        links=len(topo.links),
    )
    plan = _plan_for(topo, flows)
    if plan.stats.infeasible:
        rec.infeasible = plan.stats.infeasible
        return rec
    st = plan.stats
    rec.rounds = st.rounds
    rec.lower_bound = st.lower_bound
    rec.gap = st.gap
    rec.tagged = len(st.tagged_flows)
    rec.detoured = len(st.detoured_flows)
    rec.peak_table = st.peak_table
    rec.table_deficit = st.table_deficit

    tm = timing or Timing()
    for k in range(plan_runs):
        rng = random.Random(seed * 1_000_003 + k)
        res = RunningPlan(topo, flows, plan.rounds, tm, rng).run()
        rec.plan_runs += 1
        rec.plan_violation_ticks += res.violation_ticks
        rec.plan_packets_lost += res.packets_lost
        rec.plan_packets_looped += res.packets_looped
        rec.plan_packets_injected += res.packets_injected
        rec.plan_make_span += res.make_span
        if res.violation_ticks:
            rec.plan_runs_with_violation += 1

    if baseline:
        b = simulate_baseline(topo, flows, random.Random(seed * 7 + 1))
        rec.base_violation_ticks = b.violation_ticks
        rec.base_packets_lost = b.packets_lost
        rec.base_packets_looped = b.packets_looped

    if fail_runs and plan.rounds:
        for k in range(fail_runs):
            rng = random.Random(seed * 999_983 + k)
            # fail a switch that the plan actually touches, somewhere mid-plan
            sws = sorted({c.sw for r in plan.rounds for c in r.commands})
            if not sws:
                break
            sw = rng.choice(sws)
            res = RunningPlan(topo, flows, plan.rounds, tm, rng).run(
                fail_switch=sw, fail_tick=rng.randint(1, max(1, 3 * len(plan.rounds)))
            )
            rec.fail_runs += 1
            if res.rollback_used:
                rec.fail_rollback_runs += 1
            rec.fail_unrecoverable += len(res.rollback_unrecoverable)
            rec.fail_exempt += len(res.rollback_exempt)
    return rec


def scenario_batch(
    count: int = 120,
    seed0: int = 0,
    plan_runs: int = 3,
    fail_runs: int = 1,
    tight_every: int = 3,
    **kwargs,
) -> List[TaskRecord]:
    out: List[TaskRecord] = []
    for i in range(count):
        seed = seed0 + i
        sc = random_scenario(
            seed,
            tight=(seed % tight_every == 0),
            **kwargs,
        )
        if not sc.flows:
            continue
        out.append(
            run_task(
                f"rand-{seed}",
                sc.topo,
                sc.flows,
                seed,
                plan_runs=plan_runs,
                fail_runs=fail_runs,
            )
        )
    return out


def hand_batch(plan_runs: int = 4, fail_runs: int = 2) -> List[TaskRecord]:
    out = []
    for name, (topo, flows) in [
        ("merge", merge_scenario()),
        ("swap", swap_scenario()),
        ("detour", detour_scenario()),
        ("chain", chain_scenario()),
    ]:
        stable = sum(ord(c) * (i + 1) for i, c in enumerate(name)) & 0xFFFF
        out.append(run_task(name, topo, flows, stable, plan_runs, fail_runs))
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate(records: Sequence[TaskRecord]) -> dict:
    ok = [r for r in records if not r.infeasible]
    infeasible = [r for r in records if r.infeasible]
    total_plan_ticks = sum(r.plan_violation_ticks for r in ok)
    total_base_ticks = sum(r.base_violation_ticks for r in ok)
    runs = sum(r.plan_runs for r in ok)
    rounds = [r.rounds for r in ok]
    gaps = [r.gap for r in ok]
    summary = {
        "tasks": len(records),
        "tasks_planned": len(ok),
        "tasks_infeasible": len(infeasible),
        "infeasible_reasons": sorted({r.infeasible for r in infeasible}),
        "plan_runs": runs,
        "plan_violation_ticks": total_plan_ticks,
        "plan_runs_with_violation": sum(r.plan_runs_with_violation for r in ok),
        "baseline_violation_ticks": total_base_ticks,
        "baseline_over_plan_ratio": (
            round(total_base_ticks / total_plan_ticks, 2) if total_plan_ticks else None
        ),
        "plan_packets_lost": sum(r.plan_packets_lost for r in ok),
        "plan_packets_looped": sum(r.plan_packets_looped for r in ok),
        "baseline_packets_lost": sum(r.base_packets_lost for r in ok),
        "baseline_packets_looped": sum(r.base_packets_looped for r in ok),
        "avg_rounds": round(statistics.mean(rounds), 2) if rounds else 0,
        "median_rounds": statistics.median(rounds) if rounds else 0,
        "max_rounds": max(rounds) if rounds else 0,
        "avg_lower_bound": round(statistics.mean([r.lower_bound for r in ok]), 2) if ok else 0,
        "avg_gap": round(statistics.mean(gaps), 2) if gaps else 0,
        "max_gap": max(gaps) if gaps else 0,
        "tasks_with_tagged_flows": sum(1 for r in ok if r.tagged),
        "tasks_with_detour": sum(1 for r in ok if r.detoured),
        "tasks_with_table_deficit": sum(1 for r in ok if r.table_deficit),
        "fail_runs": sum(r.fail_runs for r in ok),
        "fail_rollback_runs": sum(r.fail_rollback_runs for r in ok),
        "fail_unrecoverable": sum(r.fail_unrecoverable for r in ok),
        "fail_exempt": sum(r.fail_exempt for r in ok),
    }
    # by task family
    fam: Dict[str, List[TaskRecord]] = {}
    for r in ok:
        key = r.task.split("-")[0]
        fam.setdefault(key, []).append(r)
    summary["by_family"] = {
        k: {
            "tasks": len(v),
            "avg_rounds": round(statistics.mean([x.rounds for x in v]), 2),
            "avg_gap": round(statistics.mean([x.gap for x in v]), 2),
            "plan_violation_ticks": sum(x.plan_violation_ticks for x in v),
            "baseline_violation_ticks": sum(x.base_violation_ticks for x in v),
        }
        for k, v in sorted(fam.items())
    }
    return summary


def run_acceptance(
    count: int = 120,
    plan_runs: int = 3,
    fail_runs: int = 1,
    include_hand: bool = True,
    **kwargs,
) -> Tuple[List[TaskRecord], dict]:
    """Run the hand-built stress cases plus ``count`` random tasks."""
    t0 = time.time()
    records = []
    if include_hand:
        records += hand_batch(plan_runs=max(plan_runs, 4), fail_runs=max(fail_runs, 2))
    records += scenario_batch(count=count, plan_runs=plan_runs, fail_runs=fail_runs, **kwargs)
    summary = aggregate(records)
    summary["elapsed_s"] = round(time.time() - t0, 2)
    return records, summary


def big_sweep(
    families: int = 6,
    per_family: int = 50,
    plan_runs: int = 6,
    fail_runs: int = 2,
) -> Tuple[List[TaskRecord], dict]:
    """A larger sweep that varies the generator across families.

    Each family changes fabric size, flow count, capacity tightness and how
    often the new path is a small perturbation (which drives cycles and thus
    version-tagged updates).
    """
    families_spec = [
        dict(switches=8, degree=3, flows=4, cap=12, tight=False, swap_bias=0.2),
        dict(switches=8, degree=3, flows=5, cap=10, tight=False, swap_bias=0.7),
        dict(switches=10, degree=3, flows=6, cap=10, tight=True, swap_bias=0.5),
        dict(switches=10, degree=4, flows=8, cap=8, tight=True, swap_bias=0.6),
        dict(switches=12, degree=3, flows=8, cap=10, tight=False, swap_bias=0.8),
        dict(switches=14, degree=3, flows=10, cap=12, tight=True, swap_bias=0.4),
    ][:families]
    t0 = time.time()
    records: List[TaskRecord] = hand_batch(plan_runs=8, fail_runs=4)
    for k in range(max(8, per_family // 8)):
        topo, flows = detour_family(k, name=f"detourfam{k}")
        records.append(
            run_task(f"detourfam-{k}", topo, flows, 900_000 + k,
                     plan_runs=plan_runs, fail_runs=fail_runs)
        )
    seed = 0
    for fi, spec in enumerate(families_spec):
        for j in range(per_family):
            sc = random_scenario(seed, name=f"fam{fi}", **spec)
            seed += 1
            if not sc.flows:
                continue
            records.append(
                run_task(f"fam{fi}-{j}", sc.topo, sc.flows, seed, plan_runs=plan_runs,
                         fail_runs=fail_runs)
            )
    summary = aggregate(records)
    summary["elapsed_s"] = round(time.time() - t0, 2)
    return records, summary


def write_report(records: Sequence[TaskRecord], summary: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "summary": summary,
                "records": [r.snapshot() for r in records],
            },
            fh,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


if __name__ == "__main__":  # pragma: no cover
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    recs, summ = run_acceptance(count=n)
    json.dump(summ, sys.stdout, indent=2)
    sys.stdout.write("\n")
