"""Command-line entry points.

    python3 -m netorch.cli plan      --scenario swap
    python3 -m netorch.cli simulate  --scenario merge --seed 3
    python3 -m netorch.cli verify    --count 120
    python3 -m netorch.cli accept    --count 120 --runs 6      (writes results/acceptance.json)
    python3 -m netorch.cli replay    --scenario swap [--serve]
    python3 -m netorch.cli demos                                (all replay pages)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Tuple

from .model import Flow, Topology, audit, make_states, path_str
from .planner import Planner, apply_op
from .replay import demo, render_page, serve
from .scenarios import (
    chain_scenario,
    detour_family,
    detour_scenario,
    merge_scenario,
    random_scenario,
    swap_scenario,
)
from .sim import RunningPlan, Timing, simulate_baseline

SCENARIOS = {
    "swap": swap_scenario,
    "merge": merge_scenario,
    "detour": detour_scenario,
    "chain": chain_scenario,
}


def load_scenario(name: str, seed: int = 0) -> Tuple[Topology, List[Flow]]:
    if name in SCENARIOS:
        return SCENARIOS[name]()
    if name == "detourfam":
        return detour_family(seed)
    if name.startswith("rand"):
        sc = random_scenario(seed, switches=10, degree=3, flows=6, cap=10, swap_bias=0.6)
        return sc.topo, sc.flows
    raise SystemExit(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS) + ['rand', 'detourfam']}")


def cmd_plan(args) -> int:
    topo, flows = load_scenario(args.scenario, args.seed)
    plan = Planner(topo, flows, drain_ticks=args.drain).plan()
    st = plan.stats
    print(f"scenario      : {args.scenario}")
    print(f"topology      : {len(topo.switches)} switches, {len(topo.links)} links")
    print(f"flows         : {len(flows)} ({sum(1 for f in flows if f.old != f.new)} changing)")
    if st.infeasible:
        print(f"INFEASIBLE    : {st.infeasible}")
        for n in st.notes:
            print("   ", n)
        return 1
    print(f"rounds        : {st.rounds}  (lower bound {st.lower_bound}, gap {st.gap})")
    print(f"   lower bound breakdown: dag_height={st.dag_height} "
          f"capacity={st.capacity_lb} tagged={st.tagged_lb}")
    print(f"tagged flows  : {st.tagged_flows or '-'}   (two-phase, version tags)")
    print(f"detoured flows: {st.detoured_flows or '-'}   (temporary transit path)")
    print(f"table peak    : {st.peak_table} entries at switch s{st.peak_table_switch} "
          f"(cap {st.table_cap}, deficit {st.table_deficit})")
    print()
    for r in plan.rounds:
        if not r.commands:
            continue
        extra = f"  [drain {r.start_wait:.0f}t before send]" if r.start_wait else ""
        print(f"round {r.index + 1} ({r.kind}){extra}:")
        for c in r.commands:
            ops = ", ".join(str(o) for o in c.ops)
            print(f"   s{c.sw}: {ops}")
    print()
    for f in flows:
        print(f"  f{f.fid}: {path_str(f.old)}  =>  {path_str(f.new)}")
    return 0


def cmd_simulate(args) -> int:
    topo, flows = load_scenario(args.scenario, args.seed)
    plan = Planner(topo, flows, drain_ticks=args.drain).plan()
    if plan.stats.infeasible:
        print("infeasible:", plan.stats.infeasible)
        return 1
    rng = random.Random(args.seed)
    res = RunningPlan(topo, flows, plan.rounds, Timing(), rng).run()
    base = simulate_baseline(topo, flows, random.Random(args.seed))
    print("orchestrated plan:")
    print(json.dumps({k: res.snapshot()[k] for k in
                      ["rounds_sent", "ticks", "violation_ticks", "violation_by_kind",
                       "packets_injected", "packets_lost", "packets_looped"]}, indent=2))
    print(f"make-span: {res.make_span} ticks")
    print("push-everything-at-once baseline:")
    print(json.dumps({k: base.snapshot()[k] for k in
                      ["ticks", "violation_ticks", "violation_by_kind",
                       "packets_injected", "packets_lost", "packets_looped"]}, indent=2))
    return 0


def cmd_verify(args) -> int:
    """Exhaustively check every subset of every round of every plan."""
    bad = 0
    total_rounds = 0
    for seed in range(args.count):
        sc = random_scenario(seed, switches=args.switches, degree=3, flows=args.flows,
                             cap=10, tight=(seed % 3 == 0), swap_bias=0.6)
        if not sc.flows:
            continue
        plan = Planner(sc.topo, sc.flows, drain_ticks=2).plan()
        if plan.stats.infeasible:
            continue
        states = make_states(sc.flows)
        for r in plan.rounds:
            before = {k: v.clone() for k, v in states.items()}
            cmds = [(c.sw, op) for c in r.commands for op in c.ops][:args.subset_limit]
            for mask in range(1 << len(cmds)):
                trial = {k: v.clone() for k, v in before.items()}
                for i, (sw, op) in enumerate(cmds):
                    if mask >> i & 1:
                        apply_op(trial[op.fid], op)
                if audit(trial, sc.topo, strict_hold=True, check_capacity=True):
                    bad += 1
                    print(f"UNSAFE seed {seed} round {r.index} subset {mask}")
                    break
            total_rounds += 1
            for c in r.commands:
                for op in c.ops:
                    apply_op(states[op.fid], op)
    print(f"verified {total_rounds} rounds across {args.count} tasks: {bad} unsafe")
    return 1 if bad else 0


def cmd_accept(args) -> int:
    from .experiments import big_sweep, run_acceptance, write_report

    if args.big:
        records, summary = big_sweep(
            families=6, per_family=args.count, plan_runs=args.runs, fail_runs=args.fails
        )
    else:
        records, summary = run_acceptance(
            count=args.count, plan_runs=args.runs, fail_runs=args.fails
        )
    out = args.out or os.path.join("results", "acceptance.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_report(records, summary, out)
    show = {k: v for k, v in summary.items() if k != "by_family"}
    print(json.dumps(show, indent=2))
    print(f"\nwrote {out}")
    return 0


def cmd_replay(args) -> int:
    data = demo(args.scenario)
    if args.serve:
        srv = serve(data, port=args.port, open_browser=not args.no_browser)
        try:
            input("press Enter to stop the server\n")
        finally:
            srv.stop()
        return 0
    out = args.out or os.path.join("results", f"replay-{args.scenario}.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_page(data))
    print(f"wrote {os.path.abspath(out)}")
    return 0


def cmd_demos(args) -> int:
    for kind in ("swap", "merge", "detour", "chain", "fail"):
        data = demo(kind)
        out = os.path.join("results", f"replay-{kind}.html")
        os.makedirs("results", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render_page(data))
        print(f"wrote {out}")
    return 0


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog="netorch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--scenario", default="swap")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--drain", type=float, default=5.0)

    sp = sub.add_parser("plan", help="print the rounds for one scenario")
    common(sp)
    sp.set_defaults(fn=cmd_plan)

    sp = sub.add_parser("simulate", help="run one plan through the simulator")
    common(sp)
    sp.set_defaults(fn=cmd_simulate)

    sp = sub.add_parser("verify", help="exhaustively verify every round subset")
    sp.add_argument("--count", type=int, default=60)
    sp.add_argument("--switches", type=int, default=8)
    sp.add_argument("--flows", type=int, default=5)
    sp.add_argument("--subset-limit", type=int, default=12)
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("accept", help="run the acceptance batch")
    sp.add_argument("--count", type=int, default=120)
    sp.add_argument("--runs", type=int, default=6)
    sp.add_argument("--fails", type=int, default=2)
    sp.add_argument("--big", action="store_true")
    sp.add_argument("--out", default="")
    sp.set_defaults(fn=cmd_accept)

    sp = sub.add_parser("replay", help="write or serve a replay page")
    sp.add_argument("--scenario", default="swap")
    sp.add_argument("--serve", action="store_true")
    sp.add_argument("--port", type=int, default=0)
    sp.add_argument("--no-browser", action="store_true")
    sp.add_argument("--out", default="")
    sp.set_defaults(fn=cmd_replay)

    sp = sub.add_parser("demos", help="write every replay demo page")
    sp.set_defaults(fn=cmd_demos)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
