"""Safe retreat after a switch failure.

The main planner proves safety by construction from a dependency DAG.  A
failure happens *inside* a plan, so the retreat starts from an arbitrary
safe-but-partial state.  Rather than trying to reuse the DAG theory on a state
it was not derived for, the retreat uses **runtime verification**: each
candidate step is applied to a copy of the live state and the full invariant
audit is run on it.  A step is taken only if it is safe, so the retreat is safe
by construction whatever state the failure left behind.

The target of the retreat is the flows' **old paths**.  Because a failure can
leave switches on a mix of old and new rules (some switched, some not), the
retreat works on the explicit per-switch rule map and flips the switches that
still disagree with the old path, one safe batch at a time.

Rules that keep the retreat short:

* Prefer switches that are already holding a *new-path* rule — restoring them
  is what actually un-does the half-done update.
* Batch independent commands into one round, accepting the batch only if
  **every subset** of it is safe (the criterion the main planner uses, since a
  round's commands may apply in any order).
* Never touch a failed switch.  A flow that cannot possibly be made safe
  because its live path crosses the failure is reported (exempt) rather than
  counted as a retreat violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .model import (
    Flow,
    FState,
    Topology,
    audit,
    clone_states,
    nexthop,
    nm,
)
from .planner import Command, Op, Round, apply_op

MAX_BATCH = 6          # largest candidate batch whose every subset is checked


@dataclass
class Retreat:
    rounds: List[Round]
    exempt: List[int]
    unrecoverable: List[int]
    notes: List[str]
    start_states: Dict[int, FState] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "rounds": [r.snapshot() for r in self.rounds],
            "exempt": list(self.exempt),
            "unrecoverable": list(self.unrecoverable),
            "notes": list(self.notes),
        }


def _safe(
    states: Mapping[int, FState],
    topo: Topology,
    exempt: Set[int],
    dead: Set[int],
) -> bool:
    return not audit(
        states,
        topo,
        strict_hold=True,
        check_capacity=True,
        exempt=exempt,
        failed_switches=dead,
    )


def _restore_ops(
    states: Mapping[int, FState],
    flows: Mapping[int, Flow],
    dead: Set[int],
    exempt: Set[int],
) -> List[Tuple[Op, Optional[int]]]:
    """(op, desired rule) pairs that move each flow back towards its old path."""
    out: List[Tuple[Op, Optional[int]]] = []
    for fid, st in states.items():
        if fid in exempt:
            continue
        f = flows[fid]
        if st.mode == "tagged":
            # dropping the tag-1 rules and reverting the stamp is the retreat
            for x in sorted(st.new_inst):
                if x in dead:
                    continue
                out.append((Op("drop", fid, x, f.old), None))
            continue
        for x in sorted(set(st.rules) | set(f.old) | set(f.new)):
            if x in dead:
                continue
            want = nexthop(f.old, x)
            if st.rules.get(x) == want:
                continue
            out.append((Op("flip", fid, x, f.old, want), want))
    return out


def plan_retreat(
    topo: Topology,
    flows: Sequence[Flow],
    states: Mapping[int, FState],
    dead: Iterable[int],
) -> Retreat:
    """A verified-safe retreat from ``states`` towards every flow's old path."""
    dead = set(dead)
    flows_d = {f.fid: f for f in flows}
    cur = clone_states(states)
    notes: List[str] = []
    exempt: Set[int] = set()
    unrecoverable: Set[int] = set()

    # classify the flows the failure makes hopeless
    for fid, st in cur.items():
        f = flows_d[fid]
        live = st.committed_path()
        if set(live) & dead:
            exempt.add(fid)
            notes.append(
                f"flow {fid}: its live path {_p(live)} crosses a failed switch, so it "
                "is already dropping and table updates cannot rescue it (exempt)"
            )
            continue
        if set(f.old) & dead:
            unrecoverable.add(fid)
            notes.append(
                f"flow {fid}: its old path {_p(f.old)} crosses a failed switch, so the "
                "retreat cannot restore it (unrecoverable)"
            )

    rounds: List[Round] = []
    for _guard in range(4000):
        cand = _restore_ops(cur, flows_d, dead, exempt)
        if not cand:
            break
        batch: List[Tuple[Op, Optional[int]]] = []
        for item in cand:
            if _batch_ok(cur, batch + [item], topo, exempt, dead):
                batch.append(item)
            if len(batch) >= MAX_BATCH:
                break
        if not batch:
            stuck = sorted({op.fid for op, _ in cand})
            notes.append(
                f"retreat has no safe single step for flows {stuck}; exempting them"
            )
            exempt |= set(stuck)
            continue
        cmds: Dict[int, List[Op]] = {}
        for op, _ in batch:
            apply_op(cur[op.fid], op)
            cmds.setdefault(op.sw, []).append(op)
        rounds.append(
            Round(
                index=len(rounds),
                commands=[Command(sw, ops) for sw, ops in sorted(cmds.items())],
                kind="rollback",
                label=f"retreat round {len(rounds) + 1}",
            )
        )
        if len(rounds) > 200:
            notes.append("retreat exceeded 200 rounds; stopping")
            break

    return Retreat(rounds, sorted(exempt), sorted(unrecoverable), notes, clone_states(states))


def _p(path: Sequence[int]) -> str:
    return "->".join(nm(x) for x in path)


def _batch_ok(
    states: Mapping[int, FState],
    batch: Sequence[Tuple[Op, Optional[int]]],
    topo: Topology,
    exempt: Set[int],
    dead: Set[int],
) -> bool:
    """Every subset of the batch must leave the network safe."""
    if len(batch) > MAX_BATCH:
        return False
    n = len(batch)
    for mask in range(1, 1 << n):
        trial = clone_states(states)
        for i in range(n):
            if mask >> i & 1:
                apply_op(trial[batch[i][0].fid], batch[i][0])
        if not _safe(trial, topo, exempt, dead):
            return False
    return True
