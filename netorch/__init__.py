"""netorch: safe orchestration of flow-table updates for a cloud-gaming fabric.

Public surface
--------------
``model``       topology, flows, per-instant forwarding state, audit invariants
``deps``        dependency extraction and graph algorithms
``planner``     one-phase / two-phase / detour / rollback planning
``sim``         event-driven simulator with randomised, uncertain apply times
``scenarios``   randomised scenario generation
``experiments`` batch acceptance runs, lower bound, baselines
``replay``      local web replay of a single update
"""

from .model import (
    HOST,
    TAG_NEW,
    TAG_OLD,
    Flow,
    FState,
    Topology,
    Violation,
    audit,
    clone_states,
    make_states,
    nexthop,
)
from .planner import (
    Command,
    Op,
    Plan,
    PlanStats,
    Planner,
    RollbackResult,
    Round,
    rollback_plan,
)

__all__ = [
    "HOST",
    "TAG_OLD",
    "TAG_NEW",
    "Flow",
    "FState",
    "Topology",
    "Violation",
    "audit",
    "clone_states",
    "make_states",
    "nexthop",
    "Planner",
    "Plan",
    "Round",
    "Command",
    "Op",
    "PlanStats",
    "RollbackResult",
    "rollback_plan",
]
__version__ = "0.1.0"
