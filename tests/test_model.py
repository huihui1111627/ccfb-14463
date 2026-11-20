"""Tests for the model: forwarding semantics, audit invariants, table budget."""

from __future__ import annotations

import pytest

from netorch.model import (
    HOST,
    TAG_NEW,
    TAG_OLD,
    Flow,
    FState,
    Topology,
    audit,
    link_load,
    make_states,
    nexthop,
    table_usage,
    walk,
)


def ladder(cap=10, n=5, chords=False):
    t = Topology("ladder")
    for s in range(n):
        t.add_switch(s)
    for i in range(n - 1):
        t.add_link(i, i + 1, cap)
    if chords:
        for i in range(n - 2):
            t.add_link(i, i + 2, cap)
    t.add_host(n - 1, 900)
    return t


def test_flow_rejects_non_simple_path():
    with pytest.raises(ValueError):
        Flow(0, 0, 9, (0, 1, 0, 2), (0, 2), 1)


def test_flow_rejects_wrong_ingress():
    with pytest.raises(ValueError):
        Flow(0, 3, 9, (0, 1, 2), (0, 2), 1)


def test_nexthop_semantics():
    p = (0, 1, 2)
    assert nexthop(p, 0) == 1
    assert nexthop(p, 1) == 2
    assert nexthop(p, 2) == HOST
    assert nexthop(p, 7) is None


def test_changed_switches():
    f = Flow(0, 0, 9, (0, 1, 3), (0, 2, 3), 1)
    assert f.changed_switches == {0, 1, 2}
    assert f.switches == {0, 1, 2, 3}


def test_audit_clean_at_start_and_end():
    t = ladder(chords=True)
    f = Flow(0, 0, 900, (0, 1, 2, 3, 4), (0, 2, 4), 1)
    states = make_states([f])
    assert audit(states, t) == []
    # install the whole new path
    for x in f.changed_switches:
        states[0].set_rule(x, nexthop(f.new, x))
    assert audit(states, t) == []


def test_audit_catches_latent_blackhole():
    """A holder whose rule sends packets to a switch with *no* rule is unsafe."""
    t = ladder(chords=True)
    f = Flow(0, 0, 900, (0, 1, 2, 3, 4), (0, 2, 4), 1)
    states = make_states([f])
    # switch 2 has dropped the flow (it is on neither path any more), but
    # switch 1 still holds its old rule pointing at switch 2.
    states[0].set_rule(2, None)
    vs = audit(states, t)
    assert any(v.kind == "blackhole" for v in vs)


def test_audit_catches_loop():
    t = Topology("ring")
    for s in range(3):
        t.add_switch(s)
    t.add_link(0, 1, 10)
    t.add_link(1, 2, 10)
    t.add_link(0, 2, 10)
    t.add_host(2, 900)
    f = Flow(0, 0, 900, (0, 1, 2), (0, 2), 1)
    states = make_states([f])
    # make a loop: 0 -> 1, 1 -> 0
    states[0].set_rule(0, 1)
    states[0].set_rule(1, 0)
    vs = audit(states, t)
    assert any(v.kind == "loop" for v in vs)


def test_tagged_holders_are_tag_aware():
    """A tag-1-only switch must not be required to hold the tag-0 rule."""
    t = ladder()
    f = Flow(0, 0, 900, (0, 1, 2, 3, 4), (0, 2, 4), 1)
    st = FState.fresh(f)
    st.begin_tagged(f.new)
    # install tag-1 only on a switch that is *not* on the old path
    st.install_tagged(2)
    states = {0: st}
    ok, trail, _ = walk(states, 0, TAG_OLD)
    assert ok  # the tag-0 path is untouched
    assert audit(states, t) == []


def test_old_rule_drop_is_safe_after_stamp():
    t = ladder(chords=True)
    f = Flow(0, 0, 900, (0, 1, 2, 3, 4), (0, 2, 4), 1)
    st = FState.fresh(f)
    st.begin_tagged(f.new)
    for x in f.new:
        st.install_tagged(x)
    st.flip_stamp()
    st.drop_old()
    states = {0: st}
    assert st.mode == "plain"
    assert audit(states, t) == []


def test_link_load_counts_flow_once():
    t = ladder(cap=1, chords=True)
    f = Flow(0, 0, 900, (0, 1, 2, 3, 4), (0, 2, 4), 1)
    states = make_states([f])
    load = link_load(states, t)
    assert all(v == 1 for v in load.values())


def test_table_usage_tagged_counts_two_entries():
    t = ladder(chords=True)
    f = Flow(0, 0, 900, (0, 1, 2, 3, 4), (0, 2, 4), 1)
    st = FState.fresh(f)
    st.begin_tagged(f.new)
    for x in f.new:
        st.install_tagged(x)
    used = table_usage({0: st})
    # switch 2 is on both paths and holds both rules
    assert used[2] == 2
    # switch 1 is old-only
    assert used[1] == 1


def test_topology_without_removes_switch():
    t = ladder()
    t2 = t.without([2])
    assert 2 not in t2.switches
    assert not t2.has_link(1, 2)
    assert t.has_link(1, 2)


def test_topology_egress_switches():
    t = ladder()
    assert t.egress_switches(900) == {4}
