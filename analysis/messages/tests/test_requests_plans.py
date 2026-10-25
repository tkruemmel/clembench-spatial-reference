"""Constructed-episode tests for measures/requests_plans.py (extension E1, E2)."""
import math

from analysis.messages import corpus, grammar
from analysis.messages.measures import requests_plans as rp
from analysis.messages.tests.test_alignment import FIN, make

INF = math.inf
TG = {"A": (0, 5), "B": (5, 5), "C": (5, 1)}


def P(o, fr, to, req="none", prose=None):
    m = f"PLAN: move {o} from {corpus.cell(fr)} to {corpus.cell(to)} — REQUEST: {req}"
    return f"{prose} {m}" if prose else m


def ep_(plies, start, targets=TG, objs=("B", "C"), proto="structured", **kw):
    return make([(si, mv, msg, False) for si, mv, msg in plies], proto=proto, start=start, targets=targets,
                objs=objs, **kw)


def episode(ep, bidx=None):
    return rp.episode(rp._X(ep, grammar.analyse_all([ep]), bidx))


def test_cell_classes_and_reference_error():
    ep = ep_([(0, ("A", "right"), P("A", (0, 0), (0, 5), "clear (R4,C4), keep (R3,C3) and (R3,C1) clear, "
                                                       "not (R1,C1) or (R9,C1)")),
              (1, "done", "done"),
              (0, ("A", "right"), P("A", (0, 1), (0, 5), "B please move")),
              (0, ("A", "right"), P("A", (0, 2), (0, 5), "clear (R4,C4)"))],
             start={"A": (0, 0), "B": (3, 3), "C": (5, 5)}, walls=[(2, 2)])
    rows, row = episode(ep)
    assert [(r["ply"], r["cls"]) for r in rows["e1_request_cells"]] == \
        [(0, "V"), (0, "W"), (0, "K"), (0, "S"), (0, "W"), (3, "F")]
    nocell = [r for r in rows["e2_plans"] if r["ply"] == 2][0]
    assert nocell["n_req_cells"] == 0 and nocell["names_letter"] and not nocell["names_dir"]
    assert row["ref_err"] == 4 / 6 and row["req_nocell"] == 1 / 3 and row["req_err"] == 1.0


def test_relevance():
    ep = ep_([(0, ("A", "right"), P("A", (0, 0), (0, 1), "keep (R1,C2) and (R1,C4) and (R3,C3) clear"))],
             start={"A": (0, 0), "B": (4, 4), "C": (5, 0)})
    rows, _ = episode(ep)
    got = {r["cell"]: (r["relevant"], r["relevant_own"]) for r in rows["e1_request_cells"]}
    assert got == {"(R1,C2)": (True, True), "(R1,C4)": (False, True), "(R3,C3)": (False, False)}


def test_blockers():
    walls = [(1, c) for c in range(5)]  # row 0 is A's only route to (0,5)
    ep = ep_([(1, "done", "done")], start={"A": (0, 0), "B": (0, 2), "C": (0, 3)}, walls=walls)
    S, A = "Player A", (0, 0)
    assert rp.blockers(ep, {"A": A, "B": (0, 2), "C": (5, 5)}, "A", (0, 5), S, 0) == (INF, 5, {(0, 2)})
    # in series: removing either leaves d_t infinite, but each alone raises it
    assert rp.blockers(ep, {"A": A, "B": (0, 2), "C": (0, 3)}, "A", (0, 5), S, 0)[2] == {(0, 2), (0, 3)}
    # a finished teammate's objects are not blockers
    assert rp.blockers(ep, {"A": A, "B": (0, 2), "C": (5, 5)}, "A", (0, 5), S, 1) == (INF, INF, set())
    open_ = ep_([], start={"A": A, "B": (0, 2), "C": (5, 5)})
    assert rp.blockers(open_, {"A": A, "B": (0, 2), "C": (5, 5)}, "A", (0, 5), S, 0) == (7, 5, {(0, 2)})
    assert rp.blockers(open_, {"A": A, "B": (3, 3), "C": (5, 5)}, "A", (0, 5), S, 0)[2] == set()


def corridor():
    """A walks row 0 to (0,5) behind B; the only other route is blocked by walls."""
    return ep_([(0, ("A", "right"), P("A", (0, 0), (0, 5))),
                (1, ("B", "right"), P("B", (0, 2), (5, 5))),
                (0, ("A", "right"), P("A", (0, 1), (0, 5), "clear (R1,C4)")),
                (1, ("B", "right"), P("B", (0, 3), (5, 5))),
                (0, ("A", "right"), P("A", (0, 2), (0, 5))),
                (1, ("B", "right"), P("B", (0, 4), (5, 5))),
                (0, ("A", "right"), P("A", (0, 3), (0, 5), "clear (R1,C6)")),
                (1, ("B", "down"), P("B", (0, 5), (5, 5))),
                (0, ("A", "right"), P("A", (0, 4), (0, 5)))],
               start={"A": (0, 0), "B": (0, 2), "C": (5, 0)}, walls=[(1, c) for c in range(5)])


def test_spells_recall_silent_and_useful():
    rows, row = episode(corridor())
    sp = sorted(rows["e1_spells"], key=lambda s: s["first_ply"])
    assert [(s["cell"], s["first_ply"], s["requested"], s["hard"]) for s in sp] == \
        [("(R1,C3)", 0, False, True), ("(R1,C4)", 2, True, True), ("(R1,C5)", 4, False, True),
         ("(R1,C6)", 6, True, True)]
    assert row["recall"] == 0.5 and row["silent_blocked"] == 0.5
    v = {r["ply"]: r for r in rows["e1_request_cells"]}
    assert v[2]["cls"] == v[6]["cls"] == "V" and v[2]["V_next"] and v[6]["V_next"] and v[6]["V_H"]
    # B leaves A's target at ply 7: A's d_t falls before its next turn; at ply 2 it stays infinite
    assert v[2]["useful"] is False and v[6]["useful"] is True and row["useful_V_complied"] == 0.5


def test_vacate_within_h_and_unnamed_baseline():
    ep = ep_([(0, ("A", "right"), P("A", (0, 0), (0, 5), "clear (R1,C3)")),
              (1, ("C", "down"), P("C", (0, 4), (1, 4))),
              (0, ("A", "down"), P("A", (0, 1), (1, 1))),
              (1, ("B", "down"), P("B", (0, 2), (1, 2)))],
             start={"A": (0, 0), "B": (0, 2), "C": (0, 4)})
    rows, row = episode(ep)
    r = rows["e1_request_cells"][0]
    assert (r["cls"], r["relevant"], r["V_next"], r["V_H"]) == ("V", True, False, True)
    # C on the same path, not named, moves at the holder's next turn
    assert row["V_next_onpath"] == 0.0 and row["V_base_onpath"] == 1.0 and row["V_base_onpath_n"] == 1


def keep(plies, start=None):
    rows, _ = episode(ep_(plies, start=start or {"A": (0, 0), "B": (2, 2), "C": (5, 0)}))
    return {r["ply"]: r for r in rows["e1_request_cells"]}


def test_keep_clear_compliance_and_closure():
    req = lambda fr, cl="keep (R1,C3) clear": P("A", fr, (0, 5), cl)
    entered = keep([(0, ("A", "right"), req((0, 0))), (1, ("B", "up"), P("B", (2, 2), (5, 5))),
                    (0, ("A", "down"), req((0, 1))), (1, ("B", "up"), P("B", (1, 2), (5, 5)))])
    assert (entered[0]["cls"], entered[0]["K_kept"], entered[0]["presenter_next"]) == ("K", False, "re-presents")
    passed = keep([(0, ("A", "right"), req((0, 0))), (1, ("B", "up"), P("B", (2, 2), (5, 5))),
                   (0, ("A", "right"), P("A", (0, 1), (0, 5))), (1, ("C", "right"), P("C", (5, 0), (5, 1)))])
    assert (passed[0]["K_kept"], passed[0]["K_passed"], passed[0]["presenter_next"]) == (True, True, "closes")
    plies = [(0, ("A", "down"), req((0, 0)))]
    for k in range(4):
        plies += [(1, ("C", "right" if k % 2 == 0 else "left"), P("C", (5, k % 2), (5, 1))),
                  (0, ("A", "up" if k % 2 == 0 else "down"), P("A", (1 - k % 2, 0), (0, 5)))]
    elapsed = keep(plies)
    assert (elapsed[0]["K_kept"], elapsed[0]["K_passed"], elapsed[0]["K_censored"]) == (True, False, False)
    assert keep(plies[:3])[0]["K_censored"] is True and keep(plies[:3])[0]["K_kept"] is None
    # the requester's own move enters the cell: moot; asking again after passing it errs
    moot = keep([(0, ("A", "right"), req((0, 0), "keep (R1,C2) clear")), (1, ("C", "right"), P("C", (5, 0), (5, 1))),
                 (0, ("A", "right"), req((0, 1), "keep (R1,C2) clear"))])
    assert moot[0]["K_moot"] is True and moot[0]["K_kept"] is None and moot[0]["presenter_next"] == "errs"


def test_negative_control_rolled_off():
    plies = [(0, ("A", "up"), P("A", (1, 0), (1, 3), "clear (R3,C3)"))]
    for k in range(4):
        plies += [(1, ("C", "left" if k % 2 == 0 else "right"), P("C", (5, 5 - k % 2), (5, 1))),
                  (0, ("A", "right" if k % 2 == 0 else "left"), P("A", (0, k % 2), (0, 1 - k % 2)))]
    plies.append((1, ("B", "up"), P("B", (2, 2), (5, 5))))
    ep = ep_(plies, start={"A": (1, 0), "B": (2, 2), "C": (5, 5)}, arm="final", seats=FIN)
    rows, row = episode(ep)
    assert [(d["kind"], d["exposure"], d["decision_ply"], d["outcome"]) for d in rows["e1_decisions"]] == \
        [("V", "visible", 1, False), ("V", "nc_rolled_off", 9, True)]
    up = [(u["exposure"], u["decision_ply"], u["obj"], u["dir"], u["entered"]) for u in rows["e2_uptake"]
          if u["msg_ply"] == 0]
    assert up == [("visible", 1, "B", "up", False), ("nc_rolled_off", 9, "B", "up", True)]
    assert row["V_dec_visible"] == 0.0 and row["V_dec_nc_rolled_off"] == 1.0 and row["up_enter_nc"] == 1.0


def test_say_do_and_plan_to_validity():
    st = {"A": (2, 2), "B": (0, 4), "C": (1, 5), "D": (4, 4)}
    tg = {"A": (2, 5), "B": (0, 0), "C": (5, 5), "D": (5, 5)}
    ep = ep_([(0, ("A", "right"), P("A", (2, 2), (2, 5))),
              (1, ("D", "up"), P("B", (0, 4), (1, 5))),
              (0, ("A", "down"), P("A", (2, 3), (4, 0))),
              (1, ("D", "down"), P("D", (3, 4), (3, 5))),
              (0, ("A", "up"), P("A", (3, 3), (0, 4))),
              (1, ("D", "up"), P("D", (4, 4), (5, 0))),
              (0, ("A", "left"), P("A", (2, 3), (0, 5)))],
             start=st, targets=tg, objs=("B", "C", "D"), walls=[(4, 0), (5, 1)])
    rows, _ = episode(ep)
    got = [(r["saydo"], r["to_class"], r["goal"], r["waypoint"]) for r in rows["e2_plans"]]
    assert got == [("progress", "valid", True, None), ("other_object", "own", False, "other"),
                   ("same_no_progress", "wall", False, None), ("same_no_progress", "valid", False, "progress"),
                   ("progress", "x", False, "other"), ("same_no_progress", "static", False, "other"),
                   ("same_no_progress", "dynamic", False, "other")]


def test_give_way_waypoint():
    st, tg = {"A": (0, 2), "B": (0, 0), "C": (5, 0)}, {"A": (0, 5), "B": (0, 4), "C": (5, 1)}
    rows, _ = episode(ep_([(0, ("A", "down"), P("A", (0, 2), (1, 2))),
                           (1, ("C", "right"), P("C", (5, 0), (5, 1))),
                           (0, ("A", "up"), P("A", (1, 2), (0, 1))),
                           (1, ("C", "left"), P("C", (5, 1), (5, 0))),
                           (0, ("A", "left"), P("A", (0, 2), (0, 1)))], start=st, targets=tg))
    # the last PLAN-to is on B's path too, so leaving (R1,C3) frees nothing
    assert [(r["waypoint"], r["frees"]) for r in rows["e2_plans"] if r["seat"] == "Player A"] == \
        [("give_way", True), ("other", False), ("other", False)]


def test_persistence():
    seat0 = [(("A", "right"), (0, 0), (0, 2)), (("A", "right"), (0, 1), (0, 2)), (("A", "down"), (0, 2), (3, 4)),
             (("A", "left"), (1, 2), (3, 4)), (("A", "down"), (1, 1), (0, 5)), (("A", "right"), (2, 1), (5, 4)),
             (("A", "up"), (2, 2), (3, 4)), (("A", "down"), (1, 2), (3, 4)), (("A", "left"), (2, 2), (0, 5))]
    seat1 = [("C", "right"), ("C", "left"), ("B", "right"), ("C", "right"), ("C", "left"), ("C", "right"),
             ("C", "left"), ("C", "right")]
    plies, c = [], (5, 0)
    for k, (mv, fr, to) in enumerate(seat0):
        plies.append((0, mv, P("A", fr, to)))
        if k < len(seat1):
            o, d = seat1[k]
            plies.append((1, (o, d), P(o, (3, 3) if o == "B" else c, (5, 5))))
            if o == "C":
                c = corpus.step({"C": c}, "C", d)["C"]
    rows, row = episode(ep_(plies, start={"A": (0, 0), "B": (3, 3), "C": (5, 0)}))
    assert [r["end"] for r in sorted(rows["e2_runs"], key=lambda r: r["first_ply"]) if r["seat"] == "Player A"] == \
        ["reached", "forced", "unforced", "unforced", "stale", "open"]


def test_conflicts():
    ep = ep_([(0, ("A", "right"), P("A", (0, 0), (2, 2))),
              (1, ("C", "right"), P("B", (3, 3), (2, 2))),
              (0, ("A", "right"), P("A", (0, 1), (0, 5))),
              (1, ("C", "left"), P("C", (5, 1), (5, 0))),
              (0, ("A", "down"), P("A", (0, 2), (3, 3))),
              (1, ("B", "down"), P("B", (3, 3), (4, 3))),
              (0, ("A", "up"), P("A", (1, 2), (0, 5)))],
             start={"A": (0, 0), "B": (3, 3), "C": (5, 0)})
    rows, row = episode(ep)
    assert [(s["kinds"], s["start_ply"], s["end"], s["latency"]) for s in rows["e2_conflicts"]] == \
        [("vertex", 2, "resolved", 1), ("hold", 5, "resolved", 1)]
    assert row["conflict_n"] == 5 and row["conflict"] == 0.4
    # the other PLAN moves the held object: no conflict
    ep["turns"][3]["slots"]["plan_obj"] = "B"
    ep["turns"][3]["slots"]["plan_to"] = (4, 3)
    rows, _ = episode(ep)
    assert [s["kinds"] for s in rows["e2_conflicts"]] == ["vertex"]


def test_uptake_kinds():
    def up(b_start, move):
        ep = ep_([(0, ("A", "right"), P("A", (0, 0), (0, 3))), (1, move, P("B", b_start, (5, 5)))],
                 start={"A": (0, 0), "B": b_start, "C": (5, 0)})
        return [(u["obj"], u["dir"], u["kind"], u["entered"]) for u in episode(ep)[0]["e2_uptake"]]
    assert up((1, 2), ("B", "left")) == [("B", "up", "path", False)]
    assert up((1, 2), ("B", "up")) == [("B", "up", "path", True)]
    assert up((1, 3), ("B", "down")) == [("B", "up", "to", False)]


def test_run_tables_baseline_and_notes():
    start = {"A": (0, 0), "B": (3, 3), "C": (5, 0)}
    st = ep_([(0, ("A", "right"), P("A", (0, 0), (0, 5), "clear (R4,C4)")),
              (1, ("B", "up"), P("B", (3, 3), (5, 5)))], start=start)
    hy = ep_([(0, ("A", "right"), P("A", (0, 0), (0, 5), prose="Keep (R3,C3) for me.")),
              (1, ("B", "up"), P("B", (3, 3), (5, 5), prose="ok"))], start=start, proto="hybrid")
    none = ep_([(0, ("A", "right"), None), (1, ("B", "up"), None)], start=start, proto="none")
    eps = [st, hy, none]
    out = rp.run("dev", eps, grammar.analyse_all(eps))
    v = out["e1_request_cells"][0]
    assert (v["cls"], v["V_next"], v["covered"], v["base_rate"]) == ("V", True, True, 1.0)
    assert [r["proto"] for r in out["rp_episodes"]] == ["structured", "hybrid"]
    assert [r["n_shadow"] for r in out["e2_plans"] if r["proto"] == "hybrid" and r["ply"] == 0] == [1]
    assert {"rp_cells", "rp_baselines", "rp_proto", "rp_summary", "rp_coverage"} <= set(out)
    assert any("prompts differ on REQUEST" in n["note"] for n in out["rp_notes"])
