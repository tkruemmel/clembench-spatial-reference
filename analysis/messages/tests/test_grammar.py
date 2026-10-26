"""Constructed suites for the G-* grammars (spec §4, §9.1).

Positive and negative strings, the prompt's own example sentences and the
known failure strings. A failing case changes a rule's logic, never a list.
"""
from analysis.messages.grammar import DEFAULT, analyse, audits, clauses, kept, lists, normalise, target_mentions, tokenize

PROMPT_EXAMPLES = [
    "I'm moving A up toward its goal at (R2,C3). Could you move your object out of (R2,C3)? "
    "I'll need that cell free in a few turns.",
    "I'm taking A down column C3 to its goal at (R4,C3), passing through (R3,C3) on the way.",
]


def loc(s, arm="dev", mode="strict", **kw):
    P = dict(DEFAULT, **kw)
    return [(p["letter"], p["cell"]) for p in analyse(s, arm, "freeform", P=P)["loc"] if kept(p["frames"], mode)]


def frames(s, **kw):
    P = dict(DEFAULT, **kw)
    return [sorted(p["frames"]) for p in analyse(s, "dev", "freeform", P=P)["loc"]]


def kinds(s, arm="dev"):
    return [(t.kind, t.val) for t in tokenize(s, arm, DEFAULT)]


def test_lists_from_sources():
    W = lists()
    assert len(W["say"]) == 52 and len(W["exist"]) == 28 and len(W["keep"]) == 7 and len(W["put"]) == 28
    assert {"said", "says"}.isdisjoint(W["say"]) and "say" in W["say"]
    assert W["goal_noun_audit"] == {"aim", "destination", "finish", "objective"}


def test_g_cell_forms():
    for s in ["(R3,C4)", "R3C4", "R3,C4", "(R3, C4)", "r3c4", "( r3 , c4 )", "R3 C4"]:
        assert kinds(s) == [("cell", (2, 3))], s
    assert [k for k, _ in kinds("(3,4) and row 3 column 4")] == ["punct", "num", "punct", "num", "punct", "word",
                                                                 "line", "line"]
    assert ("cell", (0, 1)) in kinds("for 3 rounds, (R1,C2)")
    assert not any(k == "cell" for k, _ in kinds("forR3C4"))


def test_g_line_forms():
    for s, v in [("row 4", ("R", 3)), ("column 5", ("C", 4)), ("column C5", ("C", 4)), ("columns C5", ("C", 4)),
                 ("C5", ("C", 4)), ("R4", ("R", 3))]:
        assert kinds(s) == [("line", v)], s
    assert kinds("column C8 is clear")[0] == ("line", ("C", 7))


def test_g_seat_precedence():
    assert kinds("Player B moved")[0] == ("seat", "Player B")
    assert ("letter", "B") not in kinds("Player B moved")
    assert kinds("Player Orange's E", "final")[:1] == [("seat", "Player Orange")]
    assert kinds("Blue has E", "final")[0] == ("seat", "Player Blue")
    assert ("letter", "E") in kinds("Blue has E", "final")
    # colours are not seats on the development arm
    assert kinds("Blue has E", "dev")[0] == ("word", "blue")


def test_g_letter_article_rule():
    assert ("letter", "A") in kinds("A is at (R1,C1)")
    assert ("letter", "A") in kinds("A moves up")
    assert ("letter", "A") in kinds("A up")
    assert ("letter", "A") in kinds("A, then B")
    assert ("letter", "A") in kinds("A")
    assert ("letter", "A") not in kinds("A clear path exists")
    assert ("letter", "A") not in kinds("A good plan.")
    assert ("letter", "A") in kinds("then A clear path")  # not sentence-initial
    assert ("letter", "A") not in kinds("Done. A free cell")
    P = dict(DEFAULT, article=False)
    assert ("letter", "A") in [(t.kind, t.val) for t in tokenize("A clear path", "dev", P)]
    assert ("letter", "a") not in kinds("a path") and ("word", "a") in kinds("a path")


def test_g_dir():
    assert [v for k, v in kinds("up down left right upwards leftward") if k == "dir"] == \
        ["up", "down", "left", "right", "up", "left"]
    assert not any(k == "dir" for k, _ in kinds("north"))
    assert [(t.kind, t.val) for t in tokenize("north", "dev", dict(DEFAULT, compass=True))] == [("dir", "up")]


def test_clauses():
    toks = tokenize("A is at (R1,C1), and B is at (R2,C2); C up — D down\nE left", "dev", DEFAULT)
    assert len(clauses(toks, DEFAULT)) == 5
    toks = tokenize("A is at (R1,C1), B is at (R2,C2)", "dev", DEFAULT)
    assert len(clauses(toks, DEFAULT)) == 1
    assert len(clauses(toks, dict(DEFAULT, comma=True))) == 2
    toks = tokenize("Keep (R3,C4) clear, if you can", "dev", DEFAULT)
    assert len(clauses(toks, DEFAULT)) == 2  # comma + subordinator; the cell's comma never splits
    toks = tokenize("goal-ward A", "dev", DEFAULT)
    assert len(clauses(toks, DEFAULT)) == 1  # an in-word hyphen is no dash


def test_g_loc_positive():
    assert loc("E is at (R2,C6)") == [("E", (1, 5))]
    assert loc("E at (R2,C6)") == [("E", (1, 5))]
    assert loc("B waits at (R2,C3).") == [("B", (1, 2))]  # agreeing form is indicative
    assert loc("A stays at (R1,C1)") == [("A", (0, 0))]
    assert loc("I note A is at (R1,C1)") == [("A", (0, 0))]  # base form of a reporting verb
    assert loc("Since C is at (R5,C5) I go up") == [("C", (4, 4))]  # factive subordinator
    assert loc("(R4,C5) is occupied by F") == [("F", (3, 4))]
    assert loc("(R4,C5) has F") == [("F", (3, 4))]
    assert loc("(R4,C5) holds F") == [("F", (3, 4))]
    assert loc("A's at r3, c4 now") == [("A", (2, 3))]
    assert loc("D, currently sitting right at (R1,C2)") == [("D", (0, 1))]  # 3 tokens between


def test_g_loc_gap_and_blocking():
    s = "D is one two three four at (R1,C2)"
    assert loc(s) == [] and loc(s, gap=8) == [("D", (0, 1))]
    assert loc("D is right at (R1,C2)", gap=2) == [("D", (0, 1))]
    assert loc("A and B at (R1,C1)") == [("B", (0, 0))]  # sentence-initial A + "and" is the article
    assert loc("then A and B at (R1,C1)") == [("B", (0, 0))]  # another letter in between blocks A
    assert loc("E near (R2,C2) at (R3,C3)") == []  # another cell in between


def test_g_loc_exclusions():
    for s, fr in [
        ("I'm moving A up toward its goal at (R2,C3)", "destination"),
        ("B's goal is at (R4,C4)", "destination"),
        ("A will be at (R2,C3)", "destination"),
        ("A is going to be at (R2,C3)", "destination"),
        ("If D is at (R4,C4), I'll go left.", "conditional"),
        ("once C is at (R2,C2) I go", "conditional"),
        ("Orange said D is at (R3,C3)", "reported"),
        ("according to you, D is at (R3,C3)", None),  # the comma bounds the look-back
        ("B wait at (R2,C3) until A passes", "directive"),
        ("B stay at (R1,C1)", "directive"),  # "A stay" would be the article
        ("keep A at (R1,C1)", "directive"),
        ("park B at (R2,C2)", "directive"),
        ("C is not at (R1,C1)", "negation"),
        ("C isn't at (R1,C1)", "negation"),
        ("Is C at (R2,C2)?", "question"),
        ("<think>A is at (R1,C1)</think>", "think"),
    ]:
        got = frames(s)
        assert got, s
        if fr is None:
            assert got == [[]], (s, got)
        else:
            assert fr in got[0], (s, got)
            assert loc(s) == [], s
    assert loc("according to Blue D is at (R3,C3)") == []  # reported, inside the look-back
    assert frames("No, A is at (R1,C1)") == [[]]  # look-back stops at the comma
    assert frames("No, A is at (R1,C1)", lookback=0) == [[]]
    assert frames("never A at (R1,C1)") == [["negation"]]
    assert frames("never A at (R1,C1)", lookback=0) == [[]]


def test_g_loc_lenient():
    assert loc("C is not at (R1,C1)", mode="lenient") == [("C", (0, 0))]
    assert loc("B wait at (R2,C3)", mode="lenient") == []
    assert loc("A will be at (R2,C3)", mode="lenient") == []


def test_known_failure_strings():
    for s in ["unblocked", "I'll wait", "column C8 is clear", "A goes to (R2,C3)", "move B to (R3,C3)"]:
        assert loc(s) == [], s
    assert loc(PROMPT_EXAMPLES[0]) == []
    assert loc(PROMPT_EXAMPLES[1]) == []


def test_g_from():
    def frm(s):
        return [(p["letter"], p["cell"], sorted(p["frames"])) for p in analyse(s, "dev", "freeform")["frm"]]
    assert frm("D from (R3,C4) to (R4,C4)") == [("D", (2, 3), [])]
    assert frm("C moves from (R2,C3) to (R2,C4)") == [("C", (1, 2), [])]
    assert frm("I moved B from (R3,C4)") == [("B", (2, 3), ["past"])]
    assert frm("after C moves from (R2,C3)") == [("C", (1, 2), ["future"])]
    assert frm("A will go from (R1,C1)") == [("A", (0, 0), ["future"])]


def test_g_move_frames():
    def mv(s, **kw):
        return [(m["letter"], m["dir"], m["frame"]) for m in analyse(s, "dev", "freeform", P=dict(DEFAULT, **kw))["moves"]]
    assert mv("I'm moving A up toward its goal at (R2,C3)") == [("A", "up", "present")]
    assert mv("Moving B down to (R3,C3)") == [("B", "down", "present")]
    assert mv("I moved A up") == [("A", "up", "past")]
    assert mv("Next I'll move D left") == [("D", "left", "future")]
    assert mv("A up, then I will move B left") == [("A", "up", "present"), ("B", "left", "future")]
    assert mv("after E passes, D down") == [("D", "down", "sequence")]
    assert mv("If C clears, A right") == [("A", "right", "conditional")]
    assert mv("I won't move B down") == [("B", "down", "negated")]
    assert mv("Should I move B down?") == [("B", "down", "question")]
    assert mv("A. Then up") == []  # different clauses
    assert mv("A. Then up", move_scope="message") == [("A", "up", "present")]


def test_slots_regions():
    s = "PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: please clear (R3,C3) if you can"
    slots = dict(prose=None, plan="move A from (R2,C3) to (R4,C3)", request="please clear (R3,C3) if you can")
    an = analyse(s, "dev", "structured", slots=slots)
    assert {r for r, *_ in an["cells"]} == {"plan", "request"}
    assert [(p["region"], p["letter"]) for p in an["frm"]] == [("plan", "A")]


def test_g_target():
    an = analyse("My goal is (R4,C3). A's target is (R4,C3). B goes to (R1,C1)", "dev", "freeform")
    t = target_mentions(an, {"A": (3, 2), "B": (0, 0), "D": (5, 5)})
    assert t["A"] == dict(stated=True, named=True)
    assert t["B"] == dict(stated=False, named=True)
    assert t["D"] == dict(stated=False, named=False)


def test_audits():
    s = "A is near (3,4). B sits beside (R2,C2), heading east toward its destination; X at (R1,C1)"
    an = analyse(s, "dev", "freeform")
    a = audits(s, an)
    assert a["cell"] == 1 and a["direction"] == 1 and a["x_at"] == 1
    assert a["letter_cell_fwd"] == 1  # B ... (R2,C2), not a G-loc pair
    assert a["goal_noun"] == 1
    an = analyse("(R2,C2) next to C", "dev", "freeform")
    assert audits("(R2,C2) next to C", an)["letter_cell_rev"] == 1


def test_normalise():
    an = analyse("A is at (R1,C1), X moved; Player B wait 3 turns in row 2", "dev", "freeform")
    assert normalise(an, own={"A"}) == ["<own>", "is", "at", "<cell>", "<x>", "moved", "<seat>", "wait", "<num>",
                                        "turns", "in", "<line>"]


def test_directive_base_form_after_auxiliary_or_possessive():
    # review 2026-10-17: participles and nouns identical to the base form are not imperatives
    assert loc("B is set at (R2,C5)") == [("B", (1, 4))]
    assert loc("B's position at (R2,C5) blocks me") == [("B", (1, 4))]
    assert loc("B wait at (R2,C5)") == []
    assert loc("B, stay at (R2,C5)") == []


def test_cannot_is_can_not():
    assert [(t.kind, t.val) for t in tokenize("cannot", "dev", DEFAULT)] == [("word", "can"), ("word", "not")]
    assert frames("C cannot be at (R1,C1)") and "negation" in frames("C cannot be at (R1,C1)")[0]


def test_reverse_rule_skips_possessive():
    assert loc("(R3,C4) has B's goal") == []
    assert loc("(R3,C4) has B") == [("B", (2, 3))]
