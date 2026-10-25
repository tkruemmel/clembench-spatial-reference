"""Grammars G-* (spec §4) and the word lists of §4.1, built from their sources.

    python3 -m analysis.messages.grammar --build-lists   # data/wordlists.json

Every list is taken whole from a source that does not depend on the logs:
the reference grammar (Huddleston & Pullum 2002, CGEL), VerbNet 3.4 class
files copied into data/verbnet3.4/ at commit VN_COMMIT, WordNet 3.0 read by
NLTK, and the game's prompt and code. No word is added or removed after
messages have been read; what a list misses is reported by the audits.

`analyse(text, arm, proto, P)` tokenizes one message, splits it into
clauses per region (prose / plan / request) and returns every grammar match.
P holds the specification-curve parameters (§9.3); DEFAULT is the primary.
"""
import functools
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LISTS = os.path.join(DATA, "wordlists.json")
VN_COMMIT = "ae8e9cfdc2c0d3414b748763612f1a0a34194cc1"
VN_CLASSES = {"say": "say-37.7", "exist": "exist-47.1", "keep": "keep-15.2", "put": "put-9.1"}

# Huddleston & Pullum 2002 lists as given in spec §4.1 (chapters there).
# Multiword items are space-joined and matched as token sequences.
CGEL = dict(
    modals=["can", "could", "may", "might", "must", "shall", "should", "will", "would", "'ll", "'d",
            "need", "ought", "dare", "going to", "have to", "has to", "had to"],
    aux=["be", "am", "is", "are", "was", "were", "been", "being", "'s", "'re", "'m",
         "have", "has", "had", "having", "'ve", "do", "does", "did", "done", "doing"],
    present_aux=["am", "is", "are", "has"],  # the present forms among WordNet's irregular verb forms
    goal_preps=["to", "toward", "towards", "into", "onto"],
    conditional=["if", "unless", "provided", "providing", "assuming", "supposing", "as long as",
                 "so long as", "in case"],
    temporal_nonfactive=["once", "until", "till", "when", "whenever", "before", "after", "as soon as"],
    temporal_factive=["since", "while", "now that"],
    causal=["because", "since"],
    coordinators=["and", "or", "but", "nor"],
    negators=["not", "n't", "no", "none", "nobody", "nothing", "nowhere", "neither", "nor", "never"],
)
PROMPT = dict(
    goal_nouns=["goal", "goals", "target", "targets"],  # prompt_final/ templates
    dirs=["up", "down", "left", "right", "upward", "upwards", "downward", "downwards",
          "leftward", "leftwards", "rightward", "rightwards"],
)
# curve only: the cardinal points as an alternative direction vocabulary
COMPASS = ["north", "south", "east", "west", "northward", "northwards", "southward", "southwards",
           "eastward", "eastwards", "westward", "westwards"]
THINK = ["<think>", "</think>"]
FINAL_COLOURS = ("Green", "Orange", "Blue", "Yellow", "Red", "Purple")
DIR_CANON = {d: d.split("ward")[0] for d in PROMPT["dirs"]}
DIR_CANON.update({"north": "up", "south": "down", "east": "right", "west": "left"})
DIR_CANON.update({w: DIR_CANON[w.split("ward")[0]] for w in COMPASS if "ward" in w})


# ---------------------------------------------------------------- word lists

def _vn_members(cls):
    """Single-word members of a VerbNet class and all its subclasses."""
    root = ET.parse(os.path.join(DATA, "verbnet3.4", f"{cls}.xml")).getroot()
    names = {m.get("name") for el in root.iter() if el.tag in ("VNCLASS", "VNSUBCLASS")
             for m in el.find("MEMBERS")}
    return sorted(n for n in names if "_" not in n), sorted(n for n in names if "_" in n)


def build_lists():
    import nltk
    from nltk.corpus import wordnet as wn
    out = dict(CGEL)
    out.update(PROMPT)
    out["compass"], out["think"] = COMPASS, THINK
    dropped = {}
    for k, cls in VN_CLASSES.items():
        out[k], dropped[k] = _vn_members(cls)
    # §9.2 goal-noun audit: WordNet synonyms in the place and aim senses, minus
    # G-target's own nouns and minus "object", the game's noun for its pieces
    audit = set()
    for s in ("finish.n.04", "aim.n.02"):
        audit |= {l.lower() for l in wn.synset(s).lemma_names()}
    audit -= {"goal", "target", "object"}
    out["goal_noun_audit"] = sorted(audit)  # matched by WordNet noun lemma, so plurals count
    out["sources"] = dict(
        cgel="Huddleston & Pullum 2002, The Cambridge Grammar of the English Language",
        verbnet=f"VerbNet 3.4, github.com/cu-clear/verbnet commit {VN_COMMIT}",
        verbnet_classes=VN_CLASSES, verbnet_dropped_multiword=dropped,
        wordnet=f"WordNet {wn.get_version()} via NLTK {nltk.__version__}",
        nltk=nltk.__version__, wordnet_version=wn.get_version(),
        prompt="prompt_final/{dyad,nplayer}/*.template")
    with open(LISTS, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


@functools.lru_cache(None)
def lists():
    W = json.load(open(LISTS))
    sets = {k: set(v) for k, v in W.items() if isinstance(v, list)}
    sets["sources"] = W["sources"]
    return sets


@functools.lru_cache(None)
def _wn():
    import nltk
    from nltk.corpus import wordnet as wn
    src = lists()["sources"]
    assert (nltk.__version__, wn.get_version()) == (src["nltk"], src["wordnet_version"]), \
        "WordNet/NLTK differ from the versions the lists were built with"
    wn.ensure_loaded()
    return wn


@functools.lru_cache(None)
def verb_lemma(w):
    """WordNet 3.0 verb lemma of a lower-case token (morphy), or None."""
    return _wn().morphy(w, "v")


@functools.lru_cache(None)
def noun_lemma(w):
    return _wn().morphy(w, "n")


def inflected(w):
    """§4.1: a token is an inflected verb form if its verb lemma differs from it."""
    lem = verb_lemma(w)
    return lem is not None and lem != w


@functools.lru_cache(None)
def past_form(w):
    """An -ed inflection or one of WordNet's irregular verb forms (verb.exc),
    other than the -ing forms and the present forms of be and have."""
    if not inflected(w) or w.endswith("ing") or w in lists()["present_aux"]:
        return False
    return w.endswith("ed") or w in _wn()._exception_map["v"]


# ---------------------------------------------------------------- tokens

Tok = namedtuple("Tok", "kind val text i0 i1")
_SEAT = {
    "dev": r"(?i:player)\s+[A-F](?![A-Za-z0-9])",
    "final": r"(?i:player)\s+(?i:green|orange|blue|yellow|red|purple)\b|(?:" + "|".join(FINAL_COLOURS) + r")\b",
}
_TOKEN = r"""
 (?P<think></?think>)
|(?P<cell>(?<![A-Za-z0-9])\(?\s*[Rr]\s*(?P<cr>\d+)\s*,?\s*[Cc]\s*(?P<cc>\d+)(?!\d)(?:\s*\))?)
|(?P<seat>{seat})
|(?P<line>(?i:\b(?P<lw>row|column)s?\s+[RrCc]?(?P<ln>\d+)\b)|(?<![A-Za-z0-9])(?P<lk>[RrCc])(?P<lx>\d+)(?![A-Za-z0-9]))
|(?P<contr>n['’]t\b|['’](?:ll|d|s|re|ve|m)\b)
|(?P<word>[A-Za-z]+(?=n['’]t\b)|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)
|(?P<nl>\n)
|(?P<punct>[^\sA-Za-z0-9])
"""
_TOKRE = {arm: re.compile(_TOKEN.format(seat=s), re.X) for arm, s in _SEAT.items()}
_NEG_STEM = {"ca": "can", "wo": "will", "sha": "shall"}  # can't, won't, shan't
BOUNDARY = set(".;!?")
DASHES = {"—", "–"}


def tokenize(text, arm, P):
    W = lists()
    dirs = W["dirs"] | (W["compass"] if P["compass"] else set())
    toks = []
    for m in _TOKRE[arm].finditer(text):
        g = m.group(0)
        if m.group("cell"):
            toks.append(Tok("cell", (int(m.group("cr")) - 1, int(m.group("cc")) - 1), g, m.start(), m.end()))
        elif m.group("line"):
            if m.group("lw"):
                kind = "R" if m.group("lw").lower() == "row" else "C"
                toks.append(Tok("line", (kind, int(m.group("ln")) - 1), g, m.start(), m.end()))
            else:
                toks.append(Tok("line", (m.group("lk").upper(), int(m.group("lx")) - 1), g, m.start(), m.end()))
        elif m.group("seat"):
            name = g.split()[-1].capitalize() if arm == "final" else g.split()[-1]
            toks.append(Tok("seat", f"Player {name}", g, m.start(), m.end()))
        elif m.group("think"):
            toks.append(Tok("think", g, g, m.start(), m.end()))
        elif m.group("contr"):
            toks.append(Tok("word", g.replace("’", "'").lower(), g, m.start(), m.end()))
        elif m.group("word") and g.lower() == "cannot":  # can + not, as "can't" is ca + n't
            toks.append(Tok("word", "can", g[:3], m.start(), m.start() + 3))
            toks.append(Tok("word", "not", g[3:], m.start() + 3, m.end()))
        elif m.group("word"):
            low = g.lower()
            if text[m.end():m.end() + 3].replace("’", "'").lower() == "n't":
                low = _NEG_STEM.get(low, low)
            if len(g) == 1 and g in "ABCDEF":
                toks.append(Tok("letter", g, g, m.start(), m.end()))
            elif low in dirs:
                toks.append(Tok("dir", DIR_CANON[low], g, m.start(), m.end()))
            elif g.isdigit():
                toks.append(Tok("num", int(g), g, m.start(), m.end()))
            else:
                toks.append(Tok("word", low, g, m.start(), m.end()))
        elif m.group("nl"):
            toks.append(Tok("nl", "\n", g, m.start(), m.end()))
        else:
            toks.append(Tok("punct", g, g, m.start(), m.end()))
    if P["article"]:
        toks = _article_rule(toks)
    return toks


def _sentence_initial(toks, i):
    j = i - 1
    return j < 0 or toks[j].kind == "nl" or (toks[j].kind == "punct" and toks[j].val in BOUNDARY)


def _article_rule(toks):
    """A sentence-initial "A" is the letter only if the next token is an
    inflected verb form, an auxiliary or modal, at/to/from, a direction, a
    cell or punctuation (end of text counts as punctuation)."""
    W = lists()
    out = list(toks)
    for i, t in enumerate(toks):
        if t.kind != "letter" or t.val != "A" or not _sentence_initial(toks, i):
            continue
        nx = toks[i + 1] if i + 1 < len(toks) else None
        if nx is None or nx.kind in ("cell", "dir", "punct", "nl"):
            continue
        if nx.kind == "word" and (nx.val in W["aux"] or nx.val in W["modals"] or nx.val in ("at", "to", "from")
                                  or inflected(nx.val)):
            continue
        out[i] = Tok("word", "a", t.text, t.i0, t.i1)
    return out


# ---------------------------------------------------------------- clauses

def clauses(toks, P):
    """[(i0, i1, terminator)] token spans. Boundaries: . ; ! ? newline, a dash,
    and a comma followed by a coordinator or subordinator (any comma if
    P['comma']). A boundary token belongs to no clause."""
    W = lists()
    starters = W["coordinators"] | W["conditional"] | W["temporal_nonfactive"] | W["temporal_factive"] | W["causal"]
    first = {s.split()[0] for s in starters}
    out, start = [], 0

    def close(end, term):
        if end > start:
            out.append((start, end, term))

    i = 0
    while i < len(toks):
        t = toks[i]
        cut = None
        if t.kind == "nl" or (t.kind == "punct" and (t.val in BOUNDARY or t.val in DASHES)):
            cut = t.val
        elif t.kind == "punct" and t.val == "-":  # in-word hyphens sit inside word tokens
            cut = "-"
        elif t.kind == "punct" and t.val == ",":
            nx = toks[i + 1] if i + 1 < len(toks) else None
            if P["comma"] or (nx is not None and nx.kind == "word" and nx.val in first
                              and _starts_listed(toks, i + 1, starters)):
                cut = ","
        if cut is not None:
            close(i, cut)
            start = i + 1
        i += 1
    close(len(toks), None)
    return out


def _starts_listed(toks, i, items):
    return any(_seq_at(toks, i, s.split()) for s in items)


def _seq_at(toks, i, words):
    return all(i + k < len(toks) and toks[i + k].kind == "word" and toks[i + k].val == w for k, w in enumerate(words))


def _has(toks, idxs, items):
    """Does any listed word or word sequence occur at the token positions idxs?"""
    idxs = sorted(idxs)
    s = set(idxs)
    for i in idxs:
        t = toks[i]
        if t.kind != "word":
            continue
        for it in items:
            w = it.split()
            if t.val == w[0] and all(i + k in s for k in range(len(w))) and _seq_at(toks, i, w):
                return True
    return False


# ---------------------------------------------------------------- frames

FRAMES = ("destination", "conditional", "reported", "directive", "negation", "question", "think")
LENIENT = ("destination", "directive")


def _frames(toks, span, look, before, clause_term, think):
    """Exclusion frames hit by a candidate pair (spec §4, G-loc).
    span: token positions between L and the relation word; look: look-back
    positions; before: the position immediately before L (or None)."""
    W = lists()
    both = list(span) + list(look)
    hit = set()
    if _has(toks, both, W["goal_preps"] | W["modals"] | W["goal_nouns"]):
        hit.add("destination")
    if _has(toks, both, W["conditional"] | W["temporal_nonfactive"]):
        hit.add("conditional")
    if any(toks[i].kind == "word" and inflected(toks[i].val) and verb_lemma(toks[i].val) in W["say"] for i in both) \
            or _has(toks, both, {"according to"}):
        hit.add("reported")
    if any(toks[i].kind == "word" and (toks[i].val in W["exist"] or toks[i].val in W["put"]) and _imperative(toks, i)
           for i in span) \
            or (before is not None and toks[before].kind == "word"
                and (toks[before].val in W["keep"] or toks[before].val in W["put"])):
        hit.add("directive")
    if _has(toks, both, W["negators"]):
        hit.add("negation")
    if clause_term == "?":
        hit.add("question")
    if think:
        hit.add("think")
    return hit


def _imperative(toks, i):
    """A base form between L and `at` marks the imperative ("B wait at") unless
    the token before it is an auxiliary, making it a participle ("B is set at"),
    or a possessive 's, making it a noun ("B's position at")."""
    prev = toks[i - 1] if i else None
    return not (prev is not None and prev.kind == "word" and prev.val in lists()["aux"] | {"'s"})


def _lookback(toks, i, c0, n):
    """Up to n positions before i inside the clause, not crossing a comma."""
    out = []
    j = i - 1
    while j >= c0 and len(out) < n:
        if toks[j].kind == "punct" and toks[j].val == ",":
            break
        out.append(j)
        j -= 1
    return out


def kept(frames, mode):
    """Is a candidate kept under the strict (every exclusion) or lenient
    (destination and directive only) frame set?"""
    return not (frames & set(FRAMES if mode == "strict" else LENIENT))


# ---------------------------------------------------------------- analysis

DEFAULT = dict(comma=False, gap=4, lookback=3, frames="strict", article=True, compass=False, move_scope="clause")


def regions(text, proto, slots):
    """[(region, text)]: freeform is prose; structured/hybrid split by the master's schema."""
    if not slots:
        return [("prose", text)]
    out = [("prose", slots["prose"])] if slots.get("prose") else []
    return out + [("plan", slots["plan"]), ("request", slots["request"])]


def analyse(text, arm, proto, slots=None, P=DEFAULT):
    """Every grammar match in one message. Clause ids are global over regions."""
    toks_all, cl_all, out = [], [], dict(think=False, loc=[], frm=[], moves=[], cells=[], lines=[],
                                          seats=[], letters=[], dirs=[], clause_info=[])
    out["think"] = any(t in (text or "") for t in THINK)
    for region, seg in regions(text, proto, slots):
        toks = tokenize(seg, arm, P)
        base = len(cl_all)
        for ci, (c0, c1, term) in enumerate(clauses(toks, P)):
            cid = base + ci
            cl_all.append((region, toks, c0, c1, term))
            _clause_matches(out, toks, cid, region, c0, c1, term, P)
        toks_all.append((region, toks))
    if P["move_scope"] == "message":
        out["moves"] = _message_moves(cl_all)
    out["tokens"] = toks_all
    out["n_clauses"] = len(cl_all)
    return out


def _clause_matches(out, toks, cid, region, c0, c1, term, P):
    W = lists()
    rng = range(c0, c1)
    words = {toks[i].val for i in rng if toks[i].kind == "word"}
    letters = [i for i in rng if toks[i].kind == "letter"]
    cellpos = [i for i in rng if toks[i].kind == "cell"]
    out["clause_info"].append(dict(id=cid, region=region, term=term,
                                   letters=[toks[i].val for i in letters], cells=[toks[i].val for i in cellpos],
                                   goal_noun=bool(words & W["goal_nouns"]), my_its=bool(words & {"my", "its"}),
                                   goal_noun_audit=any(noun_lemma(w) in W["goal_noun_audit"] for w in words),
                                   text=" ".join(toks[i].text for i in rng)))
    for kind, key in (("cell", "cells"), ("line", "lines"), ("seat", "seats"), ("letter", "letters"), ("dir", "dirs")):
        out[key] += [(region, cid, toks[i].val, i) for i in rng if toks[i].kind == kind]
    blocked = lambda a, b: any(toks[k].kind in ("letter", "cell") for k in range(a, b))
    # G-loc forward: L ... at (R,C); G-from prose: L ... from (R,C)
    for i in letters:
        for rel, key in (("at", "loc"), ("from", "frm")):
            for j in range(i + 1, min(c1, i + P["gap"] + 2)):
                if toks[j].kind == "word" and toks[j].val == rel and j + 1 < c1 and toks[j + 1].kind == "cell":
                    if blocked(i + 1, j):
                        break
                    span, look = range(i + 1, j), _lookback(toks, i, c0, P["lookback"])
                    fr = (_frames(toks, span, look, i - 1 if i > c0 else None, term, out["think"]) if key == "loc"
                          else _from_frames(toks, list(span) + look, term, out["think"]))
                    out[key].append(dict(region=region, clause=cid, letter=toks[i].val, cell=toks[j + 1].val,
                                         rule="fwd", frames=fr, text=" ".join(toks[k].text for k in range(c0, c1))))
                    break
    # G-loc reverse: (R,C) is occupied by / has / holds L
    for i in cellpos:
        for seq in (("is", "occupied", "by"), ("has",), ("holds",)):
            j = i + 1 + len(seq)
            possessive = j + 1 < c1 and toks[j + 1].kind == "word" and toks[j + 1].val == "'s"  # "has B's goal"
            if _seq_at(toks, i + 1, seq) and j < c1 and toks[j].kind == "letter" and not possessive:
                look = _lookback(toks, i, c0, P["lookback"])
                fr = _frames(toks, range(i + 1, j), look, None, term, out["think"])
                out["loc"].append(dict(region=region, clause=cid, letter=toks[j].val, cell=toks[i].val,
                                       rule="rev", frames=fr, text=" ".join(toks[k].text for k in range(c0, c1))))
    if P["move_scope"] == "clause":
        out["moves"] += _pairs_moves(toks, cid, region, list(rng), term)


def _from_frames(toks, idxs, term, think):
    """G-from prose exclusions: past frames ("moved ... from") and future
    frames (modals, conditional and non-factive temporal subordinators)."""
    W = lists()
    hit = set()
    if any(toks[i].kind == "word" and past_form(toks[i].val) for i in idxs):
        hit.add("past")
    if _has(toks, idxs, W["modals"] | W["conditional"] | W["temporal_nonfactive"]):
        hit.add("future")
    if _has(toks, idxs, W["negators"]):
        hit.add("negation")
    if term == "?":
        hit.add("question")
    if think:
        hit.add("think")
    return hit


def move_frame(toks, idxs, term):
    """Frame of a letter-direction pair (spec §5.2): the first that applies of
    negated, question, conditional, past, future (modal), sequence
    (non-factive temporal subordinator), else present."""
    W = lists()
    if _has(toks, idxs, W["negators"]):
        return "negated"
    if term == "?":
        return "question"
    if _has(toks, idxs, W["conditional"]):
        return "conditional"
    if any(toks[i].kind == "word" and past_form(toks[i].val) for i in idxs):
        return "past"
    if _has(toks, idxs, W["modals"]):
        return "future"
    if _has(toks, idxs, W["temporal_nonfactive"]):
        return "sequence"
    return "present"


def _pairs_moves(toks, cid, region, idxs, term):
    """Each letter with the first direction after it and before the next letter.
    The frame span runs from just after the previous pair (or the scope start)
    through the direction. `term` is the clause terminator, or a dict from
    position to terminator when the scope is the whole message."""
    out, prev_end = [], idxs[0] if idxs else 0
    for i in (p for p in idxs if toks[p].kind == "letter"):
        nxt = [p for p in idxs if p > i and toks[p].kind == "letter"]
        stop = nxt[0] if nxt else idxs[-1] + 1
        d = next((p for p in idxs if i < p < stop and toks[p].kind == "dir"), None)
        if d is None:
            continue
        span = [p for p in idxs if prev_end <= p <= d]
        out.append(dict(region=region, clause=cid, letter=toks[i].val, dir=toks[d].val,
                        frame=move_frame(toks, span, term[d] if isinstance(term, dict) else term)))
        prev_end = d + 1
    return out


def _message_moves(cl_all):
    """Curve variant: letter-direction pairing over the whole message, per region (clause -1)."""
    out = []
    for region in dict.fromkeys(c[0] for c in cl_all):
        cls = [c for c in cl_all if c[0] == region]
        idxs = [i for _, _, c0, c1, _ in cls for i in range(c0, c1)]
        term = {i: t for _, _, c0, c1, t in cls for i in range(c0, c1)}
        out += _pairs_moves(cls[0][1], -1, region, idxs, term)
    return out


def target_mentions(an, own_targets):
    """G-target for each own object o -> dict(stated, named) (spec §4).
    stated: a clause with a goal noun, o's target cell, and o's letter or my/its.
    named: o's target cell anywhere in the message."""
    out = {}
    for o, tc in own_targets.items():
        named = any(c == tc for _, _, c, _ in an["cells"])
        stated = any(ci["goal_noun"] and tc in ci["cells"] and (o in ci["letters"] or ci["my_its"])
                     for ci in an["clause_info"])
        out[o] = dict(stated=stated, named=named)
    return out


def analyse_all(eps, P=DEFAULT):
    """{(episode id, ply): analysis} for every message unit (about 20 s on the
    development arm; a process pool is slower, the results pickle badly)."""
    return {(ep["meta"]["id"], t["ply"]): analyse(t["msg"], ep["meta"]["arm"], ep["meta"]["proto"], t["slots"], P)
            for ep in eps for t in ep["turns"] if t["is_msg"]}


# ---------------------------------------------------------------- audits (§9.2)

BROAD_CELL = re.compile(r"(?i)\br\s*\d+\s*,?\s*c\s*\d+|\(\s*\d+\s*,\s*\d+\s*\)|\brow\s+\d+\s*,?\s*(?:and\s+)?col(?:umn)?\s+\d+")
X_AT = re.compile(r"\bX\s+(?:is\s+)?at\s*\(?\s*R\s*\d+\s*,?\s*C\s*\d+", re.I)


def audits(text, an, P=DEFAULT):
    """Omission counts for one message (spec §9.2): broad cell forms outside
    G-cell; prose letter-cell pairs in the adjacency stratum (nearest cell
    within 7 tokens, no letter between) that neither G-loc nor G-from paired,
    forward and reverse order; compass words while G-dir excludes them; goal-noun
    synonyms in a clause with a letter or my/its outside G-target; `X at (R,C)`."""
    W = lists()
    n_cells = sum(t.kind == "cell" for _, toks in an["tokens"] for t in toks)
    res = dict(cell=max(0, len(BROAD_CELL.findall(text or "")) - n_cells), letter_cell_fwd=0, letter_cell_rev=0,
               direction=0, goal_noun=0, x_at=len(X_AT.findall(text or "")))
    got = {(p["region"], p["letter"], p["cell"]) for p in an["loc"] + an["frm"]}
    for region, toks in an["tokens"]:
        for i, t in enumerate(toks):
            if t.kind == "word" and not P["compass"] and t.val in W["compass"]:
                res["direction"] += 1
            if t.kind != "letter" or region != "prose":
                continue
            # the adjacency stratum: the nearest cell on each side within 7
            # tokens with no other letter between; pairs G-loc or G-from
            # already found (in any frame) are no omission
            for side, rng in (("fwd", range(i + 1, min(len(toks), i + 8))), ("rev", range(i - 1, max(-1, i - 8), -1))):
                for j in rng:
                    if toks[j].kind == "letter":
                        break
                    if toks[j].kind == "cell":
                        if (region, t.val, toks[j].val) not in got:
                            res["letter_cell_" + side] += 1
                        break
    res["goal_noun"] = sum(ci["goal_noun_audit"] and not ci["goal_noun"] and bool(ci["letters"] or ci["my_its"])
                           for ci in an["clause_info"])
    return res


def normalise(an, own, regions=("prose", "plan", "request")):
    """§7 vocabulary normalisation of the chosen regions: cells, lines, object
    letters (own / other), seats, X and numbers become placeholders;
    punctuation is dropped."""
    out = []
    for region, toks in an["tokens"]:
        if region not in regions:
            continue
        for t in toks:
            if t.kind == "cell":
                out.append("<cell>")
            elif t.kind == "line":
                out.append("<line>")
            elif t.kind == "letter":
                out.append("<own>" if t.val in own else "<other>")
            elif t.kind == "seat":
                out.append("<seat>")
            elif t.kind == "num":
                out.append("<num>")
            elif t.kind == "dir":
                out.append(t.text.lower())
            elif t.kind == "word":
                out.append("<x>" if t.text == "X" else t.val)
    return out


if __name__ == "__main__":
    if "--build-lists" in sys.argv:
        W = build_lists()
        for k in ("say", "exist", "keep", "put"):
            print(k, len(W[k]), "dropped", W["sources"]["verbnet_dropped_multiword"][k])
        print("goal_noun_audit", W["goal_noun_audit"])
