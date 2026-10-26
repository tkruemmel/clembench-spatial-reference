"""E5 records: batch-reply parsing, the acceptance rules, the union with G-loc
and the replay truth (extension §4, E5, narrowed to teammate `at` statements;
spec §5.4, §9.6).

A reply is one JSON array for a batch; each item is
[message number, letter, cell, span]. Acceptance, applied by script and never
repaired:
- the item has that shape, the number names a message of the batch, the
  letter is A-F and the cell is written (Rr,Cc);
- the span is a verbatim substring of that message;
- the letter is a G-letter token of the message inside an occurrence of the
  span (so the article and seat-precedence rules hold in context), and the
  span holds the cell by G-cell.
Rejects are counted by reason: parse, schema, message, span_not_verbatim,
letter_not_in_span, cell_not_in_span.
"""
import collections
import json
import re

from analysis.messages import corpus, grammar

CELL = re.compile(r"^\(R(\d+),C(\d+)\)$")
REASONS = ("parse", "schema", "message", "span_not_verbatim", "letter_not_in_span", "cell_not_in_span")


def parse(text):
    """The JSON array of a reply, or None. A code fence or text around the
    array is tolerated; the items themselves are never altered."""
    s = (text or "").strip()
    for cand in (s, s[s.find("["):s.rfind("]") + 1] if "[" in s else ""):
        try:
            out = json.loads(cand)
        except ValueError:
            continue
        if isinstance(out, list):
            # [[[n, ...], ...]] wraps the answer in one more array: treated as no
            # answer, so the batch is called again, never unwrapped
            if len(out) == 1 and isinstance(out[0], list) and all(isinstance(x, list) for x in out[0]):
                return None
            return out
    return None


def check(item, msgs, arm):
    """Reject reason for one item, or None. msgs: the batch's message texts."""
    if not (isinstance(item, list) and len(item) == 4 and isinstance(item[0], int)
            and all(isinstance(x, str) for x in item[1:]) and re.fullmatch("[A-F]", item[1]) and CELL.match(item[2])
            and item[3].strip()):
        return "schema"
    m, letter, cell, span = item
    if not 1 <= m <= len(msgs):
        return "message"
    msg = msgs[m - 1]
    starts = [x.start() for x in re.finditer(re.escape(span), msg)]
    if not starts:
        return "span_not_verbatim"
    toks = grammar.tokenize(msg, arm, grammar.DEFAULT)
    if not any(any(t.kind == "letter" and t.val == letter and s <= t.i0 and t.i1 <= s + len(span) for t in toks)
               for s in starts):
        return "letter_not_in_span"
    if corpus.uncell(cell) not in {t.val for t in grammar.tokenize(span, arm, grammar.DEFAULT) if t.kind == "cell"}:
        return "cell_not_in_span"
    return None


def accept(text, msgs, arm):
    """({1-based message number: [record dicts]}, [(reason, item)]) for one batch reply.
    A record is dict(letter, cell (zero-based), span)."""
    items = parse(text)
    if items is None:
        return {}, [("parse", text)]
    ok, bad = collections.defaultdict(list), []
    for it in items:
        why = check(it, msgs, arm)
        if why:
            bad.append((why, it))
        else:
            ok[it[0]].append(dict(letter=it[1], cell=corpus.uncell(it[2]), span=it[3]))
    return dict(ok), bad


# ---------------------------------------------------------------- union with G-loc (spec §5.4)

def teammate(ep, t, letter):
    return ep["owner"].get(letter) not in (None, t["seat"])


def gloc_pairs(an, mode="strict"):
    """G-loc (letter, cell) pairs kept under the frame set, every region and rule."""
    return {(p["letter"], p["cell"]) for p in an["loc"] if grammar.kept(p["frames"], mode)}


def model_only_at(eps, A, records):
    """{(episode id, ply, letter, cell)}: accepted teammate `at` records whose
    (message, letter, cell) G-loc (strict, kept) did not produce.
    records: {(episode id, ply): [record dicts]}."""
    out = set()
    for ep, t in corpus.messages(eps):
        k = (ep["meta"]["id"], t["ply"])
        g = gloc_pairs(A[k])
        for r in records.get(k, ()):
            if teammate(ep, t, r["letter"]) and (r["letter"], r["cell"]) not in g:
                out.add(k + (r["letter"], r["cell"]))
    return out


def canonical(p):
    """G-loc pair in a canonical form of spec §5.4: `L at (R,C)` or `L is at (R,C)`."""
    r, c = p["cell"]
    return p["rule"] == "fwd" and bool(re.search(rf"(?<![A-Za-z0-9']){p['letter']} (?:is )?at \(R{r + 1},C{c + 1}\)",
                                                 p["text"]))


def agreement(eps, A, records, annotated):
    """Agreement of the two extractors on teammate `at` statements per tier, over
    the annotated messages, plus the model's recall on canonical G-loc forms."""
    acc = collections.defaultdict(collections.Counter)
    for ep, t in corpus.messages(eps):
        k = (ep["meta"]["id"], t["ply"])
        if k not in annotated:
            continue
        mate = lambda lc: teammate(ep, t, lc[0])
        g = {x for x in gloc_pairs(A[k]) if mate(x)}
        canon = {(p["letter"], p["cell"]) for p in A[k]["loc"]
                 if grammar.kept(p["frames"], "strict") and canonical(p) and mate((p["letter"], p["cell"]))}
        m = {(r["letter"], r["cell"]) for r in records.get(k, ()) if mate((r["letter"], r["cell"]))}
        acc[ep["meta"]["tier"]].update(grammar=len(g), model=len(m), both=len(g & m), canonical=len(canon),
                                       canonical_found=len(canon & m))
    rows = []
    for tier, a in sorted(acc.items()):
        union = a["grammar"] + a["model"] - a["both"]
        rows.append(dict(tier=tier, n_grammar=a["grammar"], n_model=a["model"], n_both=a["both"],
                         grammar_only=a["grammar"] - a["both"], model_only=a["model"] - a["both"],
                         disagree_share=(union - a["both"]) / union if union else None,
                         n_canonical=a["canonical"], canonical_found=a["canonical_found"],
                         canonical_recall=a["canonical_found"] / a["canonical"] if a["canonical"] else None))
    return rows


def truth(ep, t, rec):
    """D2's truth rule: a teammate's object is on the stated cell at B_t-."""
    return t["board"].get(rec["letter"]) == rec["cell"]
