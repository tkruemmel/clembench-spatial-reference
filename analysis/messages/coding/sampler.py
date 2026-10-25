"""§9.6 author check: pools and the seeded blind sample.

    python3 -m analysis.messages.coding.sampler --pools [--model <id>]   # sizes only, writes nothing
    python3 -m analysis.messages.coding.sampler [--model <id>]           # draw once
    python3 -m analysis.messages.coding.sampler --tool                  # rebuild tool.html from items.json

Test arm (S2-S4), drawn from the episodes complete when E5 ran on it (finish
--draw); episodes finished later are left out. Strong tier only:
the final arm has no Nemotron run. Grammar part: GRAMMAR_PER_TIER G-loc teammate
pairs (strict, kept) per tier at random. Model part: MODEL_PER_TIER model-only
teammate `at` records per tier from the primary prompt. A tier with fewer
gives all it has. Messages that exactly repeat an earlier one of the same seat
are left out (the same text gets the same label). Seed SEED, stated here.

The item file holds only what the coder sees: id, message, letter, cell and the
character ranges to highlight. Tier, extractor, replay truth and the episode are
in the key file, which the tool never reads.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

from analysis.messages import corpus, grammar
from analysis.messages.extract import records

SEED = 0
GRAMMAR_PER_TIER = 140
MODEL_PER_TIER = 70
PROMPT = "primary"
MODEL = "gpt-5.6-luna"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample")
ITEMS, KEY, TOOL = (os.path.join(OUT, f) for f in ("items.json", "key.json", "tool.html"))
ITEM_FIELDS = ("id", "text", "letter", "cell", "marks")


# ---------------------------------------------------------------- pools and draw

def pools(eps, A, recs=None):
    """({tier: [(episode id, ply, letter, cell)]} grammar part, same for the model part), sorted."""
    keep = {(ep["meta"]["id"], t["ply"]): (ep, t) for ep, t in corpus.messages(eps) if not t["dup"]}
    gp, mp = {"strong": [], "weak": []}, {"strong": [], "weak": []}
    for (eid, ply), (ep, t) in keep.items():
        for L, c in records.gloc_pairs(A[(eid, ply)]):
            if records.teammate(ep, t, L):
                gp[ep["meta"]["tier"]].append((eid, ply, L, c))
    for x in records.model_only_at(eps, A, recs or {}):
        if x[:2] in keep:
            mp[keep[x[:2]][0]["meta"]["tier"]].append(x)
    return {k: sorted(v) for k, v in gp.items()}, {k: sorted(v) for k, v in mp.items()}


def draw(gp, mp, seed=SEED):
    """[(item key, extractor, tier)] shuffled: up to GRAMMAR_PER_TIER grammar pairs
    and MODEL_PER_TIER model-only records per tier, at random."""
    rng = random.Random(seed)
    picks = []
    for part, pool, k in (("grammar", gp, GRAMMAR_PER_TIER), ("model", mp, MODEL_PER_TIER)):
        for tier in ("strong", "weak"):
            picks += [(x, part, tier) for x in rng.sample(pool[tier], min(k, len(pool[tier])))]
    rng.shuffle(picks)
    return picks


def late(units, annotated):
    """Episode ids with an eligible message E5 has not answered: finished after
    the extraction ran (finish.e5 refuses to leave a seen message unanswered)."""
    return {ep["meta"]["id"] for ms in units.values() for ep, t in ms if (ep["meta"]["id"], t["ply"]) not in annotated}


def marks(text, arm, letter, cell):
    """[i0, i1, kind] of the G-letter tokens of the letter and the G-cell tokens of the
    cell, as UTF-16 offsets (JavaScript string indices)."""
    u16 = lambda i: len(text[:i].encode("utf-16-le")) // 2
    return [[u16(t.i0), u16(t.i1), t.kind] for t in grammar.tokenize(text, arm, grammar.DEFAULT)
            if (t.kind == "letter" and t.val == letter) or (t.kind == "cell" and t.val == cell)]


def build(picks, index):
    """(items, key) from the draw; index maps (episode id, ply) -> (ep, turn)."""
    items, key = [], {}
    for i, ((eid, ply, L, c), extractor, tier) in enumerate(picks, 1):
        ep, t = index[(eid, ply)]
        iid = f"{i:03d}"
        items.append(dict(id=iid, text=t["msg"], letter=L, cell=corpus.cell(c), marks=marks(t["msg"], ep["meta"]["arm"], L, c)))
        key[iid] = dict(tier=tier, extractor=extractor, truth=t["board"].get(L) == c, episode=eid, ply=ply,
                        letter=L, cell=corpus.cell(c), proto=ep["meta"]["proto"])
    return items, key


def write_tool(items_path=ITEMS, out=TOOL, **text):
    """The coding page for an item file; `text` may set ask (question around letter and cell), help and title."""
    raw = open(items_path, "rb").read()
    data = json.dumps(dict(sha=hashlib.sha256(raw).hexdigest(), items=json.loads(raw), **text)).replace("<", "\\u003c")
    with open(out, "w") as f:
        f.write(open(os.path.join(HERE, "tool.html")).read().replace("__DATA__", data))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL, help="annotator model id of the primary extraction")
    ap.add_argument("--pools", action="store_true")
    ap.add_argument("--tool", action="store_true")
    a = ap.parse_args(argv)
    if a.tool:
        write_tool()
        print("wrote", TOOL)
        return
    if not a.pools and os.path.exists(ITEMS):
        raise SystemExit(f"refused: {ITEMS} exists; checked items are never redrawn (spec §9.6)")
    model = a.model
    eps = corpus.load("test")
    A = grammar.analyse_all(eps)
    recs = {}
    if model:
        from analysis.messages.extract.run import eligible, load_records
        units = eligible(eps, A)
        recs, _, _, annotated, n_units = load_records(eps, PROMPT, model, A, units)
        out = late(units, annotated)
        print(f"primary extraction by {model}: {len(annotated)} eligible messages annotated; "
              f"{len(out)} of {len(eps)} episodes finished after it, left out")
        eps = [ep for ep in eps if ep["meta"]["id"] not in out]
    gp, mp = pools(eps, A, recs)
    for part, p in (("grammar", gp), ("model", mp)):
        print(f"{part} pool: " + ", ".join(f"{k} {len(v)}" for k, v in p.items()))
    if a.pools:
        return
    picks = draw(gp, mp)
    items, key = build(picks, {(ep["meta"]["id"], t["ply"]): (ep, t) for ep, t in corpus.messages(eps)})
    os.makedirs(OUT, exist_ok=True)
    with open(ITEMS, "w") as f:
        json.dump(items, f, indent=1)
    with open(KEY, "w") as f:
        json.dump(dict(model=model, seed=SEED, key=key), f, indent=1)
    write_tool()
    n = collections.Counter((x[1], x[2]) for x in picks)
    print(f"wrote {len(items)} items {dict(sorted(n.items()))} to {OUT}; open {TOOL} in a browser")


if __name__ == "__main__":
    main(sys.argv[1:])
