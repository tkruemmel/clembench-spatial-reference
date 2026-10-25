"""Author check of the E5 destination extraction (extract/dest.py), as spec §9.6's model part.

    python3 -m analysis.messages.coding.dest --pools                          # sizes only
    python3 -m analysis.messages.coding.dest                                  # draw once: coding/dest/
    python3 -m analysis.messages.coding.dest ~/Downloads/labels_<id>.json     # import and score

Test arm (S2-S4), freeform, strong tier, drawn like coding/sampler.py from the
episodes complete when the destination extraction ran. Pool: the records the
measures use (each message's commitment per object and its descriptions of this
turn's step, extract/dest.py), from messages that are not exact repeats of the
same seat. PER_TIER records per tier at random (seed SEED), or all of a tier with
fewer. Coded yes if the message says that its writer brings that object to that
cell, this turn or later. The check passes when precision is at least .90
(metrics.part_result). The key holds tier and kind; scoring refuses labels for
another item file.
"""
import collections
import hashlib
import json
import os
import random
import sys

from analysis.messages import corpus, grammar
from analysis.messages.coding import importer, metrics, sampler
from analysis.messages.extract import dest

SEED = 0
PER_TIER = 60
OUT = os.path.join(sampler.HERE, "dest")
ITEMS, KEY, TOOL, METRICS = (os.path.join(OUT, f) for f in ("items.json", "key.json", "tool.html", "metrics.json"))
TEXT = dict(
    title="Author check, destinations",
    ask=["Does the message say that its writer brings object ", " to ", ", this turn or later?"],
    help="Yes if the message says that its writer is moving, or will move, its own lettered object to the "
         "highlighted cell, this turn or in coming turns. No if it does not, for example where the object is now, a "
         "goal stated without a move, a cell only passed, left or avoided, a condition, a question, a negation, a "
         "request to someone else, or another object. Labels are saved in this browser as you go. Export them when "
         "done and pass the downloaded file to python3 -m analysis.messages.coding.dest.")


def pools(eps, A):
    """{tier: [(episode id, ply, letter, cell, kind)]} of the records the measures use, sorted."""
    recs, _, annotated, units = dest.load(eps, A)
    late = sampler.late(units, annotated)
    print(f"destination extraction: {len(late)} of {len(eps)} episodes finished after it, left out")
    out = {"strong": [], "weak": []}
    for ep, t in corpus.messages(eps):
        rs = recs.get((ep["meta"]["id"], t["ply"]))
        if not rs or t["dup"] or ep["meta"]["id"] in late:
            continue
        used = [(r, "now") for r in rs if dest.kind(ep, t, r) == "now"] + [(r, "later") for r in dest.commitments(ep, t, rs)]
        out[ep["meta"]["tier"]] += [(ep["meta"]["id"], t["ply"], r["letter"], r["cell"], k) for r, k in used]
    return {k: sorted(set(v)) for k, v in out.items()}


def draw(pool, seed=SEED):
    rng = random.Random(seed)
    picks = [(x, tier) for tier in ("strong", "weak") for x in rng.sample(pool[tier], min(PER_TIER, len(pool[tier])))]
    rng.shuffle(picks)
    return picks


def build(picks, index):
    items, key = [], {}
    for i, ((eid, ply, L, c, k), tier) in enumerate(picks, 1):
        ep, t = index[(eid, ply)]
        iid = f"{i:03d}"
        items.append(dict(id=iid, text=t["msg"], letter=L, cell=corpus.cell(c),
                          marks=sampler.marks(t["msg"], ep["meta"]["arm"], L, c)))
        key[iid] = dict(tier=tier, kind=k, extractor="model", episode=eid, ply=ply, letter=L, cell=corpus.cell(c),
                        proto=ep["meta"]["proto"], truth=None)
    return items, key


def score(labels_path):
    raw = open(ITEMS, "rb").read()
    k = json.load(open(KEY))
    lab = json.load(open(labels_path))
    if lab["items_sha256"] != hashlib.sha256(raw).hexdigest():
        raise SystemExit("refused: these labels were made for a different item file")
    rows = importer.join(json.loads(raw), k["key"], lab["labels"])
    res = metrics.part_result(rows)
    res["by_kind"] = {kd: sum(r["label"] == "yes" for r in rows if r["kind"] == kd) / max(1, sum(r["kind"] == kd for r in rows))
                      for kd in ("now", "later")}
    with open(METRICS, "w") as f:
        json.dump(res, f, indent=1)
    for tier, x in res["tiers"].items():
        print(f"{tier:6s} n={x['n']:3d} yes={x['yes']:3d} precision {x['precision']:.3f} [{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]")
    print(f"{'passes' if res['passed'] else 'fails'}; by kind {res['by_kind']}")
    return res


def main(argv):
    if argv and not argv[0].startswith("--"):
        return score(argv[0])
    eps = corpus.load("test", protos=("freeform",))
    A = grammar.analyse_all(eps)
    pool = pools(eps, A)
    print("pool:", {k: len(v) for k, v in pool.items()})
    if "--pools" in argv:
        return
    if os.path.exists(ITEMS):
        raise SystemExit(f"refused: {ITEMS} exists; checked items are never redrawn")
    picks = draw(pool)
    items, key = build(picks, {(ep["meta"]["id"], t["ply"]): (ep, t) for ep, t in corpus.messages(eps)})
    os.makedirs(OUT, exist_ok=True)
    with open(ITEMS, "w") as f:
        json.dump(items, f, indent=1)
    with open(KEY, "w") as f:
        json.dump(dict(model=dest.MODEL, seed=SEED, key=key), f, indent=1)
    sampler.write_tool(ITEMS, TOOL, **TEXT)
    print(f"wrote {len(items)} items {dict(collections.Counter((k['tier'], k['kind']) for k in key.values()))} to {OUT}; "
          f"open {TOOL}")


if __name__ == "__main__":
    main(sys.argv[1:])
