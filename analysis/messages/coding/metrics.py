"""§9.6 metrics and decision rule.

    python3 -m analysis.messages.coding.metrics

Per part (grammar, model) and tier: precision (share coded yes) with a
Clopper-Pearson 95 % interval. A part passes when its precision is at least .90
in each tier (compared exactly, as fractions). The sample is strong-tier only,
since the final arm has no Nemotron run, so there is no tier difference. Decision:
- both parts pass: D2 from the union, the grammar-alone version beside it;
- only the grammar part passes: D2 from the grammar alone, the union exploratory;
- the grammar part fails: D2 exploratory under the manifest name of spec §5.4.
A tier with no items fails its part. Misread items are written out as candidate
G-loc unit-test strings: grammar items coded no (negative cases) and model-only
items coded yes (positive cases G-loc missed).
"""
import csv
import json
import os
from fractions import Fraction

from analysis.messages import stats
from analysis.messages.coding.importer import CODED
from analysis.messages.coding.sampler import OUT

FLOOR = Fraction(9, 10)
TIERS = ("strong",)
MANIFEST = "teammate letter–cell pairings in indicative frames, true at the moment of the message"


def part_result(rows):
    """Precision per tier and pass/fail for one part."""
    tiers, frac = {}, {}
    for tier in TIERS:
        r = [x for x in rows if x["tier"] == tier]
        k, n = sum(x["label"] == "yes" for x in r), len(r)
        lo, hi = stats.clopper_pearson(k, n)
        yes = [x for x in r if x["label"] == "yes"]
        tiers[tier] = dict(n=n, yes=k, precision=k / n if n else None, ci_lo=lo, ci_hi=hi,
                           true_among_yes=sum(str(x["truth"]) == "True" for x in yes) / len(yes) if yes else None)
        if n:
            frac[tier] = Fraction(k, n)
    passed = len(frac) == len(TIERS) and all(p >= FLOOR for p in frac.values())
    return dict(tiers=tiers, passed=passed)


def decide(grammar_pass, model_pass):
    if not grammar_pass:
        return f"grammar part fails: D2 is exploratory, reported as \"{MANIFEST}\""
    if model_pass:
        return "both parts pass: D2 from the union, the grammar-alone version beside it"
    return "only the grammar part passes: D2 from the grammar alone, the union exploratory"


def misreads(rows):
    pick = lambda ex, lab: [dict(text=x["text"], letter=x["letter"], cell=x["cell"]) for x in rows
                            if x["extractor"] == ex and x["label"] == lab]
    return dict(negative=pick("grammar", "no"), positive=pick("model", "yes"))


def evaluate(rows):
    parts = {p: part_result([x for x in rows if x["extractor"] == p]) for p in ("grammar", "model")}
    return dict(parts=parts, decision=decide(parts["grammar"]["passed"], parts["model"]["passed"]))


def main():
    rows = list(csv.DictReader(open(CODED)))
    res = evaluate(rows)
    for p, r in res["parts"].items():
        for tier, x in r["tiers"].items():
            prec = "n/a" if x["precision"] is None else f"{x['precision']:.3f} [{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]"
            print(f"{p:8s} {tier:6s} n={x['n']:3d} yes={x['yes']:3d} precision {prec}")
        print(f"{p:8s} {'passes' if r['passed'] else 'fails'}")
    print("decision:", res["decision"])
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "misreads.json"), "w") as f:
        json.dump(misreads(rows), f, indent=1)


if __name__ == "__main__":
    main()
