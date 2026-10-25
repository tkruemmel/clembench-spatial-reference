"""After the §9.6 labelling, one command to the thesis-ready outputs (spec §13 steps 3-6).

    python3 -m analysis.messages.finish --draw                          # E5 on the test arm, both blind samples
    python3 -m analysis.messages.finish ~/Downloads/labels_<id>.json [~/Downloads/labels_<id2>.json]   # the morning
    python3 -m analysis.messages.finish --status                        # what the test arm still waits for
    python3 -m analysis.messages.finish --test --models dsv41           # one model's test arm, before the others complete
    python3 -m analysis.messages.finish --test                          # once, when the campaigns are complete

The draw (spec §9.6): E5 and the E5 destination extraction run on the strong
models' completed test episodes, and coding/sampler and coding/dest draw from
the episodes complete at that point. The complete test run
later reuses these E5 answers.

The morning: import the labels, score §9.6 (and, given a second file, the
destination check of coding/dest), re-run the development outputs that D2 and
the destination measures touch, with D2 named and sourced by the decision, and
build the development thesis pack.

--test refuses unless every final-arm
tree of the models it runs is complete, and unless its output directory does
not exist yet (each test run happens once). It runs E5 and the E5 destination
extraction on the test arm, the E3/E4 probes for DeepSeek, every module, and
the test thesis pack.
- With --models (a partial run) it reads only those models' episodes and
  writes out/test_<models>/; a tier there is those models alone, and the pack
  says so. The complete run later uses the same cached E5 records and probes,
  so its per-model numbers equal the partial run's; it compares them.
- Without --models it needs both strong models' final arm complete and writes
  out/test/. Without a Nemotron final arm D2 is not tested (spec §2).
"""
import csv
import glob
import json
import os
import subprocess
import sys
from unittest import mock

from . import corpus

PY = [sys.executable, "-m"]
STRONG = ("qwen38", "dsv41")
PER_TREE = 4 * 3 * 15  # protocols x levels S2-S4 x boards
E5_MODEL = "gpt-5.6-luna"
WORKERS = {"primary": 16, "dest": 64}  # parallel calls; a medium-effort destination answer takes about 3 min


def sh(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(list(args), check=True, cwd=corpus.REPO)


def status():
    """Completed final-arm episodes per (model, suite) on S2-S4, from file names only."""
    out = {}
    for m in corpus.MODELS:
        for s in ("dyad", "3p", "6p"):
            pat = os.path.join(corpus.REPO, corpus.tree("final", m, s), "*", "*", "S[234]_*", "instance_*",
                               "interactions.json")
            out[(m, s)] = len(glob.glob(pat))
    return out


def print_status():
    st = status()
    for (m, s), n in sorted(st.items()):
        print(f"  {m:9s} {s:5s} {n:3d}/{PER_TREE}{'' if n == PER_TREE else '  incomplete'}")
    return st


def draw():
    for prompt in ("primary", "dest"):
        sh(*PY, "analysis.messages.finish", "--e5", "test", ",".join(STRONG), prompt)
    sh(*PY, "analysis.messages.coding.sampler")
    sh(*PY, "analysis.messages.coding.dest")


def morning(labels, dest_labels=None):
    sh(*PY, "analysis.messages.coding.importer", labels)
    sh(*PY, "analysis.messages.coding.metrics")
    if dest_labels:
        sh(*PY, "analysis.messages.coding.dest", dest_labels)
    # only D2, the destination measures and the validity table depend on the checks; the other outputs stand
    sh(*PY, "analysis.messages.run", "dev", "--only", "accuracy,dest,tables")
    sh(*PY, "analysis.messages.pack", "dev")
    print("\nTest arm:")
    print_status()
    print("then: python3 -m analysis.messages.finish --test [--models dsv41]")


# ---------------------------------------------------------------- E5 on the test arm

def usable(rows, units):
    """Input keys a cached batch answers usably for this run: the reply parses and
    every message of its batch is in `units` (extract.run.load_records reads a
    batch only then, since the acceptance rules need all of its messages)."""
    from .extract import records
    return {k for r in rows if records.parse(r["text"]) is not None and all(x in units for x in r["keys"])
            for k in r["keys"]}


def e5(scope, models, prompt="primary", model=None, max_usd=10.0, rounds=3):
    """E5 for `models`' messages, calling only the inputs no usable cached batch
    answers, in batches of their own, so an earlier run's answers are never
    replaced (load_cache keeps the latest row per input). A partial test run and
    the complete one therefore read the same records for the shared model."""
    from . import grammar
    from .extract import client, dest, run as x
    model = model or (dest.MODEL if prompt == dest.PROMPT else E5_MODEL)  # the cache label each module reads
    eps = corpus.load(scope, models=models)
    units = (dest.eligible if prompt == dest.PROMPT else x.eligible)(eps, grammar.analyse_all(eps))
    real = client.load_cache

    def usable_cache(name, m):
        c = real(name, m)
        ok = usable({id(r): r for r, _ in c.values()}.values(), units)
        return {k: v for k, v in c.items() if k in ok}

    complete = dest.backend() if prompt == dest.PROMPT else client.openai_api("low")
    for _ in range(rounds):  # an unparseable reply stays unusable and is called again
        done = usable_cache(prompt, model)
        rest = {k: v for k, v in units.items() if k not in done}
        print(f"E5 {prompt} {scope} {','.join(models)}: {len(units)} inputs, {len(rest)} to call", flush=True)
        if not rest:
            return
        # extract() skips inputs cached anywhere; here only usable answers count
        with mock.patch.object(client, "load_cache", usable_cache):
            x.extract(prompt, model, complete, rest, workers=WORKERS[prompt], max_usd=max_usd)
    raise SystemExit(f"E5 incomplete after {rounds} rounds; rerun finish --test")


# ---------------------------------------------------------------- the test arm

def out_name(models):
    return "test" if models is None else "test_" + "_".join(models)


def compare(full="test"):
    """Rows of each partial run's models against the complete run, per CSV with
    a model column; prints what differs (tier-level and pooled tables differ by design)."""
    for part in sorted(glob.glob(os.path.join(corpus.HERE, "out", "test_*"))):
        models = set(os.path.basename(part).split("_")[1:])
        same, diff = 0, []
        for p in sorted(glob.glob(os.path.join(part, "*.csv"))):
            q = os.path.join(corpus.HERE, "out", full, os.path.basename(p))
            if not os.path.exists(q):
                continue
            rows = [[r for r in csv.DictReader(open(f)) if r.get("model") in models] for f in (p, q)]
            if not rows[0]:
                continue
            key = lambda r: json.dumps(r, sort_keys=True)
            if sorted(map(key, rows[0])) == sorted(map(key, rows[1])):
                same += 1
            else:
                diff.append(os.path.basename(p)[:-4])
        print(f"{os.path.basename(part)} against {full}: {same} tables identical for {sorted(models)}, "
              f"{len(diff)} differ{': ' + ', '.join(diff) if diff else ''}")


def test(models=None):
    name = out_name(models)
    if os.path.exists(os.path.join(corpus.HERE, "out", name)):
        raise SystemExit(f"refused: out/{name} exists; each test run happens once (spec §4)")
    st = print_status()
    need = models or STRONG
    missing = [k for k in st if k[0] in need and st[k] < PER_TREE]
    if missing:
        raise SystemExit(f"refused: the final arm of {sorted(set(m for m, _ in missing))} is incomplete {missing}")
    if models is None and not any(st[("nemotron", s)] for s in ("dyad", "3p", "6p")):
        print("no Nemotron final arm: D2 and every weak-tier comparison are not tested (spec §2)")
    loaded = models or corpus.MODELS
    for prompt in ("primary", "dest"):
        sh(*PY, "analysis.messages.finish", "--e5", "test", ",".join(loaded), prompt)
    if "dsv41" in loaded:
        for task in ("e3", "e4"):
            sh(*PY, "analysis.messages.probe.run", "test", "--model", "dsv41", "--task", task, "--max-usd", "30")
    sh(*PY, "analysis.messages.run", "test", *(["--models", ",".join(models), "--out", name] if models else []))
    sh(*PY, "analysis.messages.pack", name)
    if models is None:
        compare()


def main(a):
    if "--status" in a:
        print_status()
    elif "--draw" in a:
        draw()
    elif "--e5" in a:  # internal: run by test() in its own process, so the episodes leave memory with it
        i = a.index("--e5")
        e5(a[i + 1], tuple(a[i + 2].split(",")), *a[i + 3:i + 4])
    elif "--test" in a:
        models = tuple(a[a.index("--models") + 1].split(",")) if "--models" in a else None
        if models and not set(models) <= set(corpus.MODELS):
            raise SystemExit(f"unknown model in {models}; choose from {corpus.MODELS}")
        test(models)
    elif a:
        morning(*a[:2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
