"""Every output of the message analysis for one scope (spec §13), into out/<scope>/.

    python3 -m analysis.messages.run dev
    python3 -m analysis.messages.run calib   # coverage, frame composition and the cross-arm report only
    python3 -m analysis.messages.run test    # once, when the campaigns are complete

    python3 -m analysis.messages.run dev --only accuracy,failure   # a subset
    python3 -m analysis.messages.run test --models dsv41 --out test_dsv41   # one model's test arm (finish.py)
"""
import csv
import importlib
import json
import math
import os
import sys
import time

from . import corpus, grammar

OUT = os.path.join(corpus.HERE, "out")
# (module, entry) in run order; coverage and cases take extra inputs below
MODULES = ["measures.private_reference", "measures.accuracy", "measures.alignment", "measures.requests_plans",
           "failure", "vocab", "curve", "probe.run", "extract.dest"]
VALIDATION = {"both": "both", "only": "grammar", "grammar": "failed"}  # first word of the §9.6 decision
CALIB_TABLES = ("frame", "audit", "coverage", "cross_arm")  # name prefixes the calibration set may produce


def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, (set, frozenset)):
        return ";".join(sorted(map(str, v)))
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, default=str)
    return v


def write(scope, tables):
    d = os.path.join(OUT, scope)
    os.makedirs(d, exist_ok=True)
    for name, rows in tables.items():
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(d, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows({k: _fmt(v) for k, v in r.items()} for r in rows)


def _read(scope, name):
    p = os.path.join(OUT, scope, f"{name}.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def case_features(eps, A):
    """Per-episode flags the §10 cases are drawn on: a teammate-position
    statement (D2's indicator) and a filled REQUEST."""
    out = dict(teammate_statement={}, request_filled={})
    for ep in eps:
        msgs = [t for t in ep["turns"] if t["is_msg"]]
        out["teammate_statement"][ep["meta"]["id"]] = any(
            grammar.kept(p["frames"], "strict") and ep["owner"][p["letter"]] != t["seat"]
            for t in msgs for p in A[(ep["meta"]["id"], t["ply"])]["loc"])
        out["request_filled"][ep["meta"]["id"]] = any(t["slots"] and not t["slots"]["request_none"] for t in msgs)
    return out


def d2_inputs(scope, eps, A):
    """The §9.6 decision and E5's model-only teammate records for this scope.
    Before the check D2 is 'pending'; the union is computed whenever E5 replies
    exist, and the decision says which version is primary."""
    p = os.path.join(corpus.HERE, "coding", "sample", "metrics.json")
    validation = VALIDATION[json.load(open(p))["decision"].split()[0]] if os.path.exists(p) else "pending"
    model = "gpt-5.6-luna"  # the E5 annotator
    from .extract import records, run as e5
    recs, _, _, annotated, n_units = e5.load_records(eps, "primary", model, A)
    if not annotated:
        return validation, None, []
    units = e5.eligible(eps, A)
    missing = sum(1 for ms in units.values() for ep, t in ms if (ep["meta"]["id"], t["ply"]) not in annotated)
    if missing:
        raise SystemExit(f"refused: {missing} eligible messages have no E5 answer; finish the extraction first")
    tabs = e5.report(eps, A, "primary", model)
    for name in ("alt1", "alt2"):
        tabs = {k: v + e5.report(eps, A, name, model)[k] for k, v in tabs.items()}
    return validation, records.model_only_at(eps, A, recs), tabs


def _opt(argv, name):
    return next((a.split("=", 1)[1] if "=" in a else argv[argv.index(a) + 1] for a in argv if a.startswith(name)), None)


def main(argv):
    scope = argv[1] if len(argv) > 1 else "dev"
    only = _opt(argv, "--only")
    only = set(only.split(",")) if only else None
    models = tuple(_opt(argv, "--models").split(",")) if _opt(argv, "--models") else corpus.MODELS
    out_dir = _opt(argv, "--out") or scope
    t0 = time.time()
    eps = corpus.load(scope, models=models)
    A = grammar.analyse_all(eps)
    print(f"{scope}: {len(eps)} episodes, {len(A)} messages analysed ({time.time() - t0:.0f} s)")
    tables = {}
    if scope != "calib":
        for name in MODULES:
            if only and name.split(".")[-1] not in only:
                continue
            mod = importlib.import_module(f".{name}", __package__)
            t1 = time.time()
            if name == "measures.accuracy":
                validation, extra, e5_tables = d2_inputs(scope, eps, A)
                tables.update(e5_tables or {})
                out = mod.run(scope, eps, A, extra=extra, validation=validation)
            else:
                out = mod.run(scope, eps, A)
            tables.update(out)
            print(f"  {name}: {len(out)} tables ({time.time() - t1:.0f} s)")
    else:  # calibration: frame composition and opportunity counts feed the cross-arm report only
        for name in ("measures.accuracy", "measures.alignment"):
            out = importlib.import_module(f".{name}", __package__).run(scope, eps, A)
            tables.update({k: v for k, v in out.items() if k.startswith(CALIB_TABLES) or k == "opportunity_counts"})
    if not only or "coverage" in only:
        cov = importlib.import_module(".measures.coverage", __package__)
        dev_rates = _read("dev", "audit_rates") if scope != "dev" else None
        opp = tables.get("opportunity_counts")
        tables.update(cov.run(scope, eps, A, dev_rates=dev_rates, opportunity_counts=opp))
    if scope != "calib" and (not only or "cases" in only):
        from . import cases, failure
        tables.update(cases.run(scope, eps, A, features=case_features(eps, A),
                                info_visible=lambda ep, h: failure.info_visible(ep, h, A)))
    if scope != "calib" and (not only or "tables" in only):
        from . import tables as tb
        tables.update(tb.run(scope, tables, out_dir))
    write(out_dir, tables)
    print(f"wrote {len(tables)} tables to {os.path.join(OUT, out_dir)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv)
