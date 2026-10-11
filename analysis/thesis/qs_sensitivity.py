"""Aborts-as-zero sensitivity of the Quality Score comparisons (decision D6).

The Quality Score excludes aborted episodes (clembench's convention).  Here
every QS comparison the chapter leads with is recomputed twice: aborts
excluded (`excl`) and aborts scored 0 (`zero`), and each row says whether the
sign or the p < .05 decision changes.  Comparisons:

  1. cell QS and % Played per model and suite (tab:clembench's cells);
  2. the protocol ladder per suite, S2-S4, paired over (model, level, board);
  3. team-size steps per model, paired over (level, protocol, board), all
     levels and per level;
  4. solo against the dyad, S2-S4, paired over (model, level, board): the
     contrasts of tab:decomposition;
  5. DeepSeek minus Qwen per suite, paired over (level, protocol, board).

Paired tests are tables_comm.perm_p (sign-flip, 100,000 resamples, seed 0).
All exploratory and uncorrected; the confirmatory contrasts are on success and
efficiency, which aborts do not touch differently.

    ARM=updated python3 analysis/thesis/qs_sensitivity.py           # report
    ARM=updated python3 analysis/thesis/qs_sensitivity.py --check   # pinned flips
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tables_comm as tc                                           # noqa: E402

Q, D, N = "Qwen3.8", "DeepSeek-V4.1-Flash", "Nemotron"
GROUPS = {"Qwen": (Q,), "DeepSeek": (D,), "strong": (Q, D), "Nemotron": (N,)}
LADDER = (("none", "structured"), ("structured", "freeform"), ("freeform", "hybrid"),
          ("none", "freeform"), ("none", "hybrid"))
SOLO = (("solo_silent", "none"), ("solo_thinking", "structured"), ("solo_thinking", "freeform"),
        ("solo_thinking", "hybrid"))

# (section, label, flip) of every row whose sign or p < .05 decision flips.
# Pinned for the reported data only; the original arm is not analysed here.
CHECK = {"original": None, "updated": [
    ("ladder", "6p Qwen none->structured", "p<.05"),
    ("model", "all suites DeepSeek-Qwen", "p<.05"),
    ("teamsize", "DeepSeek 3p->6p S2", "p<.05"),
    ("teamsize", "DeepSeek 3p->6p S3", "p<.05"),
    ("teamsize", "DeepSeek 3p->6p all", "sign"),
    ("teamsize", "DeepSeek dyad->6p S4", "p<.05"),
    ("teamsize", "Nemotron dyad->6p S1", "sign"),
    ("teamsize", "Nemotron dyad->6p all", "sign"),
    ("teamsize", "Qwen 3p->6p S4", "sign"),
    ("teamsize", "Qwen dyad->6p S4", "p<.05"),
    ("teamsize", "strong 3p->6p S2", "sign"),
    ("teamsize", "strong 3p->6p S4", "sign"),
    ("teamsize", "strong 3p->6p all", "p<.05"),
    ("teamsize", "strong dyad->6p S2", "p<.05"),
    ("teamsize", "strong dyad->6p S4", "p<.05")]}


def qz(e, conv):
    """Quality of one episode: None for an abort under `excl`, 0 under `zero`."""
    return (0.0 if conv == "zero" else None) if e["ab"] else e["q"]


def paired(a, b, conv):
    """Differences b - a over keys present in both with a defined Quality."""
    d = []
    for k, x in a.items():
        y = b.get(k)
        if y is not None and qz(x, conv) is not None and qz(y, conv) is not None:
            d.append(qz(y, conv) - qz(x, conv))
    return d


ROWS = []


def compare(section, label, a, b):
    """One comparison under both conventions; records whether it flips."""
    out = {}
    for conv in ("excl", "zero"):
        d = paired(a, b, conv)
        same = conv == "zero" and d == paired(a, b, "excl")
        out[conv] = (st.mean(d) if d else float("nan"), len(d),
                     out["excl"][2] if same else tc.perm_p(d) if len(d) > 1 else float("nan"))
    (m1, n1, p1), (m2, n2, p2) = out["excl"], out["zero"]
    flip = []
    if (m1 > 0) != (m2 > 0) and abs(m1) > 1e-9 and abs(m2) > 1e-9:
        flip.append("sign")
    if (p1 < .05) != (p2 < .05):
        flip.append("p<.05")
    ROWS.append((section, label, flip))
    print(f"  {label:<44} excl {m1:+7.2f} (n={n1:>3}, p={p1:.4f})   zero {m2:+7.2f} "
          f"(n={n2:>3}, p={p2:.4f})   {'FLIP ' + '+'.join(flip) if flip else ''}")


def main():
    E = tc.all_episodes(tc.load_optima())

    print(f"ARM={tc.ARM}\n1. Cell Quality Score per model and suite (% Played; QS excl -> zero)")
    for m in (Q, D, N):
        cells = []
        for s in ("solo",) + tc.SUITES:
            g = [e for e in E if e["model"] == m and e["suite"] == s]
            ex = [e["q"] for e in g if not e["ab"]]
            cells.append(f"{s} {100 * len(ex) / len(g):5.1f}% {st.mean(ex) if ex else float('nan'):6.2f}"
                         f"->{st.mean(qz(e, 'zero') for e in g):6.2f}")
        print(f"  {m:<20} " + "   ".join(cells))

    def sel(models, suite, cond=None, levels=None):
        return {(e["model"], e["level"], e["game_id"]) + (() if cond else (e["cond"],)): e
                for e in E if e["model"] in models and e["suite"] == suite
                and (cond is None or e["cond"] == cond) and (levels is None or e["level"] in levels)}

    print("\n2. Protocol ladder per suite, S2-S4, b - a")
    for s in tc.SUITES:
        for gname, models in GROUPS.items():
            for a, b in LADDER:
                compare("ladder", f"{s} {gname} {a}->{b}",
                        sel(models, s, a, tc.LEVELS), sel(models, s, b, tc.LEVELS))

    print("\n3. Team-size steps per model, larger minus smaller (negative = cost)")
    for gname, models in GROUPS.items():
        for a, b in tc.TEAMSIZE_STEPS:
            for lv in (None,) + tc.ALL_LEVELS:
                lvs = None if lv is None else (lv,)
                compare("teamsize", f"{gname} {a}->{b} {lv or 'all'}",
                        sel(models, a, levels=lvs), sel(models, b, levels=lvs))

    print("\n4. Solo against the dyad, S2-S4, dyad minus solo")
    for gname, models in GROUPS.items():
        for a, b in SOLO:
            compare("solo", f"{gname} {a}->{b}",
                    sel(models, "solo", a, tc.LEVELS), sel(models, "dyad", b, tc.LEVELS))

    print("\n5. DeepSeek minus Qwen, paired over (level, protocol, board)")
    for s in ("solo",) + tc.SUITES:
        a = {k[1:]: e for k, e in sel((Q,), s).items()}
        b = {k[1:]: e for k, e in sel((D,), s).items()}
        compare("model", f"{s} DeepSeek-Qwen", a, b)
    key = lambda e: (e["suite"], e["level"], e["game_id"], e["cond"])
    compare("model", "all suites DeepSeek-Qwen",
            {key(e): e for e in E if e["model"] == Q}, {key(e): e for e in E if e["model"] == D})

    flips = sorted((s, l, "+".join(f)) for s, l, f in ROWS if f)
    print(f"\n{len(flips)} of {len(ROWS)} comparisons flip:")
    for f in flips:
        print("  ", *f)
    if "--check" in sys.argv:
        assert CHECK[tc.ARM] is not None, "no pinned flips for this arm"
        assert flips == CHECK[tc.ARM], flips
        print("check ok")


if __name__ == "__main__":
    assert qz({"ab": 1, "q": None}, "zero") == 0.0 and qz({"ab": 1, "q": None}, "excl") is None
    assert paired({1: {"ab": 0, "q": 50.0}}, {1: {"ab": 1, "q": None}}, "zero") == [-50.0]
    main()
