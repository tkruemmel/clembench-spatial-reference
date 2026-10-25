"""Vocabulary ranking (spec §7): exploratory, chooses no words.

Method: Monroe, Colaresi & Quinn 2008, the weighted log-odds ratio with an
informative Dirichlet prior (eq. 16, 20, 22), on unigrams and bigrams.

Terms. `grammar.normalise(an, own, (region,))` per message and region, `own`
the speaker's objects, so cells, lines, letters (<own>/<other>), seats, X and
numbers are placeholders and punctuation is dropped. Bigrams are adjacent pairs
within one region's token sequence of one message. Unigrams and bigrams form one
multinomial. Every accepted message in the window counts; exact repeats within a
seat are not removed (this is an episode-level distribution, not a message rate).

Weighting. Each episode's term counts are rescaled to the same total N / E (N
the total term count of the contrast, E its episodes), so every episode weighs
the same and N is preserved. A group's total n_i = E_i N / E is then fixed under
relabelling.

Prior. The pooled corpus of the two groups: a_w = y_w (the pooled rescaled count),
so alpha_0 = N, a prior as heavy as the data. A documented deviation from Monroe
fn. 18 (prior from other text): the pooled prior does not depend on the labels,
so the permutation below is exact, and it puts the null mean of z at about 0.

  delta_w = log((y_wi + a_w) / (n_i + a0 - y_wi - a_w)) - log((y_wj + a_w) / (n_j + a0 - y_wj - a_w))
  var_w   = 1 / (y_wi + a_w) + 1 / (y_wj + a_w);   z_w = delta_w / sqrt(var_w)

Calibration. Episode labels are permuted within strata (the design cell other
than the contrasted factor; per contrast in CONTRASTS), B = 2,000, seed 0. A
stratum that holds only one side carries no within-stratum contrast and is
dropped (counted in the meta table). Each term's z is studentized against its
own permutation distribution: z_stud = (z - mean z*) / sd z*. The centring
removes what the strata confound (a term frequent in the levels where one side
is frequent); without confounding mean z* is about 0. Family-wise threshold:
the 95th percentile over permutations of max_w |z*_stud|.

Support. Only terms used in at least MIN_SUPPORT = 5 episodes (both sides
together) are ranked; the dropped terms stay in the totals n_i and N.

Windows. `all` every message; `prefatal` messages before corpus.first_fatal
(spec §6 scope rule, every outcome comparison); `first` each speaker's first
message, kept only if it is before the first fatal decision.

Reporting. The top 30 terms per side with episode support and whether
|z_stud| exceeds the threshold. No aggregate divergence (Gentzkow, Shapiro &
Taddy 2019).
"""
import collections

import numpy as np
from scipy import sparse

from . import corpus, grammar, stats

B, SEED, MIN_SUPPORT, TOP, CHUNK = 2000, 0, 5, 30, 100
PROSE, SLOTS, TALK = ("prose",), ("plan", "request"), ("freeform", "hybrid")


def _contrast(name, sides, regions, window, keep, is_a, strata):
    return dict(name=name, sides=sides, regions=regions, window=window, keep=keep, is_a=is_a, strata=strata)


CONTRASTS = [
    _contrast("freeform_vs_hybrid_prose", ("freeform", "hybrid"), PROSE, "all",
              lambda m: m["proto"] in TALK, lambda m: m["proto"] == "freeform", ("model", "suite", "level")),
    _contrast("structured_vs_hybrid_slots", ("structured", "hybrid"), SLOTS, "all",
              lambda m: m["proto"] in ("structured", "hybrid"), lambda m: m["proto"] == "structured",
              ("model", "suite", "level")),
    _contrast("dyad_vs_6p_prose", ("dyad", "6p"), PROSE, "all",
              lambda m: m["proto"] in TALK and m["suite"] in ("dyad", "6p"), lambda m: m["suite"] == "dyad",
              ("model", "proto", "level")),
    _contrast("strong_vs_weak_freeform", ("strong", "weak"), PROSE, "all",
              lambda m: m["proto"] == "freeform", lambda m: m["tier"] == "strong", ("suite", "level")),
] + [
    _contrast(f"success_vs_failure_{s}_{w}", ("success", "failure"), PROSE, w,
              lambda m, s=s: m["proto"] in TALK and m["suite"] == s, lambda m: m["success"] == 1,
              ("model", "proto", "level"))
    for s in ("3p", "6p") for w in ("first", "prefatal")
]


# ---------------------------------------------------------------- terms

def window_turns(ep, window):
    """The episode's message turns in a window (module docstring)."""
    T = [t for t in ep["turns"] if t["is_msg"]]
    if window == "all":
        return T
    if window == "first":
        seen = set()
        T = [t for t in T if not (t["seat"] in seen or seen.add(t["seat"]))]
    fatal = corpus.first_fatal(ep)
    return [t for t in T if fatal is None or t["ply"] < fatal]


def episode_terms(ep, A, regions, window):
    """Counter of unigrams and within-region bigrams over the episode's window."""
    c = collections.Counter()
    for t in window_turns(ep, window):
        an = A[(ep["meta"]["id"], t["ply"])]
        for r in regions:
            w = grammar.normalise(an, ep["objs"][t["seat"]], (r,))
            c.update(w)
            c.update(map(" ".join, zip(w, w[1:])))
    return c


def rescale(docs):
    """E x V csr of term counts, each row rescaled to N / E. Returns (X, vocab, N)."""
    vocab = sorted(set().union(*docs))
    ix = {w: i for i, w in enumerate(vocab)}
    trip = [(e, ix[w], v) for e, d in enumerate(docs) for w, v in d.items()]
    rows, cols, vals = (np.array(x) for x in zip(*trip))
    X = sparse.csr_matrix((vals.astype(float), (rows, cols)), shape=(len(docs), len(vocab)))
    n = np.asarray(X.sum(1)).ravel()
    N = float(n.sum())
    return (sparse.diags(N / len(docs) / n) @ X).tocsr(), vocab, N


# ---------------------------------------------------------------- statistic

def monroe(yi, yj, a, ni, nj, a0):
    """delta, var and z of the log-odds ratio i - j with prior pseudo-counts a (total a0)."""
    d = np.log((yi + a) / (ni + a0 - yi - a)) - np.log((yj + a) / (nj + a0 - yj - a))
    v = 1 / (yi + a) + 1 / (yj + a)
    return d, v, d / np.sqrt(v)


def perm_labels(g, strata, B=B, seed=SEED):
    """E x B boolean labels, each column g permuted within strata (stats.permute_within)."""
    rng = np.random.default_rng(seed)
    return np.stack([stats.permute_within(g, strata, rng) for _ in range(B)], axis=1)


def rank(X, g, strata, N, B=B, seed=SEED, min_support=MIN_SUPPORT):
    """Monroe z per term with support >= min_support, studentized by the stratified
    permutation null, and the family-wise threshold. Arrays over the kept terms."""
    g = np.asarray(g, bool)
    sup = np.asarray((X > 0).sum(0)).ravel()
    keep = np.flatnonzero(sup >= min_support)
    Xt = X[:, keep].T.tocsr()
    E = g.size
    ni = N / E * g.sum()
    nj = N - ni
    y = np.asarray(Xt.sum(1)).ravel()  # pooled counts = the prior's a_w

    def z_of(G):  # G: E x b labels -> V x b
        yi = Xt @ G.astype(float)
        return monroe(yi, y[:, None] - yi, y[:, None], ni, nj, N)[2]

    y_a = Xt @ g.astype(float)
    d, _, z = monroe(y_a, y - y_a, y, ni, nj, N)
    G = perm_labels(g, strata, B, seed)
    s = ss = 0
    for c in range(0, B, CHUNK):
        Z = z_of(G[:, c:c + CHUNK])
        s, ss = s + Z.sum(1), ss + (Z * Z).sum(1)
    mu = s / B
    sd = np.sqrt(np.maximum(ss / B - mu * mu, 0))
    sd_safe = np.where(sd > 0, sd, np.inf)  # a term constant under relabelling gets z_stud 0
    mx = np.concatenate([np.abs((z_of(G[:, c:c + CHUNK]) - mu[:, None]) / sd_safe[:, None]).max(0)
                         for c in range(0, B, CHUNK)])
    bt = (Xt > 0).astype(float)
    return dict(keep=keep, z=z, delta=d, mu=mu, sd=sd, z_stud=(z - mu) / sd_safe,
                threshold=float(np.quantile(mx, 0.95)), y_a=y_a, y_b=y - y_a,
                sup_a=bt @ g.astype(float), sup_b=bt @ (~g).astype(float), n_terms_all=X.shape[1])


# ---------------------------------------------------------------- entry point

def run(scope, eps, A, B=B, seed=SEED, min_support=MIN_SUPPORT):
    """{vocab_<contrast>: top-30 rows per side, vocab_meta: one row per contrast}."""
    out, meta = {}, []
    for c in CONTRASTS:
        a, b = c["sides"]
        sel = [ep for ep in eps if ep["meta"]["proto"] != "none" and c["keep"](ep["meta"])]
        docs = [episode_terms(ep, A, c["regions"], c["window"]) for ep in sel]
        empty = sum(not d for d in docs)
        sel, docs = [ep for ep, d in zip(sel, docs) if d], [d for d in docs if d]
        st = [tuple(ep["meta"][k] for k in c["strata"]) for ep in sel]
        g = [bool(c["is_a"](ep["meta"])) for ep in sel]
        sides = collections.defaultdict(set)
        for s, x in zip(st, g):
            sides[s].add(x)
        used = {s for s, v in sides.items() if len(v) == 2}
        sel_i = [i for i, s in enumerate(st) if s in used]
        g = np.array([g[i] for i in sel_i], bool)
        row = dict(contrast=c["name"], scope=scope, side_a=a, side_b=b, regions="+".join(c["regions"]),
                   window=c["window"], strata=" x ".join(c["strata"]), n_strata=len(used),
                   n_strata_dropped=len(sides) - len(used), n_episodes_a=int(g.sum()), n_episodes_b=int((~g).sum()),
                   n_episodes_empty=empty, n_episodes_one_sided=len(sel) - len(sel_i), prior="pooled two groups, a_w = y_w", B=B, seed=seed,
                   min_support=min_support)
        if not used:
            meta.append(dict(row, note="skipped: no stratum holds both sides"))
            continue
        X, terms, N = rescale([docs[i] for i in sel_i])
        r = rank(X, g, [st[i] for i in sel_i], N, B, seed, min_support)
        zs, thr = r["z_stud"], r["threshold"]
        row.update(tokens_N=N, alpha0=N, n_terms_all=r["n_terms_all"], n_terms_ranked=zs.size, threshold=thr,
                   n_above_a=int((zs > thr).sum()), n_above_b=int((zs < -thr).sum()), note="")
        meta.append(row)
        order = np.argsort(-zs, kind="stable")
        rows = []
        for side, idx in ((a, order[zs[order] > 0]), (b, order[::-1][zs[order[::-1]] < 0])):
            for k, i in enumerate(idx[:TOP], 1):
                w = terms[r["keep"][i]]
                rows.append(dict(contrast=c["name"], side=side, rank=k, term=w, n_gram=w.count(" ") + 1,
                                 z_stud=float(zs[i]), z_monroe=float(r["z"][i]), perm_mean=float(r["mu"][i]),
                                 perm_sd=float(r["sd"][i]), delta=float(r["delta"][i]),
                                 count_a=float(r["y_a"][i]), count_b=float(r["y_b"][i]),
                                 support_a=int(r["sup_a"][i]), support_b=int(r["sup_b"][i]),
                                 above_threshold=bool(abs(zs[i]) > thr)))
        out[f"vocab_{c['name']}"] = rows
    out["vocab_meta"] = meta
    return out
