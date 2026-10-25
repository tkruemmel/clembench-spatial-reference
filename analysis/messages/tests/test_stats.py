import itertools
from collections import Counter

import numpy as np
from scipy import optimize, stats
from scipy.stats.contingency import odds_ratio

from analysis.messages.stats import (
    clogit, clopper_pearson, exact_conditional_or, holm, mde_or, perm_pvalue,
    permute_within, signflip, strat_bootstrap, studentized_bootstrap, t_interval, wilson)


def test_holm():
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    out = holm([0.01, np.nan, 0.04])
    assert np.isnan(out[1]) and np.allclose([out[0], out[2]], [0.02, 0.04])
    assert holm([0.5, 0.6]) == [1.0, 1.0]


def test_binomial_intervals():
    assert np.allclose(clopper_pearson(0, 10), (0.0, 0.3084971078))
    assert np.allclose(clopper_pearson(5, 10), (0.1870860284, 0.8129139716))
    assert np.allclose(wilson(0, 10), (0.0, 0.2775327999))
    assert np.allclose(wilson(5, 10), (0.2365930905, 0.7634069095))
    assert clopper_pearson(10, 10)[1] == 1.0
    assert all(np.isnan(clopper_pearson(0, 0))) and all(np.isnan(wilson(0, 0)))


def test_mean_intervals():
    x = np.random.default_rng(3).normal(1.0, 2.0, 15)
    m, lo, hi = t_interval(x)
    assert np.allclose((lo, hi), stats.t.interval(0.95, 14, loc=x.mean(), scale=stats.sem(x)))
    mb, blo, bhi = studentized_bootstrap(x, B=4000)
    assert mb == m and blo < m < bhi and studentized_bootstrap(x, B=4000) == (mb, blo, bhi)


def _brute_signflip(d, studentize):
    def st(v):
        return abs(v.mean()) / (v.std(ddof=1) / np.sqrt(v.size)) if studentize else abs(v.mean())
    obs = st(d)
    return np.mean([st(np.array(s) * d) >= obs - 1e-9 for s in itertools.product([1, -1], repeat=d.size)])


def test_signflip_exact_matches_brute_force():
    d = np.array([0.3, -1.2, 2.5, 0.0, 1.1, -0.4, 0.9, 1.7])
    for studentize in (True, False):
        r = signflip(d, studentize=studentize)
        assert r["exact"] and r["n"] == 8
        assert np.isclose(r["p"], _brute_signflip(d, studentize))
    # all-equal nonzero: sd 0, only the two constant flips are as extreme
    assert np.isclose(signflip([2.0] * 5)["p"], 2 / 32)
    assert signflip([1.5])["p"] == 1.0 and signflip([0, 0, 0])["note"]


def test_signflip_exact_vs_monte_carlo():
    d = np.random.default_rng(1).normal(0.5, 1.0, 12)
    ex = signflip(d)
    mc = signflip(d, exact_max=11, B=100000)
    assert ex["exact"] and not mc["exact"]
    assert abs(ex["p"] - mc["p"]) < 0.01


def test_signflip_n20_fast():
    r = signflip(np.random.default_rng(2).normal(0.3, 1.0, 20))
    assert r["exact"] and 0 < r["p"] <= 1


def _fisher_doubling(t):
    return min(1.0, 2 * min(stats.fisher_exact(t, "less").pvalue, stats.fisher_exact(t, "greater").pvalue))


def test_exact_conditional_or_single_table_vs_scipy():
    for t in ([[7, 2], [3, 8]], [[0, 5], [4, 6]], [[3, 0], [2, 5]], [[12, 5], [9, 15]], [[1, 9], [11, 3]]):
        r = exact_conditional_or([tuple(np.ravel(t))])
        ref = odds_ratio(t, kind="conditional")
        ci = ref.confidence_interval(0.95)
        assert np.allclose([r["or_"], r["lo"], r["hi"]], [ref.statistic, ci.low, ci.high], rtol=1e-6)
        assert np.isclose(r["p"], _fisher_doubling(t))
        assert r["n_informative"] == 1 and r["T"] == t[0][0]


def _strata_pmf_T(tables, psi):
    pmf, lo = np.ones(1), 0
    for a, b, c, d in tables:
        n1, n0, m1 = a + b, c + d, a + c
        u = np.arange(max(0, m1 - n0), min(n1, m1) + 1)
        pmf, lo = np.convolve(pmf, stats.nchypergeom_fisher.pmf(u, n1 + n0, n1, m1, psi)), lo + u[0]
    return lo + np.arange(pmf.size), pmf


def test_exact_conditional_or_three_strata():
    tables = [(3, 2, 1, 4), (5, 1, 4, 6), (2, 3, 2, 7)]
    r = exact_conditional_or(tables + [(0, 0, 3, 4)])
    assert r["n_informative"] == 3 and r["n_dropped"] == 1 and r["T"] == 10
    # conditional score equation sum_k E_psi[a_k] = T, solved independently
    score = lambda lp: sum(stats.nchypergeom_fisher.mean(a + b + c + d, a + b, a + c, np.exp(lp))
                           for a, b, c, d in tables) - 10
    assert np.isclose(r["or_"], np.exp(optimize.brentq(score, -10, 10, xtol=1e-12)), rtol=1e-8)
    u, pmf = _strata_pmf_T(tables, r["lo"])
    assert np.isclose(pmf[u >= 10].sum(), 0.025)
    u, pmf = _strata_pmf_T(tables, r["hi"])
    assert np.isclose(pmf[u <= 10].sum(), 0.025)
    u, pmf = _strata_pmf_T(tables, 1.0)
    assert np.isclose(r["p"], min(1, 2 * min(pmf[u <= 10].sum(), pmf[u >= 10].sum())))


def _pairs(n10, n01, n11, n00):
    y, x, s = [], [], []
    for k, (xc, xk) in enumerate([(1, 0)] * n10 + [(0, 1)] * n01 + [(1, 1)] * n11 + [(0, 0)] * n00):
        y += [1, 0]; x += [xc, xk]; s += [k, k]
    return np.array(y), np.array(x), np.array(s)


def test_clogit_matched_pairs():
    r = clogit(*_pairs(7, 3, 5, 4))
    assert not r["separated"] and r["n_strata"] == 19 and r["n_used"] == 38
    assert np.isclose(r["or_"], 7 / 3, rtol=1e-6)
    assert np.isclose(r["se"], np.sqrt(1 / 7 + 1 / 3), rtol=1e-6)
    sep = clogit(*_pairs(6, 0, 2, 2))
    assert sep["separated"] and sep["or_"] == np.inf and np.isnan(sep["se"])


def test_clogit_binary_equals_exact_conditional_mle():
    # one binary covariate: the clogit likelihood is the conditional hypergeometric one
    tables = [(3, 2, 1, 4), (5, 1, 4, 6), (2, 3, 2, 7), (1, 1, 1, 1)]
    y, x, s = [], [], []
    for k, (a, b, c, d) in enumerate(tables):
        for yy, xx, n in ((1, 1, a), (1, 0, b), (0, 1, c), (0, 0, d)):
            y += [yy] * n; x += [xx] * n; s += [k] * n
    y += [1, 1]; x += [0, 1]; s += [99, 99]  # all-case stratum, dropped
    r = clogit(y, x, s)
    assert r["n_strata"] == 4 and r["n_dropped"] == 1
    assert np.isclose(r["or_"], exact_conditional_or(tables)["or_"], rtol=1e-6)


def test_clogit_recursion_vs_subset_enumeration():
    rng = np.random.default_rng(5)
    sizes, cases = [5, 6, 4, 6, 5, 6, 5, 4], [2, 3, 1, 2, 3, 2, 2, 2]
    X = rng.normal(size=(sum(sizes), 2))
    s = np.repeat(np.arange(len(sizes)), sizes)
    y = np.concatenate([rng.permutation([1] * m + [0] * (n - m)) for n, m in zip(sizes, cases)])
    X[y == 1, 0] += 0.8

    def ll(b):
        tot = 0.0
        for k in range(len(sizes)):
            eta = X[s == k] @ b
            den = [eta[list(S)].sum() for S in itertools.combinations(range(eta.size), cases[k])]
            tot += eta[y[s == k] == 1].sum() - np.logaddexp.reduce(den)
        return tot

    r = clogit(y, X, s)
    assert not r["separated"]
    assert np.isclose(r["loglik"], ll(r["beta"]), rtol=1e-10)
    ref = optimize.minimize(lambda b: -ll(b), np.zeros(2), method="BFGS", options={"gtol": 1e-9})
    assert np.allclose(r["beta"], ref.x, atol=1e-4)
    h, e = 1e-4, np.eye(2)
    H = np.array([[(ll(r["beta"] + h * (e[i] + e[j])) - ll(r["beta"] + h * (e[i] - e[j]))
                    - ll(r["beta"] - h * (e[i] - e[j])) + ll(r["beta"] - h * (e[i] + e[j]))) / (4 * h * h)
                   for j in range(2)] for i in range(2)])
    assert np.allclose(r["se"], np.sqrt(np.diag(np.linalg.inv(-H))), rtol=1e-3)


def test_strat_bootstrap():
    rng = np.random.default_rng(0)
    strata = ["L1"] * 5 + ["L2"] * 7 + ["L3"] * 3
    units = [(st, v) for st, v in zip(strata, rng.normal(size=15))]
    sizes = Counter(strata)

    def stat(items):
        assert Counter(st for st, _ in items) == sizes
        return float(np.mean([v for _, v in items]))

    a = strat_bootstrap(units, strata, stat, B=500, seed=0)
    assert a == strat_bootstrap(units, strata, stat, B=500, seed=0)
    assert a != strat_bootstrap(units, strata, stat, B=500, seed=1)
    assert a[1] < a[0] < a[2]
    # ratio estimator on (numerator, denominator) units
    r = strat_bootstrap([(1, 2), (3, 4), (0, 1), (2, 2)], [0, 0, 1, 1],
                        lambda it: sum(n for n, _ in it) / sum(d for _, d in it), B=500)
    assert np.isclose(r[0], 6 / 9) and r[1] <= r[0] <= r[2]


def test_permutation():
    rng = np.random.default_rng(0)
    lab, strata = np.arange(12), np.repeat([0, 1, 2], 4)
    out = permute_within(lab, strata, rng)
    assert all(sorted(out[strata == k]) == sorted(lab[strata == k]) for k in range(3))
    x = np.r_[np.zeros(10), np.ones(10)] + np.random.default_rng(1).normal(0, 0.3, 20)
    g = np.r_[np.zeros(10), np.ones(10)]
    diff = lambda dat: dat[0][dat[1] == 1].mean() - dat[0][dat[1] == 0].mean()
    perm = lambda dat, r: (dat[0], r.permutation(dat[1]))
    obs, p = perm_pvalue(diff, (x, g), perm, B=500)
    assert obs > 0.5 and p < 0.01
    assert perm_pvalue(diff, (x, g), perm, B=500) == (obs, p)


def test_mde_or_monotone_in_sample_size():
    small, mid, big = (mde_or(n, n, 0.3) for n in (20, 60, 200))
    assert np.isfinite(small) and small > mid > big > 1
    assert mde_or(2, 2, 0.3) == np.inf
