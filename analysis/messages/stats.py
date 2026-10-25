"""Statistics layer of the message analysis.

Spec: thesis/message_analysis_design.md §6 (failure analysis) and §8 (statistics).
Cell intervals (Wilson, Clopper-Pearson, t, studentized bootstrap, level-stratified
bootstrap), the exact board-paired sign-flip test, Holm, exact conditional odds
ratios over stratified 2x2 tables, conditional logistic regression, a seeded
stratified permutation p-value and the minimum detectable odds ratio.
numpy and scipy only; every random step takes a seed.
"""
import numpy as np
from scipy import optimize, special, stats

TOL = 1e-12        # tie tolerance when comparing a resampled statistic to the observed
CHUNK = 1 << 16    # rows per block of sign vectors


def _groups(strata):
    """Index arrays per stratum, in order of first appearance (keeps seeded draws stable)."""
    g = {}
    for i, s in enumerate(strata):
        g.setdefault(s, []).append(i)
    return [np.asarray(v) for v in g.values()]


# ---------------------------------------------------------------- multiplicity

def holm(pvals):
    """Holm step-down adjusted p-values in input order. NaN passes through and is
    left out of the family size."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = ok.size
    if m:
        o = ok[np.argsort(p[ok], kind="stable")]
        out[o] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    return out.tolist()


# ---------------------------------------------------------------- cell intervals

def wilson(k, n, alpha=0.05):
    """Wilson score interval for k successes in n."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p, z2 = k / n, z * z / n
    c = (p + z2 / 2) / (1 + z2)
    h = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n)) / (1 + z2)
    return (max(0.0, float(c - h)), min(1.0, float(c + h)))


def clopper_pearson(k, n, alpha=0.05):
    """Exact Clopper-Pearson interval from beta quantiles."""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def t_interval(x, alpha=0.05):
    """Mean and Student-t interval; bounds are NaN below two observations."""
    x = np.asarray(x, float)
    n = x.size
    m = float(x.mean()) if n else np.nan
    if n < 2:
        return (m, np.nan, np.nan)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (m, float(m - h), float(m + h))


def studentized_bootstrap(x, B=10000, seed=0, alpha=0.05):
    """Mean and bootstrap-t interval [m - q(1-a/2) se, m - q(a/2) se]."""
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return (float(x.mean()) if n else np.nan, np.nan, np.nan)
    m, se = float(x.mean()), x.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return (m, m, m)
    xs = x[np.random.default_rng(seed).integers(0, n, (B, n))]
    dev = xs.mean(1) - m
    with np.errstate(divide="ignore", invalid="ignore"):
        # a resample with sd 0 and a shifted mean is an extreme replicate (+-inf)
        t = np.where(dev == 0, 0.0, dev / (xs.std(1, ddof=1) / np.sqrt(n)))
    # inverted_cdf picks order statistics, so infinite replicates never interpolate to NaN
    qlo, qhi = np.quantile(t, [alpha / 2, 1 - alpha / 2], method="inverted_cdf")
    return (m, float(m - qhi * se), float(m - qlo * se))


def strat_bootstrap(units, strata, stat, B=10000, seed=0, alpha=0.05):
    """Percentile interval for stat(units), resampling units with replacement within
    each stratum at its own size. Replicates where stat is NaN are dropped."""
    units = list(units)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([g[rng.integers(0, g.size, (B, g.size))] for g in _groups(strata)], axis=1)
    reps = np.array([stat([units[i] for i in row]) for row in idx], float)
    lo, hi = np.nanquantile(reps, [alpha / 2, 1 - alpha / 2])
    return (stat(units), float(lo), float(hi))


# ---------------------------------------------------------------- sign-flip test

def _flip_stat(sums, n, ss, studentize):
    """|mean| or |mean| / (sd / sqrt n) of each sign-flipped vector, from its sum.
    sum of squares is flip-invariant, so sd follows from the mean. sd 0 under a flip
    happens only when all |d_i| are equal and the signs agree, the largest |mean| a
    flip can reach; that flip gets +inf so it ranks as the most extreme."""
    m = np.abs(sums) / n
    if not studentize:
        return m
    var = np.maximum(ss - n * m * m, 0.0) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var <= TOL * ss / n, np.inf, m / np.sqrt(var / n))


def signflip(d, seed=0, B=100000, studentize=True, exact_max=20):
    """Board-paired sign-flip test of H0 d symmetric about 0 (Canay, Romano & Shaikh 2017).

    Two-sided via |statistic|; zero differences are kept (they count in n and sd).
    Exact over all 2^n sign vectors when n <= exact_max, otherwise Monte Carlo with
    p = (1 + #{stat* >= obs}) / (1 + B). Because sum d_i^2 is flip-invariant the
    studentized statistic is monotone in |mean|, so studentize changes `stat`, not `p`.
    """
    d = np.asarray(d, float).ravel()
    n = d.size
    mean = float(d.mean()) if n else np.nan
    if n < 2 or not d.any():
        return dict(n=n, mean=mean, stat=np.nan, p=1.0, exact=True,
                    note="fewer than 2 pairs or all differences zero")
    ss = float(d @ d)
    obs = float(_flip_stat(np.array([d.sum()]), n, ss, studentize)[0])
    thr = obs if np.isinf(obs) else obs - TOL * max(1.0, obs)
    exact = n <= exact_max
    total = 2 ** n if exact else B
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, total, CHUNK):
        k = min(CHUNK, total - start)
        if exact:  # row j of the block is the binary expansion of start + j
            bits = (np.arange(start, start + k)[:, None] >> np.arange(n)) & 1
        else:
            bits = rng.integers(0, 2, (k, n))
        hits += int((_flip_stat((1 - 2 * bits) @ d, n, ss, studentize) >= thr).sum())
    p = hits / total if exact else (1 + hits) / (1 + B)
    return dict(n=n, mean=mean, stat=obs, p=p, exact=exact, note=None)


# ---------------------------------------------------------------- exact conditional OR

def _lbinom(n, k):
    return special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)


def _logconv(x, y):
    """Product of two polynomials given as log-coefficient vectors."""
    out = np.full(x.size + y.size - 1, -np.inf)
    for i, xi in enumerate(x):
        out[i:i + y.size] = np.logaddexp(out[i:i + y.size], xi + y)
    return out


def _root(g, target):
    """theta with g(theta) = target for increasing g, bracketing outward from [-1, 1]."""
    lo, hi = -1.0, 1.0
    while g(lo) > target and lo > -1e4:
        lo *= 2
    while g(hi) < target and hi < 1e4:
        hi *= 2
    return optimize.brentq(lambda th: g(th) - target, lo, hi, xtol=1e-12)


def exact_conditional_or(tables, alpha=0.05):
    """Common odds ratio over stratified 2x2 tables (a, b, c, d) by exact conditional
    inference (Breslow & Day vol. I §4.4, vol. II).

    a = exposed cases, b = unexposed cases, c = exposed controls, d = unexposed controls.
    Given all margins T = sum a_k is a sum of independent noncentral hypergeometrics;
    its pmf in psi is built by log-space convolution. Returns the conditional MLE, the
    exact CI from two one-sided tests at alpha/2 each, and the two-sided p at psi = 1
    by doubling the smaller tail (no mid-p). Strata with a zero margin are dropped.
    """
    lc, tmin, T, K, n_all = np.zeros(1), 0, 0, 0, 0
    for a, b, c, d in tables:
        n_all += 1
        n1, n0, m1 = a + b, c + d, a + c
        if min(n1, n0, m1, b + d) == 0:
            continue
        lo = max(0, m1 - n0)
        u = np.arange(lo, min(n1, m1) + 1)
        lc = _logconv(lc, _lbinom(n1, u) + _lbinom(n0, m1 - u))
        tmin, T, K = tmin + lo, T + a, K + 1
    out = dict(T=T, n_informative=K, n_dropped=n_all - K)
    if K == 0:
        return dict(or_=np.nan, lo=np.nan, hi=np.nan, p=np.nan, **out)
    u = tmin + np.arange(lc.size)
    t = T - tmin

    def logw(th):
        w = lc + th * u
        return w - special.logsumexp(w)

    def logsf(th):   # log P(T >= t)
        return special.logsumexp(logw(th)[t:])

    def logcdf(th):  # log P(T <= t)
        return special.logsumexp(logw(th)[:t + 1])

    at_min, at_max = t == 0, t == lc.size - 1
    la = np.log(alpha / 2)
    return dict(
        or_=0.0 if at_min else np.inf if at_max else float(np.exp(_root(lambda th: np.exp(logw(th)) @ u, T))),
        lo=0.0 if at_min else float(np.exp(_root(logsf, la))),
        hi=np.inf if at_max else float(np.exp(_root(lambda th: -logcdf(th), -la))),
        p=float(min(1.0, 2 * np.exp(min(logsf(0.0), logcdf(0.0))))),
        **out)


# ---------------------------------------------------------------- conditional logistic regression

def _esf(eta, x, m):
    """Gail, Lubin & Rubinstein (1981) recursion in log space for one stratum.

    Over subsets S of size m weighted by exp(sum_{i in S} eta_i): returns log of the
    elementary symmetric sum (the likelihood denominator) and the mean and covariance
    of s = sum_{i in S} x_i (score and information terms). Adding item i mixes
    'S without i' and 'S with i'; carrying mixture moments keeps everything bounded.
    """
    p = x.shape[1]
    logB = np.full(m + 1, -np.inf)
    logB[0] = 0.0
    mu, cov = np.zeros((m + 1, p)), np.zeros((m + 1, p, p))
    for i in range(eta.size):
        l0, l1 = logB[1:], eta[i] + logB[:-1]
        new = np.logaddexp(l0, l1)
        live = np.isfinite(new)
        with np.errstate(invalid="ignore"):
            w0 = np.where(live, np.exp(l0 - new), 0.0)
            w1 = np.where(live, np.exp(l1 - new), 0.0)
        dm = mu[:-1] + x[i] - mu[1:]
        cov[1:] = (w0[:, None, None] * cov[1:] + w1[:, None, None] * cov[:-1]
                   + (w0 * w1)[:, None, None] * dm[:, :, None] * dm[:, None, :])
        mu[1:] = mu[1:] + w1[:, None] * dm
        logB[1:] = new
    return logB[m], mu[m], cov[m]


def _clogit_parts(beta, data):
    """Conditional log-likelihood, score and observed information."""
    ll, g, H = 0.0, np.zeros(beta.size), np.zeros((beta.size, beta.size))
    for x, m, s in data:
        lb, mu, cov = _esf(x @ beta, x, m)
        ll, g, H = ll + s @ beta - lb, g + s - mu, H + cov
    return ll, g, H


def clogit(y, X, strata, maxiter=50, tol=1e-8):
    """Conditional logistic regression on the exact conditional likelihood (any number
    of cases per stratum), Newton-Raphson with step halving from 0, Wald SE.

    Strata with all or no cases are dropped. separated=True when Newton does not
    converge or the information is (near) singular at the end: an infinite MLE from
    separation, or a covariate with no within-stratum variation. Then the affected
    coordinates get beta = +-inf (NaN if beta never left 0) and SE/CI are NaN.
    1-D X gives scalar outputs, 2-D X arrays.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    one = X.ndim == 1
    X = X.reshape(y.size, -1)
    data, n_used, n_drop = [], 0, 0
    for g in _groups(strata):
        m = int(y[g].sum())
        if 0 < m < g.size:
            data.append((X[g], m, X[g][y[g] == 1].sum(0)))
            n_used += g.size
        else:
            n_drop += 1
    p = X.shape[1]
    beta, step, ll, separated = np.zeros(p), np.zeros(p), np.nan, False
    if not data:
        beta = np.full(p, np.nan)
    else:
        ll, g, H = _clogit_parts(beta, data)
        D = np.sqrt(np.diag(H))
        D[D == 0] = 1.0
        converged = False
        for _ in range(maxiter):
            step = np.linalg.lstsq(H, g, rcond=None)[0]  # min-norm step if H is singular
            t = 1.0
            while True:
                nb = beta + t * step
                parts = _clogit_parts(nb, data)
                if parts[0] >= ll - 1e-12 or t < 1e-6:
                    break
                t /= 2
            beta, (ll, g, H) = nb, parts
            if np.abs(t * step).max() < tol:
                converged = True
                break
        # Under separation Newton either keeps stepping or stalls once the score rounds
        # to 0; either way the information along the divergent direction has collapsed.
        # Scaled by the information at 0, so the check does not depend on covariate units.
        w, Q = np.linalg.eigh(H / np.outer(D, D))
        null = w <= 1e-8
        separated = not converged or bool(null.any())
    if separated:
        se = np.full(p, np.nan)
        moving = (np.abs(Q[:, null]) > 1e-3).any(1) | (np.abs(step) > 1e-3)
        with np.errstate(invalid="ignore"):
            beta = np.where(moving, np.where(beta != 0, np.copysign(np.inf, beta), np.nan), beta)
    else:
        se = np.sqrt(np.diag(np.linalg.inv(H))) if data else np.full(p, np.nan)
    z = stats.norm.ppf(0.975)
    with np.errstate(invalid="ignore", over="ignore"):
        or_, lo, hi = np.exp(beta), np.exp(beta - z * se), np.exp(beta + z * se)
    sq = (lambda v: float(v[0])) if one else (lambda v: v)
    return dict(beta=sq(beta), se=sq(se), or_=sq(or_), ci_lo=sq(lo), ci_hi=sq(hi),
                loglik=float(ll), n_strata=len(data), n_used=n_used, n_dropped=n_drop,
                separated=separated)


# ---------------------------------------------------------------- permutation

def permute_within(labels, strata, rng):
    """Copy of labels permuted within each stratum."""
    labels = np.asarray(labels)
    out = labels.copy()
    for g in _groups(strata):
        out[g] = labels[rng.permutation(g)]
    return out


def perm_pvalue(stat_fn, data, permute_fn, B=2000, seed=0):
    """Observed stat_fn(data) and the seeded Monte Carlo permutation p, two-sided via
    |stat|: p = (1 + #{|stat*| >= |obs| - 1e-12}) / (1 + B)."""
    rng = np.random.default_rng(seed)
    obs = stat_fn(data)
    thr = abs(obs) - TOL
    hits = sum(abs(stat_fn(permute_fn(data, rng))) >= thr for _ in range(B))
    return obs, (1 + hits) / (1 + B)


# ---------------------------------------------------------------- power

def mde_or(n_case, n_ctrl, p_exp_ctrl, alpha=0.05, power=0.8, sims=2000, seed=0):
    """Smallest OR in [1, 1000] reaching `power` for a two-sided Fisher exact test of an
    unstratified case-control table (doubling rule, as in exact_conditional_or).

    Case exposure ~ Bin(n_case, psi p0 / (1 - p0 + psi p0)), control ~ Bin(n_ctrl, p0).
    One set of uniforms serves every psi (common random numbers), so the simulated
    power curve is smooth enough for bisection on log psi. inf if 1000 is not enough.
    """
    rng = np.random.default_rng(seed)
    p0, N = p_exp_ctrl, n_case + n_ctrl
    u = rng.random(sims)
    x0 = rng.binomial(n_ctrl, p0, sims)

    def pw(lpsi):
        psi = np.exp(lpsi)
        x1 = stats.binom.ppf(u, n_case, psi * p0 / (1 - p0 + psi * p0))
        m1 = x1 + x0
        tail = np.minimum(stats.hypergeom.cdf(x1, N, n_case, m1),
                          stats.hypergeom.sf(x1 - 1, N, n_case, m1))
        return np.mean(np.minimum(1.0, 2 * tail) <= alpha)

    lo, hi = 0.0, np.log(1000.0)
    if pw(hi) < power:
        return np.inf
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pw(mid) >= power else (mid, hi)
    return float(np.exp(hi))


# ---------------------------------------------------------------- spec §8 helpers

def cell_estimate(values, levels, B=10000, seed=0, alpha=0.05):
    """One cell of episode values (spec §8): est, lo, hi, n, method.
    Pooled over levels: level-stratified episode bootstrap. One level (15
    boards): Clopper-Pearson for binary values, a t interval otherwise."""
    v = [x for x in values if x is not None and not np.isnan(x)]
    lv = [l for x, l in zip(values, levels) if x is not None and not np.isnan(x)]
    if not v:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n=0, method=None)
    if len(set(lv)) > 1:
        est, lo, hi = strat_bootstrap(v, lv, np.mean, B=B, seed=seed, alpha=alpha)
        return dict(est=est, lo=lo, hi=hi, n=len(v), method="level-stratified bootstrap")
    if set(v) <= {0, 1}:
        lo, hi = clopper_pearson(int(sum(v)), len(v), alpha)
        return dict(est=float(np.mean(v)), lo=lo, hi=hi, n=len(v), method="Clopper-Pearson")
    est, lo, hi = t_interval(v, alpha)
    return dict(est=est, lo=lo, hi=hi, n=len(v), method="t")


def paired_contrast(rows, value, key, a, b, unit=("model", "suite", "level", "board"), seed=0):
    """Board-paired sign-flip contrast a - b of `value` between rows whose
    `key` is a or b (spec §8). Rows pair on `unit` (several rows of one unit
    are averaged); rows with value None are missing. Returns the signflip dict plus pairs and missing-pair counts."""
    def side(arm):
        # rows sharing a unit (e.g. two models when `unit` omits the model) are averaged
        g = {}
        for r in rows:
            if r[key] == arm and r[value] is not None:
                g.setdefault(tuple(r[u] for u in unit), []).append(r[value])
        return {k: float(np.mean(v)) for k, v in g.items()}

    A, Bv = side(a), side(b)
    both = sorted(set(A) & set(Bv))
    d = [A[k] - Bv[k] for k in both]
    out = signflip(d, seed=seed)
    out.update(contrast=f"{key}: {a} - {b}", pairs=len(both), missing=len(set(A) ^ set(Bv)))
    return out
