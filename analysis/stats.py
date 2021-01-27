from functools import partial

import multiprocessing
import numpy as np
from multiprocessing import Pool
from scipy.stats import hypergeom
from scipy.optimize import curve_fit


def false_discovery(pvalues, alpha=0.05):
    """Benjamini-Hochberg procedure for controlling false discovery rate
    """
    pvalues = np.array(pvalues)
    sorter = np.argsort(pvalues)
    n = len(pvalues)
    sig = np.zeros(n).astype(bool)
    thresholds = alpha * (np.arange(n) + 1) / n

    for i, (pvalue, threshold) in enumerate(zip(pvalues[sorter], thresholds)):
        if pvalue <= threshold:
            sig[sorter[:i + 1]] = True

    return sig


def _odds_ratio(table, zero_correction=True):
    """Computes odds ratio from 2x2 contingency table
    [[a, b],
     [c, d]]
    Uses Haldane-Anscombe correction (substitutes 0.5 for 0 values of
    b or c) if zero_correction is set to True.
    """
    ((a, b), (c, d)) = table + zero_correction * 0.5
    se = np.sqrt(np.sum([
        (1/a) + (1/b) + (1/c) + (1/d)
    ]))
    return (a * d) / (b * c), se


def fisher_exact(table, side="two.sided", zero_correction=True):
    """Computes fisher exact odds ratio.
    Output is almost exactly the same as scipy.stats.fisher_exact but here allows for
    using Haldane–Anscombe correction (substitutes 0.5 for 0 values in the table, whereas
    the scipy.stats version and R version fisher.test use integers only).
    For 95% confidence interval, uses confidence intervals computed by R function fisher.test
    """
    if side not in ("greater", "less", "two.sided"):
        raise ValueError("side parameter must be one of 'greater', 'less', or 'two.sided'")

    # Compute the p value
    # For all possible contingency tables with the observed marginals, compute the hypergeom
    # pmf of that table. Sum the p of all tables with p less than or equal to the hypergeom
    # probability of the observed table.
    N = np.sum(table)
    K = np.sum(table[:, 0])
    n = np.sum(table[0])

    odds_ratio, se = _odds_ratio(table, zero_correction=zero_correction)

    a_min = np.max([0, table[0][0] - table[1][1]])
    a_max = np.min([K, n])

    p_observed = hypergeom(N, K, n).pmf(table[0][0])
    p_value = 0.0
    for a in np.arange(a_min, a_max + 1):
        possible_table = np.array([
            [a, n - a],
            [K - a, N - n - K + a]
        ])
        p = hypergeom(N, K, n).pmf(a)

        if side == "greater":
            if _odds_ratio(possible_table)[0] >= odds_ratio:
                p_value += p
        elif side == "less":
            if _odds_ratio(possible_table)[0] <= odds_ratio:
                p_value += p
        elif side == "two.sided":
            if p <= p_observed:
                p_value += p

    if side == "greater":
        interval95 = [np.exp(np.log(odds_ratio) - (1.645 * se)), np.inf]
    elif side == "less":
        interval95 = [0, np.exp(np.log(odds_ratio) + (1.645 * se))]
    elif side == "two.sided":
        interval95 = [
                np.exp(np.log(odds_ratio) - (1.96 * se)),
                np.exp(np.log(odds_ratio) + (1.96 * se))
        ]

    return odds_ratio, np.array(interval95), p_value


def jackknife(samples, estimator, parallel=False, **kwargs):
    """Compute standard error of statistic on given samples
    samples: numpy array of sampled values
    estimator: function that takes numpy array and estimates some statistic (e.g. np.mean)
    Returns estimate of standard error of estimator
    """
    jk_n = []
    n = len(samples)

    # Compute the value of estimator over all n samples
    jk_all = estimator(np.array(samples), **kwargs)

    # Compute value of estimator for each combination of n-1 samples
    map_data = [np.concatenate([samples[:i], samples[i+1:]]) for i in range(len(samples))]
    if parallel:
        cores = multiprocessing.cpu_count()
        with Pool(cores - 1) as p:
            jk_n = p.map(partial(estimator, **kwargs), map_data)
    else:
        jk_n = [partial(estimator, **kwargs)(s) for s in map_data]
    jk_n = np.array(jk_n)

    # TODO: Estimating psths with the psueodvalues method comes out really bad
    # I don't know why at the moment so just skip that...

    # Compute pseudo values for samples (in n -> inf limit)
    jk_pseudo_values = [(n * jk_all - (n - 1) * jk_n[i]) for i in range(n)]

    est_mean = np.mean(jk_pseudo_values)
    est_var = (1 / n) * np.var(jk_pseudo_values)
    est_sem = np.sqrt(est_var)

    return est_mean, est_sem


def get_odds_ratio_matrix(group1, group2, key):
    """Generate contingency matrix of an in group response and out of group response columns
    |         group1         |         group2         |
    |------------------------|------------------------|
    | #(group1[key] == True) | #(group2[key] == True) |
    | #(group1[key] != True) | #(group2[key] != True) |
    """
    if key is None:
        contingency_table = [
            [len(group1[group1 == True]),
            len(group2[group2 == True])],
            [len(group1[group1 == False]),
            len(group2[group2 == False])]
        ]
    else:
        contingency_table = [
            [len(group1[group1[key] == True]),
            len(group2[group2[key] == True])],
            [len(group1[group1[key] == False]),
            len(group2[group2[key] == False])]
        ]

    return np.array(contingency_table)


def compute_odds_ratio(
        group,
        versus,
        key=None,
        zero_correction=True,
        side="two.sided",
    ):
    """Compute odds ratio on an in group and out group
    group and versus are pandas DataFrame objects representing
    trials from two conditions. They each should have a boolean column
    named "Response" indicating behavioral response.
    """
    table = get_odds_ratio_matrix(group, versus, key=key)
    odds, interval, pvalue = fisher_exact(table, side=side)

    return odds, interval, pvalue


def linreg(x, y):
    """Perform a simple linear regression on x, y arrays
    Returns:
        popt: optimal values of the parameters (a, b)
        pcov: estimated covariance of the estimated values of popt
        fit_fn: best fit line function, with parameters popt already filled in
        r_squared: R squared value
        r_adj: adjusted R squared value
        sigma_ab: standard deviation of best fit values in popt (squart root of diagonal of cov)
    """
    def lin(x, a, b):
        return x * a + b

    popt, pcov = curve_fit(lin, x, y)
    sigma_ab = np.sqrt(np.diagonal(pcov))
    residuals = y - lin(x, *popt)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    n = len(x)
    k = 1
    r_adj = 1 - ((1 - r_squared) * (n-1) / (n-k-1))

    def fit_fn(x):
        return lin(x, *popt)

    return popt, pcov, fit_fn, r_squared, r_adj, sigma_ab


def bootstrap(func, *args, iters=10000):
    """Return bootstrapped standard error for func with 1+ args
    """
    bootstrap_estimates = []
    for _ in range(iters):
        sampled_args = []
        for arg in args:
            sampled_args.append(
                np.random.choice(arg, replace=True, size=len(arg))
            )
        bootstrap_estimates.append(func(*sampled_args))

    return np.std(bootstrap_estimates)
