#!/usr/bin/env python
import os
import random
import copy
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from matplotlib import pyplot as plt


# Dataframe helper functions
def get_filename_frequency(filename):
    """ Extracts the frequency value of a stimulus from its filename
    """

    ss = os.path.basename(filename).split("_")
    try:
        return int(ss[1])
    except:
        return None


def filter_blocks(blocks):

    for blk in blocks:
        data = blk.data.copy()
        frequency = data["Stimulus"].apply(get_filename_frequency)
        inds = ((frequency >= 900) | (frequency <= 30)) & (data["Class"] == "Probe")
        data = data[~inds]

        blk.data = data.copy()

    return blocks


def get_probe_blocks(blocks):
    """ Returns only the blocks that contain probe trials """

    blocks = [blk for blk in blocks if "Probe" in blk.data["Class"].unique()]

    return blocks

def add_daily_interruption(data):

    grouped = data.groupby("Session")
    for session_id, inds in grouped.groups.items():
        ri, ui = get_nonprobe_interruption_rates(data.loc[inds])
        data.ix[inds, "Min"] = ri
        data.ix[inds, "Max"] = ui

    return data

def concatenate_data(blocks):
    """ Concatenate the data from a list of blocks
    """
    data = list()
    sessions = dict()
    current = 0
    for blk in blocks:
        if blk.date not in sessions:
            sessions[blk.date] = current
            current += 1
        num = sessions[blk.date]
        tmp = blk.data.copy()
        tmp["Session"] = num
        data.append(tmp)
    return pd.concat(data, ignore_index=True)


def extract_frequencies(data):
    """ Get frequency values from a pandas dataframe with a Stimulus column
    """

    return data["Stimulus"].apply(get_filename_frequency)


def get_binomial_probability(blocks, nsamples=50, frequencies=None, nbins=10,
                             log_space=False):
    """ Get the probability of any interruption rate for all probe trials across frequencies """

    from matplotlib.mlab import griddata

    print("Filtering to probe blocks")
    blocks = get_probe_blocks(blocks)

    print("Found %d blocks" % len(blocks))
    if len(blocks) == 0:
        return

    # Remove any trials outside of the 10-1000 Hz range
    blocks = filter_blocks(blocks)
    df = concatenate_data(blocks)
    df["Frequency"] = extract_frequencies(df)
    if log_space is True:
        print("Using log frequencies")
        df["Frequency"] = df["Frequency"].apply(np.log10)
        if frequencies is not None:
            frequencies = np.log10(frequencies)
    # df = df[df["Class"] == "Probe"].sort("Frequency")
    df = df.sort("Frequency")

    if frequencies is None:
        frequencies = np.linspace(df["Frequency"].min(), df["Frequency"].max(), nbins)
    round_freq = lambda f: frequencies[np.argmin(np.abs(frequencies - f))]

    # data_by_freq = dict()
    # for ii in range(len(df) - nsamples):
    #     data = df.iloc[ii: ii + nsamples]
    #     freq = data["Frequency"].median()
    #     data_by_freq.setdefault(freq, list()).append(data["Response"].sum())
    #
    # frequencies = sorted(data_by_freq.keys())
    #
    # print("Computing probability on %d data points in %d frequencies" % (len(df) - nsamples, len(data_by_freq.keys())))
    #
    p_values = np.arange(0, 1, .01)
    grouped = df.groupby(map(round_freq, df["Frequency"]))
    frequencies = sorted(grouped.groups.keys())
    data = pd.DataFrame([], columns=p_values, index=frequencies)
    print("Computing posterior probability for binomial distribution with different interruption probabilities")
    # for freq in frequencies:
    for freq, g in grouped:
        prob = list()
        # count = data_by_freq[freq]
        count = g["Response"].sum()
        nsamples = g["Response"].count()
        prob = [scipy.stats.binom.pmf(count, nsamples, p) for p in p_values]
        data.loc[freq] = np.array(prob) / sum(prob)

    if log_space is True:
        print("Converting back to linear frequencies")
        data = data.reset_index()
        data["index"] = data["index"].apply(lambda x: 10 ** x)
        data = data.set_index("index")

    frequencies = data.index.values
    p_values = data.columns.values

    # Tile and interpolate
    data = data.values.astype(np.float)
    frequencies = np.tile(frequencies.reshape((-1, 1)), (1, data.shape[1]))
    p_values = np.tile(p_values.reshape((1, -1)), (data.shape[0], 1))

    extent = [np.min(frequencies), np.max(frequencies),
              np.min(p_values), np.max(p_values)]
    xs, ys = np.mgrid[extent[0]: extent[1]: complex(0, 5 * data.shape[0]),
                      extent[2]: extent[3]: complex(0, 5 * data.shape[1])]
    zs = griddata(frequencies.ravel(), p_values.ravel(), data.ravel(),
                  xs, ys, interp="linear")


    return xs, ys, zs


def get_probe_frequencies(blocks):
    """ Gets the frequencies for the probe stimuli across blocks """

    probe_blocks = get_probe_blocks(blocks)
    if len(probe_blocks) == 0:
        return

    freqs = pd.DataFrame([], index=["20%", "35%", "50%", "65%", "80%"])
    for blk in probe_blocks:
        df = blk.data.copy()
        df["Freq"] = df["Stimulus"].apply(get_filename_frequency)
        tmp = np.unique(df[df["Class"] == "Probe"]["Freq"].values)
        vals = np.nan * np.zeros(5)
        # If highest frequency is above 500, probably the top frequency was too high to be used
        # If the lowest is below 50, then probably the bottom frequency was too low to be used.
        if (len(tmp) < 5) and (tmp.min() <= 50):
            start = 1
        else:
            start = 0
        vals[start: start + len(tmp)] = tmp
        freqs[blk.date] = vals

    freqs = freqs.T.sort_index()

    return freqs


def get_nonprobe_interruption_rates(data):
    """ Computes the rewarded class and unrewarded class interruption rates
    """

    g = data.groupby("Class")
    r = g.get_group("Rewarded")["Response"]
    u = g.get_group("Unrewarded")["Response"]

    return float(r.sum()) / r.count(), float(u.sum()) / u.count()


# Sampling functions
def sample_evenly(df, nsamples=None, groupby="Class"):
    """ Samples evenly nsamples combined points from all groups of groupby
    :param df: pandas dataframe
    :param nsamples: total number of samples from all groups
    :param groupby: column of the dataframe whose groups we want to sample

    returns a pandas dataframe with only the sampled rows
    """

    grouped = df.groupby(groupby)
    if nsamples is None:
        nsamples = min(grouped.count().values)
    samples_per = int(nsamples / len(grouped))
    output = pd.concat([g.sample(samples_per) for name, g in grouped])

    return output


def sample_mean_probe_count(df):
    """ Samples each frequency of rewarded and unrewarded stimuli the average number of times that each probe frequency was played.
    """

    grouped = df.groupby("Class")
    udf = grouped.get_group("Unrewarded")
    rdf = grouped.get_group("Rewarded")
    pdf = grouped.get_group("Probe")
    mean_count = int(np.ceil(np.mean(pdf.groupby("Frequency")["Response"].count())))

    udf = pd.concat([udf.ix[random.sample(val, mean_count)] if len(val) > mean_count else udf.ix[val] for val in
                    udf.groupby("Frequency").groups.values()])
    rdf = pd.concat([rdf.ix[random.sample(val, mean_count)] if len(val) > mean_count else rdf.ix[val] for val in
                    rdf.groupby("Frequency").groups.values()])

    return pd.concat([udf, rdf, pdf]).sort_index()


def sample_nonprobe(df, nsamples=None):
    """Samples nsamples points from the non-probe classes, leaving the probe class intact.
    :param df: pandas dataframe
    :param nsamples: number of samples from each non-probe class. Defaults to number of probes

    returns a pandas dataframe with only the sampled rows
    """

    grouped = df.groupby("Class")
    if nsamples is None:
        nsamples = int(grouped.get_group("Probe").count())
    output = pd.concat([g.sample(nsamples) for name, g in grouped if name != "Probe"])
    output = pd.concat([output, grouped.get_group("Probe")])

    return output

def sample_only_probe(df, nsamples=None):
    """ Retuns nsamples of probe stimuli """
    df = df[df["Class"] == "Probe"].copy()
    if nsamples is not None:
        if nsamples < 1:
            nsamples = int(nsamples * len(df))
        df = df.sample(nsamples)

    return df



# Analyses
def get_response_by_frequency(blocks, log=True, fracs=None, scaled=True,
                              filename="", nbootstraps=10,
                              do_plot=True, maxiter=50,
                              sample_function=sample_mean_probe_count,
                              global_interruption=True,
                              daily_interruption=True,
                              binned=True,
                              show_fracs=False,
                              **kwargs):
    """
    Computes multiple models of the concatenated data from blocks and optionally plots the fit
    """

    bird_name = blocks[0].name

    # Extract and concatenate data
    print("Getting data from all blocks")
    data = concatenate_data(blocks)
    data["Frequency"] = extract_frequencies(data)
    data = add_daily_interruption(data)

    # Estimate models
    print("Estimating models")
    models = [model_logistic(data, log=log,
                             scaled=scaled, disp=False,
                             sample_function=sample_function, maxiter=maxiter,
                             global_interruption=global_interruption,
                             daily_interruption=daily_interruption,
                             **kwargs) for ii in range(nbootstraps)]
    model = aggregate_models(models, log=log)
    # model = models[0]
    # model.model.min_val = 0.0
    # model.model.max_val = 1.0

    # Compute frequency at different points on the logistic
    if fracs is None:
        fracs = [0.2, 0.35, 0.5, 0.65, 0.8]

    frac_rates = list()
    for frac in fracs:
        r = get_frequency_probability(model, frac,
                                      log=log,
                                      min_val=0.0,
                                      max_val=1.0)
        frac_rates.append(r)
    print(", ".join(["p = %0.2f) %4.2f (Hz)" % (f, fr) for f, fr in zip(fracs,
                                                                        frac_rates)]))
    frac_rates = [fr for fr in frac_rates if 0 <= fr <= 1000]

    if do_plot:
        # Get the empirical probability of interruption
        data = data[data["Class"] == "Probe"]
        # Normalize between unrewarded and rewarded interruption rates
        data["Response"] = (data["Response"] - data["Min"]) / (data["Max"] - data["Min"])
        # Bin the data along the frequency axis
        if binned:
            round_freq = lambda f: frac_rates[np.argmin(np.abs(frac_rates - f))]
            grouped = data.groupby(map(round_freq, data["Frequency"]))
        else:
            grouped = data.groupby("Frequency")

        # Get average scaled interruption rate
        m = grouped["Response"].mean()

        # Determine scatter plot size based on number of samples in each bin.
        sizes = list()
        max_size = 0
        for f, g in grouped:
            sizes.append(len(g))
            if "Probe" in g["Class"].unique():
                max_size = max(max_size, len(g))
        sizes = np.minimum(np.array(sizes) / float(max_size), 1.0)

        # Get values for scatter plot
        freqs = m.index.values.astype(float)
        m = m.values

        fig = plt.figure(figsize=(12, 12), facecolor="white", edgecolor="white")
        ax = fig.add_subplot(111)

        ymin = min(m.min(), 0.0)
        ymax = max(m.max(), 1.0)

        # Plot quantile lines
        if show_fracs:
            for frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
                fr = get_frequency_probability(model, frac, log=log,
                                               min_val=0.0, max_val=1.0)
                ax.plot([fr, fr], [ymin, ymax], linewidth=0.5,
                        color=[0.7, 0.7, 0.7], linestyle="--", zorder=1)
                # ax.arrow(fr, ymin - 0.1 * (ymax - ymin), 0, .1 * (ymax - ymin),
                #          head_width=.05 * (max(frac_rates) - min(frac_rates)),
                #          head_length=0.05 * (ymax - ymin),
                #          fc=[0.7, 0.7, 0.7], ec=[0.7, 0.7, 0.7])
                ax.plot(fr, frac, '.', markersize=20, color=[0.7, 0.7, 0.7], zorder=3)

        # Plot min and max interruption lines
        est_freqs = np.linspace(min(frac_rates), max(frac_rates), 100)
        ax.plot(est_freqs, np.zeros_like(est_freqs), linewidth=0.5, color=[0.3, 0.3, 0.3], linestyle="--", zorder=1)
        ax.plot(est_freqs, np.ones_like(est_freqs), linewidth=0.5, color=[0.3, 0.3, 0.3], linestyle="--", zorder=1)

        # Plot individual sigmoids
        r_ests = np.zeros((len(est_freqs), len(models)))
        for ii, res in enumerate(models):
            r_ests[:, ii] = model_predict(res, est_freqs, log=log,
                                          min_value=0.0, max_value=1.0)
        ax.plot(est_freqs, r_ests, color="gray", linewidth=1, zorder=1)

        # Plot averaged sigmoid
        r_avg_est = model_predict(model, est_freqs, log=log,
                                  min_value=0.0, max_value=1.0)
        ax.plot(est_freqs, r_avg_est, color="r", linewidth=2, zorder=2)

        # Plot scatter of empirical data
        ax.scatter(freqs, m, s=(50 + 400 * sizes), zorder=1)

        ax.set_title(bird_name)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_ticks([0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
        ax.yaxis.set_ticklabels(["Rewarded", "", "", "0.5", "", "", "Nonrewarded"])
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        ax.set_ylabel("Normalized Interruption Rate")
        ax.set_xlabel("Click Frequency (Hz)")

        ax.set_xlim((min(frac_rates), max(frac_rates)))
        ax.set_ylim((ymin, ymax))

        if len(filename):
            print('Saving figure to %s' % filename)
            fig.savefig(filename, facecolor="white", edgecolor="white", dpi=450)

    return models, frac_rates


def estimate_center_frequency(blocks, log=True, scaled=True, do_plot=True, filename="", nbootstraps=5):
    """ Estimate the center frequency at each probe trial in blocks.
    """
    data = concatenate_data(blocks)
    data["Frequency"] = data["Stimulus"].apply(get_filename_frequency)
    data = data[["Frequency", "Response", "Class"]]

    grouped = data.groupby("Class")
    probe_indices = grouped.get_group("Probe").index
    cfs = list()
    for ind in probe_indices:
        fit_data = data.loc[:ind, :]
        if len(fit_data) < 20:
            continue
        ri, ui = get_nonprobe_interruption_rates(fit_data)
        res = [model_logistic(fit_data, log=log, scaled=scaled, disp=False) for ii in xrange(nbootstraps)]
        try:
            cfs.append(get_frequency_probability(res, 0.5, log=log, min_val=ri, max_val=ui))
        except ValueError: # The model wasn't significant
            cfs.append(np.nan)


    if do_plot:
        fig = plt.figure(figsize=(6, 6), facecolor="white", edgecolor="white")
        ax = fig.add_subplot(111)
        ax.plot(cfs, color="b", linewidth=2)
        ax.set_title(blocks[0].name)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        ax.set_ylabel("Estimated Center Frequency (Hz)")
        ax.set_xlabel("# Probe trials")

        if len(filename):
            print('Saving figure to %s' % filename)
            fig.savefig(filename, facecolor="white", edgecolor="white", dpi=450)

    return cfs


def bootstrap_center_frequency(blocks, log=True, scaled=True, nbootstraps=100,
                               sample_function=None, **kwargs):
    """
    Calculates center frequency (with confidence). Will create models fitting a sigmoid predicting the birds response to a given frequency.
    Multiple models will used to calculate a confidence interval for the calculated center frequencies.
    nsamples = number data points used in each model
    cfs = stores center frequencies
    models = stores all information about each model created
    fit_data = evenly samples across all tests involving probes
    res = creates the model from samples chosen in fit_data
    cf = calculated the center frequency based on res
    """

    data = concatenate_data(blocks)
    data["Frequency"] = extract_frequencies(data)
    data = add_daily_interruption(data)

    ### Change below here
    cfs = list()
    models = list()
    nfailed = 0
    for bootstrap in range(nbootstraps):

        # fit_data = sample_evenly(data, nsamples=nsamples)
        res = model_logistic(data, log=log, scaled=scaled, disp=False,
                             sample_function=sample_function, **kwargs)
        print("Bootstrap %d of %d: model p-value was %1.2e" %(bootstrap, nbootstraps, res.llr_pvalue))
        # ri, ui = get_nonprobe_interruption_rates(fit_data)
        ri, ui = 0.0, 1.0
        if res.llr_pvalue <= 0.05:
            cfs.append(get_frequency_probability(res, 0.5, log=log, min_val=ri, max_val=ui))
        else:
            nfailed += 1
            continue

        models.append(res)

    # print("%d models were not significant for %d samples from bird %s" % (nfailed, nsamples, blocks[0].name))

    return cfs, models


# Model functions
def aggregate_models(models, log=True, p_thresh=0.05):

    if log:
        varname = "LogFreq"
    else:
        varname = "Frequency"

    if not isinstance(models, list):
        models = [models]

    result = models[0]
    # if isinstance(result.model, ScaledLogit):
    #     min_vals = np.vstack([res.model.min_val for res in models])
    #     max_vals = np.vstack([res.model.max_val for res in models])
    #     result.model.min_val = np.mean(min_vals)
    #     result.model.max_val = np.mean(max_vals)

    tmp = [(res.params[varname], res.params["Intercept"]) for res in models if res.llr_pvalue <= p_thresh]
    if len(tmp) > 0:
        slopes, intercepts = zip(*tmp)
    else:
        # print("0 of %d models were significant" % len(models))
        raise ValueError("0 of %d models were significant" % len(models))

    result.params["Intercept"] = np.mean(intercepts)
    result.params[varname] = np.mean(slopes)

    return result


def model_logistic(data, log=True, scaled=False, sample_function=None,
                   method="newton", disp=True, global_interruption=True,
                   daily_interruption=True, shuffle=False, **kwargs):
    """ Compute a logistic or scaled logistic model on the data
    """

    data = data.copy()
    if log:
        data["LogFreq"] = data["Frequency"].apply(np.log10)
        freq_name = "LogFreq"
    else:
        freq_name = "Frequency"
    data["Intercept"] = 1.0

    if global_interruption:
        if daily_interruption:
            data = add_daily_interruption(data)
        else:
            ri, ui = get_nonprobe_interruption_rates(data)
            data["Min"] = ri
            data["Max"] = ui
    # Sample the non-probe stimuli so that they don't get too much emphasis
    if sample_function is not None:
        data = sample_function(data)
        if not global_interruption:
            ri, ui = get_nonprobe_interruption_rates(data)
            data["Min"] = ri
            data["Max"] = ui

    if shuffle:
        data["Response"] = np.random.permutation(data["Response"].values)

    if scaled:
        min_val = data["Min"].values
        max_val = data["Max"].values
    else:
        min_val = 0.0
        max_val = 1.0

    logit = ScaledLogit(data["Response"], data[[freq_name, "Intercept"]],
                        min_val=min_val, max_val=max_val)

    return logit.fit(method=method, disp=disp, **kwargs)


def model_predict(res, frequencies, log=True, min_value=0.0, max_value=1.0):
    """ Predicts the probability of interruption of a list of frequencies using the model
    :param models: a list of models
    :param frequencies: a list of frequencies to predict on
    :param log: predict on log frequencies (default True)
    """

    if log:
        frequencies = np.log10(frequencies)
        varname = "LogFreq"
    else:
        varname = "Frequency"

    prob = res.model._logit(res.params[varname] * frequencies + res.params["Intercept"])

    return min_value + (max_value - min_value) * prob


def get_frequency_probability(res, prob, log=True, min_val=0, max_val=1):
    """ Get the frequency that corresponds to the specified probability. If min_val and max_val are specified, scale prob between them before computing the frequency.
    """

    prob = min_val + (max_val - min_val) * prob

    if log:
        if isinstance(res.model, ScaledLogit):
            return 10 ** ((np.log(prob - min_val) - np.log(max_val - prob) - res.params["Intercept"]) / res.params["LogFreq"])
        else:
            return 10 ** ((np.log(prob) - np.log(1 - prob) - res.params["Intercept"]) / res.params["LogFreq"])

    else:
        if isinstance(res.model, ScaledLogit):
            return (np.log(prob - min_val) - np.log(max_val - prob) - res.params["Intercept"]) / res.params["Frequency"]
        else:
            return (np.log(prob) - np.log(1 - prob) - res.params["Intercept"]) / res.params["Frequency"]


class ScaledLogit(sm.Logit):

    def __init__(self, endog, exog=None, min_val=0, max_val=1, **kwargs):

        super(ScaledLogit, self).__init__(endog, exog=exog, **kwargs)

        self.min_val = min_val
        self.max_val = max_val

    def _logit(self, X):

        X = np.asarray(X)
        return (1. / (1 + np.exp(-X)))

    def cdf(self, X):

        X = np.asarray(X)
        return self.min_val + (self.max_val - self.min_val) * self._logit(X)

    def pdf(self, X):

        X = np.asarray(X)
        return (self.max_val - self.min_val) * np.exp(-X) / (1 + np.exp(-X))**2

    def score(self, params):

        y = self.endog
        X = self.exog
        p = self.cdf(np.dot(X, params))

        return np.dot((y - p) * (p - self.min_val) * (self.max_val - p) / (p * (1 - p)), X)

    def hessian(self, params):

        y = self.endog
        X = self.exog
        b = self.max_val - self.min_val
        p = self.cdf(np.dot(X, params))
        l = self._logit(np.dot(X, params))
        g = -3 * p ** 2 + 2 * (self.max_val + self.min_val + 2 * y) * p - (self.max_val * self.min_val + y * (self.min_val + self.max_val))
        d = (p - self.min_val) * (self.max_val - p) * (g * p * (1 - p) - (y - p) * (2 * p - 1) * (p - self.min_val) *
                                                       (self.max_val - p)) / (p * (1 - p)) ** 2

        return -np.dot(d * X.T, X)

    def loglike(self, params):

        y = self.endog
        X = self.exog
        p = self.cdf(np.dot(X, params))

        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def loglikeobs(self, params):

        y = self.endog
        X = self.exog
        p = self.cdf(np.dot(X, params))

        return y * np.log(p) + (1 - y) * np.log(1 - p)


# Scripting functions
def probes(args):
    import os
    from pecking_analysis.objects import get_blocks

    args.filename = os.path.abspath(os.path.expanduser(args.filename))
    blocks = get_blocks(args.filename, birds=args.bird)
    blocks = [blk for blk in blocks if "Probe" in blk.data["Class"].unique()]
    blocks = filter_blocks(blocks)

    get_response_by_frequency(blocks, do_plot=args.plot)


def run_variance_calc(args):
    import os
    from pecking_analysis.objects import get_blocks

    args.filename = os.path.abspath(os.path.expanduser(args.filename))
    blocks = get_blocks(args.filename, birds=args.bird)
    blocks = [blk for blk in blocks if "Probe" in blk.data["Class"].unique()]
    blocks = filter_blocks(blocks)

    if len(blocks) == 0:
        print("No blocks found for %s with probe trials" % str(args.bird))
        return

    cfs = estimate_center_frequency(blocks, do_plot=args.plot)
    df = pd.DataFrame(cfs, columns=["Center Frequency"])
    df["Var"] = pd.rolling_var(df["Center Frequency"], window=10)
    df["Mean"] = pd.rolling_mean(df["Center Frequency"], window=10)
    df["Var Pct"] = 100 * df["Var"] / df["Mean"]

    num_trials = 25
    print("Displaying last %d probe trials" % num_trials)
    pd.options.display.max_rows = 2 * num_trials
    print(df.iloc[-num_trials:])


if __name__ == "__main__":
    import os
    import sys
    import argparse

    h5_file = os.path.abspath(os.path.expanduser("~/Dropbox/pecking_test/data/flicker_fusion.h5"))

    parser = argparse.ArgumentParser(description="Compute probe frequencies")
    subparsers = parser.add_subparsers(title="methods",
                                       description="Valid methods",
                                       help="Which flicker_fusion analysis method to run")

    # Add options for checking probe stimuli
    probe_parser = subparsers.add_parser("probe",
                                       description="Check the current estimate for probe stimuli frequencies")
    probe_parser.add_argument("bird", help="Name of bird to check. If not specified, checks all birds for the specified date")
    probe_parser.add_argument("-f", "--filename", dest="filename", help="Path to h5 file", default=h5_file)
    probe_parser.add_argument("--plot", dest="plot", help="Try to plot results", action="store_true")
    probe_parser.set_defaults(func=probes)

    # Add options for checking center frequency variance
    var_parser = subparsers.add_parser("var",
                                       description="Check the estimate of the center frequency, plus variance as percentage of mean")
    var_parser.add_argument("bird", help="Name of bird to check. If not specified, checks all birds for the specified date")
    var_parser.add_argument("-f", "--filename", dest="filename", help="Path to h5 file", default=h5_file)
    var_parser.add_argument("--plot", dest="plot", help="Try to plot results", action="store_true")
    var_parser.set_defaults(func=run_variance_calc)

    if len(sys.argv) == 1:
        parser.print_usage()
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)
