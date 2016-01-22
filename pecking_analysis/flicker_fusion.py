import os
import copy
import numpy as np
from matplotlib import pyplot as plt
import random
import statsmodels.api as sm
import pandas as pd

def get_filename_frequency(filename):

    return int(os.path.basename(filename).split("_")[1])

def divide_frequency_range(min_val, max_val, n, log=False, round_values=True):

    if round_values:
        min_val = round(min_val, -1)
        max_val = round(max_val, -1)
    if log:
        values = 10 ** np.logspace(np.log10(min_val), np.log10(max_val), n)
    else:
        values = np.linspace(min_val, max_val, n)

    if round_values:
        values = [round(val, -1) for val in values]

    return values

def get_response_by_frequency(blocks, log=True, fracs=None, scaled=True, filename="", nbootstraps=10, method="newton"):

    if fracs is None:
        fracs = [0.2, 0.35, 0.5, 0.65, 0.8]

    data = pd.DataFrame()
    bins = np.array([30, 50, 80, 120, 170, 230, 300, 380, 470, 570, 700, 900, 1000])
    bin_freqs = lambda freq: bins[np.nonzero(np.ceil(float(freq) / bins) == 1)[0][0]]
    for ii, blk in enumerate(blocks):
        freq_df = blk.data.copy()
        freq_df["Frequency"] = freq_df["Stimulus"].apply(get_filename_frequency)
        freq_df["FreqGroup"] = freq_df["Frequency"].apply(bin_freqs)
        freq_df["Response"] = blk.data["Response"]
        data = data.append(freq_df[["Frequency", "Response", "Class", "FreqGroup"]])

    grouped = data.groupby("FreqGroup")
    # grouped = data.groupby("Frequency")
    m = grouped.mean()["Response"]
    freqs = m.index.values.astype(float)
    m = m.values

    reward_rate, unreward_rate = get_nonprobe_interruption_rates(data)
    models = [model_logistic(data, log=log, scaled=scaled, method=method, disp=False) for ii in xrange(nbootstraps)]
    est_freqs = np.arange(10, 1000, 10)
    r_ests = model_predict(models, est_freqs, log=log)

    frac_rates = list()
    for frac in fracs:
        r = get_frequency_probability(models, frac, log=log, min_val=reward_rate, max_val=unreward_rate)
        frac_rates.append(r)
    print(", ".join(["p = %0.2f) %4.2f (Hz)" % (f, fr) for f, fr in zip(fracs, frac_rates)]))

    fig = plt.figure(figsize=(6, 6), facecolor="white", edgecolor="white")
    ax = fig.add_subplot(111)
    ax.plot(freqs, m, color="b", linewidth=2)
    ax.plot(est_freqs, r_ests, color="r", linewidth=2)

    ax.plot(freqs, reward_rate * np.ones_like(freqs), linewidth=0.5, color=[0.3, 0.3, 0.3], linestyle="--")
    ax.plot(freqs, unreward_rate * np.ones_like(freqs), linewidth=0.5, color=[0.3, 0.3, 0.3], linestyle="--")

    ymin, ymax = ax.get_ylim()
    for frac, fr in zip(fracs, frac_rates):
        ax.plot(freqs, (reward_rate + (unreward_rate -reward_rate) * frac) * np.ones_like(freqs), linewidth=0.5, color=[0.7, 0.7, 0.7],
                linestyle="--")
        ax.plot([fr, fr], [ymin, ymax], linewidth=0.5, color=[0.7, 0.7, 0.7],
                linestyle="--")

    ax.set_title(blocks[0].name)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.set_ylabel("Fraction Interruption")
    ax.set_xlabel("Click Frequency (Hz)")

    if len(filename):
        print('Saving figure to %s' % filename)
        fig.savefig(filename, facecolor="white", edgecolor="white", dpi=450)

    return models

def estimate_center_frequency(blocks, log=True, scaled=True, plot=True, filename="", nbootstraps=5):

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


    if plot:
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

def get_probe_blocks(blocks):

    blocks = [blk for blk in blocks if "Probe" in blk.data["Class"].unique()]

    return blocks


def bootstrap_center_frequency(blocks, log=True, scaled=True, nbootstraps=100, nsamples=100):
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
    data["Frequency"] = data["Stimulus"].apply(get_filename_frequency)
    data = data[["Frequency", "Response", "Class"]]

    ### Change below here
    cfs = list()
    models = list()
    nfailed = 0
    for bootstrap in range(nbootstraps):

        fit_data = sample_evenly(data, nsamples=nsamples)
        res = model_logistic(fit_data, log=log, scaled=scaled, disp=False)
        print("Bootstrap %d of %d: model p-value was %1.2e" %(bootstrap, nbootstraps, res.llr_pvalue))
        ri, ui = get_nonprobe_interruption_rates(fit_data)
        try:
            cfs.append(get_frequency_probability(res, 0.5, log=log, min_val=ri, max_val=ui))
        except ValueError:
            nfailed += 1
            continue

        models.append(res)

    print("%d models were not significant for %d samples from bird %s" % (nfailed, nsamples, blocks[0].name))

    return cfs, models

def sample_evenly(df, nsamples=100, groupby="Class"):

    grouped = df.groupby(groupby)
    samples_per = int(nsamples / len(grouped))
    output = pd.concat([g.sample(samples_per) for name, g in grouped])

    return output

def concatenate_data(blocks):

    return pd.concat([blk.data.copy() for blk in blocks], ignore_index=True)

def extract_frequencies(data):

    return data["Stimulus"].apply(get_filename_frequency)

def aggregate_models(models, log=True, p_thresh=0.05):

    if log:
        varname = "LogFreq"
    else:
        varname = "Frequency"

    if not isinstance(models, list):
        models = [models]

    result = models[0]
    if isinstance(result.model, ScaledLogit):
        min_vals, max_vals = zip(*[(res.model.min_val, res.model.max_val) for res in models])
        result.model.min_val = np.mean(min_vals)
        result.model.max_val = np.mean(max_vals)

    try:
        slopes, intercepts = zip(*[(res.params[varname], res.params["Intercept"]) for res in models if res.llr_pvalue
                                   <= p_thresh])
    except ValueError: # No models were significantly better than the null
        raise ValueError("0 of %d models were significant" % len(models))

    result.params["Intercept"] = np.mean(intercepts)
    result.params[varname] = np.mean(slopes)

    return result

def model_logistic(data, log=True, scaled=False, restrict_nonprobe=True, method="bfgs", disp=True):

    data = data.copy()
    if log:
        data["LogFreq"] = data["Frequency"].apply(np.log10)
        freq_name = "LogFreq"
    else:
        freq_name = "Frequency"
    data["Intercept"] = 1.0

    if restrict_nonprobe:
        grouped = data.groupby("Class")
        udf = grouped.get_group("Unrewarded")
        rdf = grouped.get_group("Rewarded")
        pdf = grouped.get_group("Probe")
        mean_count = int(np.ceil(np.mean(pdf.groupby("Frequency")["Response"].count())))

        udf = pd.concat([udf.ix[random.sample(val, mean_count)] if len(val) > mean_count else udf.ix[val] for val in
                         udf.groupby("Frequency").groups.values()])
        rdf = pd.concat([rdf.ix[random.sample(val, mean_count)] if len(val) > mean_count else rdf.ix[val] for val in
                         rdf.groupby("Frequency").groups.values()])
        data = pd.concat([udf, rdf, pdf]).sort_index()

    min_val, max_val = get_nonprobe_interruption_rates(data)

    if scaled:
        logit = ScaledLogit(data["Response"], data[[freq_name, "Intercept"]], min_val=min_val, max_val=max_val)
    else:
        logit = sm.Logit(data["Response"], data[[freq_name, "Intercept"]])

    return logit.fit(method=method, disp=disp)

def get_nonprobe_interruption_rates(data):

    g = data.groupby("Class")
    r = g.get_group("Rewarded")["Response"]
    u = g.get_group("Unrewarded")["Response"]

    return float(r.sum()) / r.count(), float(u.sum()) / u.count()

def model_predict(models, frequencies, log=True):

    result = aggregate_models(models, log=log)

    if log:
        estimates = result.predict(np.vstack([np.log10(frequencies), np.ones_like(frequencies)]).T)
    else:
        estimates = result.predict(np.vstack([frequencies, np.ones_like(frequencies)]).T)

    return estimates

def get_frequency_probability(models, prob, log=True, min_val=0, max_val=1):

    prob = min_val + (max_val - min_val) * prob

    res = aggregate_models(models, log=log)
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
        return (1 / (1 + np.exp(-X)))

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

def filter_blocks(blocks):

    for blk in blocks:
        data = blk.data.copy()
        frequency = data["Stimulus"].apply(get_filename_frequency)
        inds = ((frequency >= 900) | (frequency <= 30)) & (data["Class"] == "Probe")
        data = data[~inds]

        blk.data = data.copy()

    return blocks


if __name__ == "__main__":

    import sys
    import os
    from pecking_analysis import importer

    csv_files = list()
    for arg in sys.argv[1:]:
        filename = os.path.abspath(os.path.expanduser(arg))
        if not os.path.exists(filename):
            IOError("File %s does not exist!" % filename)
        csv_files.append(filename)

    csv_importer = importer.PythonCSV()
    blocks = csv_importer.parse(csv_files)
    get_response_by_frequency(blocks)
