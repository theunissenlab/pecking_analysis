import os
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

def get_response_by_frequency(blocks, log=True, fracs=None, scaled=False, filename=""):

    if fracs is None:
        fracs = [0.2, 0.35, 0.5, 0.65, 0.8]

    data = pd.DataFrame()
    for ii, blk in enumerate(blocks):
        freq_df = blk.data.copy()
        freq_df["Frequency"] = blk.data["Stimulus"].apply(get_filename_frequency)
        freq_df["Response"] = blk.data["Response"]
        data = data.append(freq_df[["Frequency", "Response", "Class"]])

    grouped = data.groupby("Frequency")
    m = grouped.mean()["Response"]
    freqs = m.index.values.astype(float)
    m = m.values

    reward_rate, unreward_rate = get_nonprobe_interruption_rates(data)
    # data = data.groupby("Class").get_group("Probe")
    res = model_logistic(data, log=log, min_val=reward_rate, max_val=unreward_rate, scaled=scaled)
    if log:
        r_ests = res.predict(np.vstack([np.log10(freqs), np.ones_like(freqs)]).T)
    else:
        r_ests = res.predict(np.vstack([freqs, np.ones_like(freqs)]).T)


    frac_rates = list()
    for frac in fracs:
        r = get_frequency_probability(res, frac, log=log, min_val=reward_rate, max_val=unreward_rate, scaled=scaled)
        frac_rates.append(r)
    print(", ".join(["p = %0.2f) %4.2f (Hz)" % (f, fr) for f, fr in zip(fracs, frac_rates)]))

    fig = plt.figure(figsize=(6, 6), facecolor="white", edgecolor="white")
    ax = fig.add_subplot(111)
    ax.plot(freqs, m, color="b", linewidth=2)
    ax.plot(freqs, r_ests, color="r", linewidth=2)

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

    return res

def estimate_center_frequency(blocks, log=True, scaled=False, plot=True, filename=""):

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
        res = model_logistic(fit_data, log=log, min_val=ri, max_val=ui, scaled=scaled)
        cfs.append(get_frequency_probability(res, 0.5, log=log, min_val=ri, max_val=ui, scaled=scaled))

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

def concatenate_data(blocks):

    return pd.concat([blk.data.copy() for blk in blocks], ignore_index=True)

def extract_frequencies(data):

    return data["Stimulus"].apply(get_filename_frequency)

def model_logistic(data, log=True, min_val=0, max_val=1, scaled=False, restrict_nonprobe=True):

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

    if scaled:
        logit = ScaledLogit(data["Response"], data[[freq_name, "Intercept"]], min_val=min_val, max_val=max_val)
        res = logit.fit(method='bfgs')
    else:
        logit = sm.Logit(data["Response"], data[[freq_name, "Intercept"]])
        res = logit.fit()

    return res

def get_nonprobe_interruption_rates(data):

    g = data.groupby("Class")
    r = g.get_group("Rewarded")["Response"]
    u = g.get_group("Unrewarded")["Response"]

    return float(r.sum()) / r.count(), float(u.sum()) / u.count()

def get_frequency_probability(res, prob, log=True, min_val=0, max_val=1, scaled=False):

    prob = min_val + (max_val - min_val) * prob

    if log:
        if scaled:
            return 10 ** ((np.log(prob - min_val) - np.log(max_val - prob) - res.params["Intercept"]) / res.params["LogFreq"])
        else:
            return 10 ** ((np.log(prob) - np.log(1 - prob) - res.params["Intercept"]) / res.params["LogFreq"])
    else:
        if scaled:
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
        b = (self.max_val - self.min_val)
        p = self.cdf(np.dot(X, params))
        l = self._logit(np.dot(X, params))

        return np.dot(b * (y - p) * l * (1 - l) / (p * (1 - p)), X)

    def hessian(self, params):

        y = self.endog
        X = self.exog
        b = self.max_val - self.min_val
        p = self.cdf(np.dot(X, params))
        l = self._logit(np.dot(X, params))
        d = p * (1 - p)
        d2 = l * (1 - l) * (((y - p) * (1 - 2 * l) - b * l * (1 - l)) * d - b * (y - p) * l * (1 - l) * (1 - 2 * p)) \
             / d ** 2

        return -np.dot(d2 * X.T, X)

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


def fit_logistic_model(data, log=True, min_val=0, max_val=1, niters=100, eta=0.01, tol=1e-5):

    freqs = data["Frequency"].values
    if log:
        freqs = np.log10(freqs)
    y = data["Response"].values
    g = data.groupby("Class")

    f0 = np.random.uniform(10, 1000)
    w = np.random.normal()
    if log:
        f0 = np.log10(f0)

    def logistic(x):

        return 1.0 / (1 + np.exp(x))

    def prob(f, w, f0):

        return min_val + (max_val - min_val) * logistic(-w * (f - f0))

    def likelihood(y, f, w, f0):
        p = prob(f, w, f0)

        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def compute_gradients(y, f, w, f0):

        p = prob(f, w, f0)
        l = logistic(-w * (f - f0))
        d = p * (1 - p)
        b = (max_val - min_val)
        tmp = (y - p) / d
        d1 = tmp * l * (1 - l)
        d2 = l * (1 - l) * (((y - p) * (1 - 2 * l) - b * l * (1 - l)) * d - b * (y - p) * l * (1 - l) * (1 - 2 * p)) \
             / d ** 2
        df0 = -w * b * np.sum(d1)
        d2f0 = -w**2 * b * np.sum(d2)
        dw = b * np.sum(d1 * (f - f0))
        d2w = b * np.sum(d2 * (f - f0) ** 2)

        return df0, d2f0, dw, d2w

    def empirical_gradients(y, f, w, f0, delta=1e-6):

        il = likelihood(y, f, w, f0)

        dw = (0.5 / delta) * ((likelihood(y, f, w + delta, f0) - il) - (likelihood(y, f, w - delta, f0) - il))
        df0 = (0.5 / delta) * ((likelihood(y, f, w, f0 + delta) - il) - (likelihood(y, f, w, f0 - delta) - il))

        return df0, dw


    l0 = likelihood(y, freqs, w, f0)
    print("w=%4.3f, f0=%4.2f, ll=%4.2f" % (w, f0, l0))
    ls = list()

    for iter in xrange(niters):
        # df0, dw = empirical_gradients(y, freqs, w, f0)
        df0, d2f0, dw, d2w = compute_gradients(y, freqs, w, f0)
        w = w + eta * dw
        # w = w - dw / d2w
        f0 = f0 + eta * df0
        # f0 = f0 - df0 / d2f0

        ls.append(likelihood(y, freqs, w, f0))
        print("w=%4.3f, f0=%4.2f, ll=%4.2f" % (w, f0, ls[-1]))
        if (iter > 1) and (np.abs(ls[-1] - ls[-2]) < tol):
            print("Regression has converged below tolerance level of %2.1e" % tol)
            break

    return w, f0





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

