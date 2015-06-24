import os
import numpy as np
from matplotlib import pyplot as plt
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

def get_response_by_frequency(blocks, log=True, fracs=None):

    if fracs is None:
        fracs = [0.2, 0.5, 0.8]

    data = pd.DataFrame()
    for ii, blk in enumerate(blocks):
        freq_df = blk.data.copy()
        freq_df["Frequency"] = blk.data["Stimulus"].apply(get_filename_frequency)
        freq_df["Response"] = blk.data["Response"]
        data = data.append(freq_df[["Frequency", "Response"]])

    grouped = data.groupby("Frequency")
    # c = grouped.count()["Response"].values
    m = grouped.mean()["Response"]
    freqs = m.index.values.astype(float)
    m = m.values

    res = model_logistic(data, log=log)
    if log:
        r_ests = res.predict(np.vstack([np.log10(freqs), np.ones_like(freqs)]).T)
    else:
        r_ests = res.predict(np.vstack([freqs, np.ones_like(freqs)]).T)

    reward_rate, unreward_rate = get_nonprobe_interruption_rates(blocks)
    frac_rates = list()
    for frac in fracs:
        r = get_frequency_probability(res, frac, log=log, min_val=reward_rate, max_val=unreward_rate)
        frac_rates.append(r)
    print(", ".join(["p = %2.1f) %4.2f (Hz)" % (f, fr) for f, fr in zip(fracs, frac_rates)]))

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

def model_logistic(data, log=True):

    if log:
        data["LogFreq"] = data["Frequency"].apply(np.log10)
        freq_name = "LogFreq"
    else:
        freq_name = "Frequency"
    data["Intercept"] = 1.0

    logit = sm.Logit(data["Response"], data[[freq_name, "Intercept"]])
    res = logit.fit()

    return res

def get_nonprobe_interruption_rates(blocks):

    ri = 0
    rc = 0
    ui = 0
    uc = 0

    for blk in blocks:
        g = blk.data.groupby("Class")
        r = g.get_group("Rewarded")["Response"]
        u = g.get_group("Unrewarded")["Response"]

        ri += r.sum()
        rc += r.count()
        ui += u.sum()
        uc += u.count()

    return float(ri) / rc, float(ui) / uc

def get_frequency_probability(res, prob, log=True, min_val=0, max_val=1):

    prob = min_val + (max_val - min_val) * prob

    if log:
        return 10 ** ((np.log(prob) - np.log(1 - prob) - res.params["Intercept"]) / res.params["LogFreq"])
    else:
        return (np.log(prob) - np.log(1 - prob) - res.params["Intercept"]) / res.params["Frequency"]

def fit_logistic_model(blk, log=True, niters=100, eta=0.01):

    freqs = blk.data["Stimulus"].apply(get_filename_frequency).values
    if log:
        freqs = np.log10(freqs)
    y = blk.data["Response"].values
    g = blk.data.groupby("Class")
    b0 = g.get_group("Rewarded")["Response"].mean()
    b1 = g.get_group("Unrewarded")["Response"].mean() - b0

    f0 = np.random.uniform(10, 1000)
    w = np.random.normal()
    if log:
        f0 = np.log10(f0)

    def logistic(x):

        return 1.0 / (1 + np.exp(x))

    def prob(f, w, f0):

        return b0 + b1 * logistic(w * (f - f0))

    def likelihood(y, f, w, f0):
        p = prob(f, w, f0)

        return np.sum(np.log((y * p) + (1 - y) * (1 - p)))

    l0 = likelihood(y, freqs, w, f0)
    print("w=%4.3f, f0=%4.2f, ll=%4.2f" % (w, f0, l0))
    ls = list()

    for iter in xrange(niters):

        p = prob(freqs, w, f0)
        l = logistic(w * (freqs - f0))
        l = l * (1 - l)
        tmp = b1 / (y * p + (1 - y) * (1 - p))
        df0 = np.sum(tmp * l * (-w))
        dw = np.sum(tmp * l * (freqs - f0))

        w = w + eta * dw
        f0 = f0 + eta * df0

        ls.append(likelihood(y, freqs, w, f0))
        print("w=%4.3f, f0=%4.2f, ll=%4.2f" % (w, f0, ls[-1]))

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

