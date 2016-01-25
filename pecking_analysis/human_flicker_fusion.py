import numpy as np
import pandas as pd
from scipy.io import wavfile
import statsmodels.discrete.discrete_model import Logit


def generate_clicks(frequencies, duration=6, sample_rate=44100,
                    click_length=.001):

    sound_length = int(duration * sample_rate)
    click_length = int(click_length * sample_rate)
    click = np.random.normal(click_length)

    sounds = list()
    for ii, freq in enumerate(frequencies):
        interval = int(duration / float(freq))
        onsets = range(0, sound_length, interval)
        sound = np.zeros(duration)
        for onset in onsets:
            sound[onset:onset + click_length] += click
        sounds.append(sound)

    return sounds


def sample_evenly(df, nsamples=100, groupby="Class"):
    """ Samples evenly nsamples combined points from all groups of groupby
    :param df: pandas dataframe
    :param nsamples: total number of samples from all groups
    :param groupby: column of the dataframe whose groups we want to sample

    returns a pandas dataframe with only the sampled rows
    """

    grouped = df.groupby(groupby)
    samples_per = int(nsamples / len(grouped))
    output = pd.concat([g.sample(samples_per) for name, g in grouped])

    return output


def model_logistic(data, log=True, scaled=False, restrict_nonprobe=True, sampler=sample_evenly, method="bfgs", disp=True):
    """ Compute a logistic or scaled logistic model on the data
    """

    data = data.copy()
    if log:
        data["LogFreq"] = data["Frequency"].apply(np.log10)
        freq_name = "LogFreq"
    else:
        freq_name = "Frequency"
    data["Intercept"] = 1.0

    # Sample the non-probe stimuli so that they don't get too much emphasis
    if restrict_nonprobe:
        data = sampler(data, len(data[data["Class"] == "Probe"]) * 3)

    min_val, max_val = get_nonprobe_interruption_rates(data)

    if scaled:
        logit = ScaledLogit(data["Response"], data[[freq_name, "Intercept"]], min_val=min_val, max_val=max_val)
    else:
        logit = sm.Logit(data["Response"], data[[freq_name, "Intercept"]])

    return logit.fit(method=method, disp=disp)


def get_nonprobe_interruption_rates(data):
    """ Computes the rewarded class and unrewarded class interruption rates
    """

    g = data.groupby("Class")
    r = g.get_group("Rewarded")["Response"]
    u = g.get_group("Unrewarded")["Response"]

    return float(r.sum()) / r.count(), float(u.sum()) / u.count()


class ScaledLogit(Logit):

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


def wavwrite(sound, fs, filename, overwrite=False):

    if os.path.exists(filename) and not overwrite:
        raise IOError("File exists. To save set overwrite=True")

    wav = wavfile.write(filename, fs, sound)


def get_filename_frequency(filename):
    """ Extracts the frequency value of a stimulus from its filename
    """

    ss = os.path.basename(filename).split("_")
    try:
        return int(ss[1])
    except:
        return None


class Block(object):

    def __init__(self,
                 name=None,
                 date=None,
                 start=None,
                 data=None,
                 store=None,
                 **kwargs):
        """
        Creates a Block object that stores data about a single chunk of trials
        for the pecking test
        :param name: The bird's name
        :param date: The date of the block - A datetime.date
        :param start: A start time of the block - A datetime.time
        :param filename: The CSV filename where the data came from
        :param data: The imported pandas DataFrame
        :param store: An HDF5Store instance.
        :param kwargs: Any additional keyword arguments will be added as
        annotations
        """

        self.name = name
        self.date = date
        self.start = start
        self.data = data
        self.store = store

        self.annotations = dict()
        self.annotate(**kwargs)

    def annotate(self, **annotations):
        """
        Add an annotation to the block
        :param annotations:
        :return:
        """

        self.annotations.update(annotations)
        if self.store is not None:
            return self.store.annotate_block(self, **self.annotations)

        return True


def get_trial_data(trials, response_name="response", session=1, max_wait=10):

    # Data is stored as a dictionary of 2D arrays
    # http://www.psychopy.org/general/dataOutputs.html

    response = trials.data[response_name + ".keys"] is not None
    condition = trials.trialList[trials.thisIndex]
    className = condition["type"]
    stimFilename = os.path.basename(condition["shape"])
    if className == "nonreward" and response is True:
        correct = True
        reward = False
    elif className == "reward" and response is False:
        correct = True
        reward = True
    else:
        correct = False
        reward = False

    if response is True:
        rt = trials.data[response_name + ".rt"]
    else:
        rt = None

    return [session,
            trials.thisN,
            stimFilename,
            className,
            response,
            correct,
            rt,
            reward,
            max_wait]


def get_response_by_frequency(block, log=True, fracs=None, scaled=True, nbootstraps=10, method="newton"):
    """ Computes multiple models of the concatenated data from blocks and optionally plots the fit
    """

    # Extract and concatenate data
    data = blk.data.copy()
    if "Frequency" not in data.columns:
        data["Frequency"] = data["Stimulus"].append(get_filename_frequency)
    data = data[["Response", "Frequency", "Class"]]

    # Estimate models
    reward_rate, unreward_rate = get_nonprobe_interruption_rates(data)
    models = [model_logistic(data, log=log, scaled=scaled, method=method, disp=False) for ii in range(nbootstraps)]

    # Compute frequency at different points on the logistic
    if fracs is None:
        fracs = [0.2, 0.35, 0.5, 0.65, 0.8]
    frac_rates = list()
    for frac in fracs:
        r = get_frequency_probability(models, frac, log=log, min_val=reward_rate, max_val=unreward_rate)
        frac_rates.append(r)

    return models, frac_rates


if False:
    # Begin Experiment
    import pandas as pd
    from pecking_analysis.human_flicker_fusion import *
    from pecking_analysis.peck_data import peck_data

    block = Block(name=expInfo["participant"])
    columnNames = ["Session",
                   "Trial",
                   "Stimulus",
                   "Class",
                   "Response",
                   "Correct",
                   "RT",
                   "Reward",
                   "Max Wait"]

    # Shaping routine
    # Start experiment
    shapingModelTrials = 10
    block_data = dict()

    # End Routine 1
    # Every XX trials compute the performance of the subject
    # If performance is below some threshold (0.05) then end routine
    # End routine with "trials.finished = True"

    trialData = get_trial_data(trials, session=1,
                               response_name="shape_response")
    for key, val in zip(columnNames, trialData):
        block_data.setdefault(key, list()).append(val)

    if trials.thisN % shapingModelTrials == 0:
        block.data = pd.DataFrame(block_data)
        print block.data
        performance = peck_data(block)
        if performance is not None:
            if performance["Stats", "P-Value"] < 0.05:
                trials.finished = True

    # Probe routine
    # Start experiment
    probeModelTrials = 10
    probeConverge = 3
    minFreq = 10
    maxFreq = 100
    currentFrequencies = list()
    prevLLRPerProbe = None
    nSinceBest = 0
    block_data = dict()

    # End Routine 1
    # Every XX trials compute the model of the subject's performance.
    # Overwrite the probe stimuli
    # Check convergence of model over last XX estimates

    # Find probe condition number
    if trials.thisN == 0:
        probeNo = [ii for ii, dd in enumerate(trials.trialList) if dd["type"] == "probe"][0]

    nProbes = len(trials.data["response.keys"][probeNo])
    trialData = get_trial_data(trials, session=2)
    for key, val in zip(columnNames, trialData):
        block_data.setdefault(key, list()).append(val)

    if (nProbes % probeModelTrials == 0) and (nProbes > probeModelTrials):
        block.data = pd.DataFrame(block_data)
        models, currentFrequencies = get_response_by_frequency(block)
        llrPerProbe = [mm.llr for mm in models]
        if prevLLRPerProbe is not None:
            tstat, pvalue = ttest_ind(prevLLRPerProbe, llrPerProbe)
            # tstat < 0 should mean that llrPerProbe > prevLLRPerProbe
            if (pvalue < 0.05) and (tstat < 0):
                prevLLRPerProbe = llrPerProbe
                nSinceBest = 0
            else:
                nSinceBest += 1

        if nSinceBest >= probeConverge:
            trials.finished = True

        # Create stimuli at those frequencies
        sounds = generate_clicks([freq for freq in currentFrequencies if minFreq <= freq <= maxFreq],
                                 sample_rate=44100)
        for sound in sounds:
            wavwrite(sound, 44100, filename)
