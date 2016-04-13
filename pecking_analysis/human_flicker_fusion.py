import os
import time
import logging
import numpy as np
import pandas as pd
from scipy.io import wavfile
from statsmodels.discrete.discrete_model import Logit


def generate_clicks(frequencies, duration=6, sample_rate=44100,
                    click_length=.001):

    sound_length = int(duration * sample_rate)
    click_length = int(click_length * sample_rate)
    click = np.random.normal(click_length)
    logging.debug("Creating %d sounds of length %2.1f seconds" % (len(frequencies), duration))
    sounds = list()
    for ii, freq in enumerate(frequencies):
        logging.debug("Creating sound with click frequency %3.1f Hz" % freq)
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

    logging.info("Writing sound to filename %s" % filename)
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

    columnNames = ["Session",
                   "Trial",
                   "Stimulus",
                   "Class",
                   "Response",
                   "Correct",
                   "RT",
                   "Reward",
                   "Max Wait"]

    def __init__(self,
                 name=None,
                 trial=0,
                 reward=0,
                 correct=0,
                 **kwargs):
        """
        Creates a Block object that stores data about a single chunk of trials
        for the pecking test
        :param name: The bird's name
        :param kwargs: Any additional keyword arguments will be added as
        parameters
        """

        self.name = name
        self.trial = trial
        self.reward = reward
        self.correct = correct

        self.data = None
        self.datadict = dict()
        self.params = dict()
        self.annotate(**kwargs)

        logging.info("Created block %s" % str(self))

    def __str__(self):

        return "%s name: %s, trial: %d, reward: %d, correct: %d" % (self.__class__,
                                                                    self.name,
                                                                    self.trial,
                                                                    self.reward,
                                                                    self.correct)

    def annotate(self, **annotations):
        """
        Add an annotation to the block
        :param annotations:
        :return:
        """

        self.params.update(annotations)
        return True

    def set_data(self):

        self.data = pd.DataFrame(self.datadict)

    def check(self):

        return False

    def message(self):

        return "Trial = %d\nYour score = %d" % (self.trial,
                                                self.reward)

    def add_trial(self, trials, response, max_wait=3):

        logging.debug("Collecting trial %d" self.trial + 1)
        responded = (response.keys is not None)
        if responded is True:
            rt = response.rt
            if rt > max_wait:
                responded = False
        else:
            rt = None

        condition = trials.trialList[trials.thisIndex]
        className = condition["condition"]
        stimFilename = os.path.basename(condition["filename"])
        if className == "nonreward" and responded is True:
            correct = True
            reward = False
        elif className == "reward" and responded is False:
            correct = True
            reward = True
        else:
            correct = False
            reward = False

        trial_data = [0,
                      trials.thisN,
                      stimFilename,
                      className,
                      responded,
                      correct,
                      rt,
                      reward,
                      max_wait]

        self.trial += 1
        if reward:
            self.reward += 1
        if correct:
            self.correct += 1

        for key, val in zip(self.columns, trial_data):
            self.datadict.setdefault(key, list()).append(val)


class ShapeBlock(Block):

    def __init__(self, check_trials=10, pThreshold=.001, **kwargs):

        super(ShapeBlock, self).__init__(check_trials=check_trials,
                                         pThreshold=pThreshold,
                                         **kwargs)

    def check(self):

        if (self.trial > 0) and (self.trial % self.params["check_trials"] == 0):
            logging.info("Trial %d - Checking shaping performance" % self.trial)
            self.set_data()
            performance = peck_data(self, group1="reward", group2="nonreward")
            if performance is not None:
                if performance["Stats", "P-Value"] < self.params["pThreshold"]:
                    return True

        return False


class ProbeBlock(Block):

    nprobes = property(fget=lambda self: self.datadict["Class"].count("probe"))

    def __init__(self, frequencies=None, check_trials=5, check_converge=3,
                 pThreshold=.05, minFreq=10, maxFreq=100, **kwargs):

        super(ProbeBlock, self).__init__(check_trials=check trials,
                                         check_converge=check_converge,
                                         minFreq=minFreq,
                                         maxFreq=maxFreq,
                                         pThreshold=pThreshold,
                                         **kwargs)
        if frequencies is None:
            frequencies = [30, 45, 60, 75, 90]
        self.bestLLR = None
        self.nSinceBest = 0

    def message(self):

        return "Trial = %d\nYour score = %d\nProbes = %d" % (self.trial,
                                                             self.reward,
                                                             self.nprobes)

    def check(self):

        if (self.nprobes > 0) and (self.nprobes % self.params["check_trials"] == 0):
            logging.info("Probe %d - Checking model performance" % self.nprobes)
            self.set_data()
            self.data["Frequency"] = self.data["Stimulus"].apply(lambda ss: self.get_frequency(ss))
            start = time.time()
            models, currentFrequencies = get_response_by_frequency(self)
            logging.debug("Model computation took %3.2f seconds" % time.time() - start)
            llrPerProbe = [mm.llr / float(self.nprobes) for mm in models]
            if self.bestLLR is not None:
                logging.debug("Comparing llr distributions")
                tstat, pvalue = ttest_ind(self.bestLLR, llrPerProbe)
                logging.debug("t-stat: %3.2f, p-value: %3.2e" % (tstat, pvalue))
                # tstat < 0 should mean that llrPerProbe > prevLLRPerProbe
                if (pvalue < self.params["pThreshold"]) and (tstat < 0):
                    logging.info("Model is improving. Computing new sounds")
                    self.bestLLR = llrPerProbe
                    self.nSinceBest = 0
                    self.store_frequencies(currentFrequencies)
                    self.make_sounds()
                else:
                    self.nSinceBest += 1
            else:
                self.bestLLR = llrPerProbe
                self.nSinceBest = 0
                self.store_frequencies(currentFrequencies)
                self.make_sounds()

            if self.nSinceBest >= self.params["check_converge"]:
                return True

        return False

    def get_frequency(self, filename):

        frequency = get_filename_frequency(filename)
        if frequency is None:
            frequency = self.frequency_dict.get(filename, None)

        return frequency

    def store_frequencies(self, frequencies):

        self.frequencies = frequencies
        self.frequency_dict = dict([("sound_%d.wav", freq) for ii, freq in enumerate(self.frequencies)])

    def make_sounds(self):

        # Create stimuli at those frequencies
        frequencies = [min(max(freq, self.params["minFreq"]), self.params["minFreq"]) for freq in self.frequencies]
        sounds = generate_clicks(frequencies, sample_rate=44100)
        for sound in sounds:
            wavwrite(sound, 44100, filename)


def get_response_by_frequency(block, log=True, fracs=None, scaled=True, nbootstraps=10, method="newton"):
    """ Computes multiple models of the concatenated data from blocks and optionally plots the fit
    """

    # Extract and concatenate data
    print "Calculating model..."
    data = blk.data.copy()
    if "Frequency" not in data.columns:
        data["Frequency"] = data["Stimulus"].apply(get_filename_frequency)
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

    print "Finished"
    return models, frac_rates


if False:
    # Begin Experiment
    import pandas as pd
    from pecking_analysis.human_flicker_fusion import *
    from pecking_analysis.peck_data import peck_data

    # Shaping routine
    ## Start routine
    if shape_trials.thisN == 0:
        shape_block = ShapeBlock(name=expInfo["participant"])

    ## End Routine
    shape_block.add_trial(shape_response, shape_trials)
    msg = shape_block.message()
    finished = shape_block.check()
    if finished:
        shape_trials.finshed = True


    # Probe routine
    ## Start routine
    if probe_trials.thisN == 0:
        probe_block = ProbeBlock(name=expInfo["participant"], trial=shape_block.trial, reward=shape_block.reward, correct=shape_block.correct)

    ## End Routine
    probe_block.add_trial(probe_response, probe_trials)
    msg = probe_block.message()
    finished = probe_block.check()
    if finished:
        probe_trials.finished = True


    # Volume routine
    ## Start routine
    if volume_trials.thisN == 0:
        volume_block = ProbeBlock(frequencies=probe_block.frequencies, name=expInfo["participant"], trial=shape_block.trial + probe_block.trial, reward=shape_block.reward + probe_block.reward, correct=shape_block.correct + probe_block.correct)
    randVol = min(max(normal(0.5, 0.2), 0.0), 1.0)
    volume_sound.setVolume(randVol)

    ## End routine
    volume_block.add_trial(volume_response, volume_trials)
    msg = volume_block.message()
    finished = volume_block.check()
    if finished:
        volume_trials.finished = True
