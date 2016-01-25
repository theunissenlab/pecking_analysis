import numpy as np
from scipy.io import wavfile


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


def wavwrite(sound, fs, filename, overwrite=False):

    if os.path.exists(filename) and not overwrite:
        raise IOError("File exists. To save set overwrite=True")

    wav = wavfile.write(filename, fs, sound)


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




if False:
    # Begin Experiment
    import pandas as pd
    from pecking_analysis.human_flicker_fusion import generate_clicks, Block, get_trial_data
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
        models, currentFrequencies = get_response_by_frequency([block],
                                                               do_plot=False)
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
