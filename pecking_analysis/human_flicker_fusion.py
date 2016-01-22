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


if False:
    # Begin Experiment
    import pandas as pd
    from pecking_analysis.human_flicker_fusion import generate_clicks
    from pecking_analysis.peck_data import peck_data
    from pecking_analysis.objects import Block

    shapingModelTrials = 10
    probeModelTrials = 10
    probeConverge = 3
    minFreq = 10
    maxFreq = 100
    currentFrequencies = list()
    prevLLRPerProbe = None
    nSinceBest = 0
    block = Block(name=expInfo["participant"])

    # Create pandas dataframe as block.data
    # Block.data should be indexed on 'Time' and columns of 'Session', 'Trial', 'Stimulus', 'Class', 'Response', 'Correct', 'RT', 'Reward', 'Max Wait'
    # Time is not used anywhere that we will need it, so let's skip that.
    # We could preallocate the max number of trials, perhaps
    block.data = pd.DataFrame(columns=("Session", "Trial", "Stimulus",
                                       "Class", "Response", "Correct", "RT",
                                       "Reward", "Max Wait"))

    # Shaping routine
    # End Routine 1
    # Every XX trials compute the performance of the subject
    # If performance is below some threshold (0.05) then end routine
    # End routine with "trials.finished = True"

    response = trials.data["response.keys"] is not None
    condition = trials.trialList[trials.thisIndex]
    if condition == "Nonreward" and response == True:
        correct = True
        reward = False
    elif condition == "Reward" and response == False:
        correct = True
        reward = True
    else:
        correct = False
        reward = False

    if response == True:
        rt = trials.data["response.rt"]
    else:
        rt = None

    # Data is stored as a dictionary of 2D arrays
    # http://www.psychopy.org/general/dataOutputs.html
    # Need to store frequency in filename...
    block.data.loc[trials.thisN] = [0, trials.thisN, '',
                                    condition, response, correct,
                                    rt, reward, 10] # List of values


    if trials.thisN % shapingModelTrials == 0:
        performance = peck_data(block)
        if performance is not None:
            if performance["Stats", "P-Value"] < 0.05:
                trials.finished = True


    # Probe routine
    # End Routine 1
    # Every XX trials compute the model of the subject's performance.
    # Overwrite the probe stimuli
    # Check convergence of model over last XX estimates

    # Find probe condition number
    probeNo = trials.trialList.index("Probe") # ??
    nProbes = len(trials.data["response.keys"][probeNo])
    if nProbes % probeModelTrials == 0:
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
        sounds = generate_clicks([freq for freq in currentFrequencies if minFreq <= freq <= maxFreq], sample_rate=44100)
        for sound in sounds:
            wavwrite(sound, 44100, filename)



    # Start Routine 1
    # Randomly choose reward/nonreward/probe directory
    # Use $stimDirectory in the GUI when choosing stimulus directory
    # I don't actually think it works this way
    # Shape exp
    if random() <= rewardFrequency:
        stimDirectory = "Reward"
    else:
        stimDirectory = "Nonreward"

    # Probe exp
    randVal = random()
    if randVal <= probeFrequency:
        stimDirectory = "Probe"
    elif randVal <= probeFrequency + rewardFrequency:
        stimDirectory = "Reward"
    else:
        stimDirectory = "Nonreward"
