from __future__ import print_function, division
import os
import random

import numpy as np
import ipdb

from neosound.sound import *

output_directory = "/auto/tdrive/tlee/operant/stimuli/song_colony_distractors_0dB"
sound_file = "neostimuli.h5"
distractors = ["/auto/tdrive/tlee/operant/stimuli/distractors"]
rewarded = ["/auto/tdrive/tlee/operant/stimuli/shaping/Track5long.wav"]
nonrewarded = ["/auto/tdrive/tlee/operant/stimuli/shaping/Track1long.wav"]

nrewarded = 100
nnonrewarded = 300
ndistractors_per_stim = 1
max_delay = .15 * second
signal_distractor_ratio = 0 * dB


def get_files(file_list):

    if isinstance(file_list, str):
        file_list = [file_list]

    files = list()
    for fname in file_list:
        if os.path.exists(fname):
            if os.path.isdir(fname):
                files.extend(get_files(map(lambda x: os.path.join(fname, x), os.listdir(fname))))
            elif fname.lower().endswith(".wav"):
                print("Adding wavefile to list: %s" % fname)
                files.append(fname)
            else:
                print("File is neither directory nor .wav: %s" % fname)
        else:
            print("File does not exist: %s" % fname)

    return files

def get_sounds(file_list, manager):

    files = get_files(file_list)
    sounds = list()
    for fname in files:
        sounds.append(Sound(fname, manager=manager))

    return sounds

def process_stimulus(sound):

    sound = sound.to_mono()
    sound = sound.filter([250 * hertz, 8000 * hertz])
    sound = sound.ramp(duration=.02 * second)
    sound = sound.set_level(70 * dB)

    return sound

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
if not isinstance(signal_distractor_ratio, (list, np.ndarray)):
    signal_distractor_ratio = [signal_distractor_ratio]

sm = SoundManager(HDF5Store, os.path.join(output_directory, sound_file))
distractors = get_files(distractors)
rewarded = get_sounds(rewarded, sm)
nonrewarded = get_sounds(nonrewarded, sm)

reward_out = list()
reward_distractors = list(np.array(distractors * np.ceil(nrewarded * ndistractors_per_stim / len(distractors)))[:nrewarded * ndistractors_per_stim])
reward_distractors = list(np.random.permutation(reward_distractors))
for ii in xrange(nrewarded):
    print("Creating combined reward stimulus %d" % ii)
    r_ind = np.random.randint(0, len(rewarded))
    d_inds = range(ii * ndistractors_per_stim, (ii + 1) * ndistractors_per_stim)
    r_stim = rewarded[r_ind].to_mono()
    sdr = signal_distractor_ratio[np.random.randint(0, len(signal_distractor_ratio))]
    for d_ind in d_inds:
        r_stim = r_stim.embed(Sound(reward_distractors[d_ind], manager=sm),
                              max_start=max_delay,
                              ratio=sdr)
    combined = process_stimulus(r_stim)
    reward_out.append(combined)
    combined.save(os.path.join(output_directory, "reward_%03d.wav" % ii))


nonreward_out = list()
nonreward_distractors = list(np.array(distractors * np.ceil(nnonrewarded * ndistractors_per_stim / len(distractors)))[:nnonrewarded * ndistractors_per_stim])
nonreward_distractors = list(np.random.permutation(nonreward_distractors))
for ii in xrange(nnonrewarded):
    print("Creating combined nonreward stimulus %d" % ii)
    n_ind = np.random.randint(0, len(rewarded))
    d_inds = range(ii * ndistractors_per_stim, (ii + 1) * ndistractors_per_stim)
    n_stim = nonrewarded[n_ind].to_mono()
    sdr = signal_distractor_ratio[np.random.randint(0, len(signal_distractor_ratio))]
    for d_ind in d_inds:
        n_stim = n_stim.embed(Sound(nonreward_distractors[d_ind], manager=sm),
                              max_start=max_delay,
                              ratio=sdr)
    combined = process_stimulus(n_stim)
    nonreward_out.append(combined)
    combined.save(os.path.join(output_directory, "nonreward_%03d.wav" % ii))











