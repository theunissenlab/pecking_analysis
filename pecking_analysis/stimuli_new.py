from __future__ import print_function, division
import os
import random
import re

import numpy as np
import ipdb

from neosound.sound import *
from zeebeez.sound import utils

directory = "/auto/fhome/julie/Documents/FullVocalizationBank"
wav_files = utils.get_wav_files(directory, recursive=True)

def parse_julie_stim_name(fname):
    pattern = "([a-z0-9]+)_([0-9]{6})[-_]([a-z\-]+)[_-]([0-9a-z]+).wav"
    fname = os.path.split(fname)[-1]

    return list(re.match(pattern, fname, re.IGNORECASE).groups())

def random_isi_values(number, min_isi, max_isi, sum_isi):

    if not (number * min_isi <= sum_isi <= number * max_isi):
        print("ISI values requested are not possible!")
        return

    isis = np.random.uniform(min_isi, max_isi, number)
    curr_sum = np.sum(isis)

    while curr_sum > sum_isi:
        curr_diff = curr_sum - sum_isi
        ii = np.random.choice(np.arange(len(isis)))
        delta = curr_diff if isis[ii] - curr_diff > min_isi else np.random.uniform(0, isis[ii] - min_isi)
        isis[ii] -= delta
        curr_sum = np.sum(isis)

    while curr_sum < sum_isi:
        curr_diff = sum_isi - curr_sum
        ii = np.random.choice(np.arange(len(isis)))
        delta = curr_diff if isis[ii] + curr_diff < max_isi else np.random.uniform(0, max_isi - isis[ii])
        isis[ii] += delta
        curr_sum = np.sum(isis)

    return isis

sm = SoundManager(HDF5Store, "/tmp/pecking_manager.h5")
sounds = list()
for fname in wav_files:
    sound = Sound(fname, manager=sm)
    bird, date, call_type, number = parse_julie_stim_name(fname)
    sound.annotate(bird=bird,
                   date=date,
                   call_type=call_type,
                   number=number)
    sounds.append(sound)

def day1_reward(sounds, manager, bird, call_type, duration=6*second, single_example=False, number=100, seq_length=8):

    if not isinstance(call_type, list):
        call_type = [call_type]

    is_in = lambda s: (s.annotations["bird"] == bird) and (s.annotations["call_type"][:2].lower() in call_type)
    sounds = Sound.query(sounds, is_in)
    random.shuffle(sounds)

    if single_example:
        sounds = [sounds[0]]

    new_sounds = list()
    for s in sounds:
        s = Sound(s, manager=manager)
        s = utils.songfilt(s, frequency_range=[250*hertz, 8000*hertz])
        s = s.ramp(duration=.02 * second)
        s = s.set_level(70 * dB)
        new_sounds.append(s)
    sounds = new_sounds * np.ceil(seq_length / len(new_sounds))

    output_sounds = list()
    for ii in xrange(number):
        s = Sound.silence(duration)
        sequence = [sounds[ii] for ii in np.random.permutation(len(sounds))[:seq_length]]
        total_dur = np.sum([si.duration for si in sequence])
        isis = random_isi_values(seq_length - 1, .1, .75, float(duration) - total_dur)
        start = 0 * second
        for jj in xrange(seq_length):
            s = sequence[jj].embed(s, start=start)
            start += sequence[jj].duration
            if jj < seq_length - 1:
                start += isis[jj-1] * second

        output_sounds.append(s)

    return output_sounds

def day1_noreward(sounds, manager, bird, call_type, duration=6*second, single_example=False, number=300, seq_length=8):

    if not isinstance(call_type, list):
        call_type = [call_type]

    is_in = lambda s: (s.annotations["bird"] == bird) and (s.annotations["call_type"][:2].lower() in call_type)
    sounds = Sound.query(sounds, is_in)
    random.shuffle(sounds)

    if single_example:
        sounds = [sounds[0]]

    new_sounds = list()
    for s in sounds:
        s = Sound(s, manager=manager)
        s = utils.songfilt(s, frequency_range=[250*hertz, 8000*hertz])
        s = s.ramp(duration=.02 * second)
        s = s.set_level(70 * dB)
        new_sounds.append(s)
    sounds = new_sounds * np.ceil(seq_length / len(new_sounds))

    output_sounds = list()
    for ii in xrange(number):
        s = Sound.silence(duration)
        sequence = [sounds[ii] for ii in np.random.permutation(len(sounds))[:seq_length]]
        total_dur = np.sum([si.duration for si in sequence])
        isis = random_isi_values(seq_length - 1, .1, .75, float(duration) - total_dur)
        start = 0 * second
        for jj in xrange(seq_length):
            s = sequence[jj].embed(s, start=start)
            start += sequence[jj].duration
            if jj < seq_length - 1:
                start += isis[jj-1] * second

        output_sounds.append(s)

    return output_sounds


def day2_reward(sounds, manager, call_type, duration=6*second, number=100, seq_length=8):

    if not isinstance(call_type, list):
        call_type = [call_type]

    is_in = lambda s: (s.annotations["call_type"][:2].lower() in call_type)
    sounds = Sound.query(sounds, is_in)
    random.shuffle(sounds)

    new_sounds = list()
    for s in sounds:
        annotations = s.annotations
        s = Sound(s, manager=manager)
        s = utils.songfilt(s, frequency_range=[250*hertz, 8000*hertz])
        s = s.ramp(duration=.02 * second)
        s = s.set_level(70 * dB)
        s.annotate(**annotations)
        new_sounds.append(s)
    sounds = new_sounds * np.ceil(seq_length / len(new_sounds))

    output_sounds = list()
    for ii in xrange(number):
        s = Sound.silence(duration)
        sequence = [sounds[kk] for kk in np.random.permutation(len(sounds))[:seq_length]]
        total_dur = np.sum([si.duration for si in sequence])
        isis = random_isi_values(seq_length - 1, .1, .75, float(duration) - total_dur)
        start = 0 * second
        output_str = list()
        for jj in xrange(seq_length):
            output_str.append("%s: %s" % (sequence[jj].annotations["bird"],
                                          sequence[jj].annotations["call_type"]))
            s = sequence[jj].embed(s, start=start)
            start += sequence[jj].duration
            if jj < seq_length - 1:
                start += isis[jj-1] * second

        print("%d: " %(ii) + ", ".join(output_str))
        output_sounds.append(s)

    return output_sounds


def day2_noreward(sounds, manager, call_type, duration=6*second, number=300, seq_length=8):

    if not isinstance(call_type, list):
        call_type = [call_type]

    is_in = lambda s: (s.annotations["call_type"][:2].lower() in call_type)
    sounds = Sound.query(sounds, is_in)
    durations = [s.duration for s in sounds]
    max_duration = np.minimum(2 * np.median(durations), np.percentile(durations, 75))
    sounds = [s for s in sounds if float(s.duration) < max_duration]
    random.shuffle(sounds)

    new_sounds = list()
    for s in sounds:
        annotations = s.annotations
        s = Sound(s, manager=manager)
        s = utils.songfilt(s, frequency_range=[250*hertz, 8000*hertz])
        s = s.ramp(duration=.02 * second)
        s = s.set_level(70 * dB)
        s.annotate(**annotations)
        new_sounds.append(s)
    sounds = new_sounds * np.ceil(seq_length / len(new_sounds))

    output_sounds = list()
    for ii in xrange(number):
        s = Sound.silence(duration)
        sequence = [sounds[kk] for kk in np.random.permutation(len(sounds))[:seq_length]]
        total_dur = np.sum([si.duration for si in sequence])
        isis = random_isi_values(seq_length - 1, .1, .75, float(duration) - total_dur)
        start = 0 * second
        output_str = list()
        for jj in xrange(seq_length):
            output_str.append("%s: %s" % (sequence[jj].annotations["bird"],
                                          sequence[jj].annotations["call_type"]))
            s = sequence[jj].embed(s, start=start)
            start += sequence[jj].duration
            if jj < seq_length - 1:
                start += isis[jj-1] * second

        print("%d: " %(ii) + ", ".join(output_str))
        output_sounds.append(s)

    return output_sounds

def day3_reward(sounds, manager, call_type, duration=6*second, number=100, seq_length=8):

    if not isinstance(call_type, list):
        call_type = [call_type]

    is_in = lambda s: (s.annotations["call_type"][:2].lower() in call_type)
    sounds = Sound.query(sounds, is_in)
    durations = [s.duration for s in sounds]
    max_duration = np.minimum(2 * np.median(durations), np.percentile(durations, 75))
    sounds = [s for s in sounds if float(s.duration) < max_duration]
    random.shuffle(sounds)

    new_sounds = dict()
    for s in sounds:
        annotations = s.annotations
        call_type = annotations["call_type"][:2].lower()
        s = Sound(s, manager=manager)
        s = utils.songfilt(s, frequency_range=[250*hertz, 8000*hertz])
        s = s.ramp(duration=.02 * second)
        s = s.set_level(70 * dB)
        s.annotate(**annotations)
        new_sounds.setdefault(call_type, list()).append(s)

    sounds = dict()
    for ct, sound_list in new_sounds.iteritems():
        sounds[ct] = sound_list * np.ceil(seq_length / len(sound_list))

    output_sounds = list()
    for ii in xrange(number):
        s = Sound.silence(duration)

        current_type = np.random.choice(sounds.keys())
        sequence = [sounds[current_type][kk] for kk in np.random.permutation(len(sounds[current_type]))[:seq_length]]
        total_dur = np.sum([si.duration for si in sequence])
        isis = random_isi_values(seq_length - 1, .1, .75, float(duration) - total_dur)
        start = 0 * second
        output_str = list()
        for jj in xrange(seq_length):
            output_str.append("%s: %s" % (sequence[jj].annotations["bird"],
                                          sequence[jj].annotations["call_type"]))
            s = sequence[jj].embed(s, start=start)
            start += sequence[jj].duration
            if jj < seq_length - 1:
                start += isis[jj-1] * second

        print("%d: " %(ii) + ", ".join(output_str))
        output_sounds.append(s)

    return output_sounds

def day3_noreward(sounds, manager, call_type, duration=6*second, number=300, seq_length=8, ntypes=2, random_types=True):

    if not isinstance(call_type, list):
        call_type = [call_type]

    is_in = lambda s: (s.annotations["call_type"][:2].lower() in call_type)
    sounds = Sound.query(sounds, is_in)
    durations = [s.duration for s in sounds]
    max_duration = np.minimum(2 * np.median(durations), np.percentile(durations, 75))
    sounds = [s for s in sounds if float(s.duration) < max_duration]
    random.shuffle(sounds)

    new_sounds = dict()
    for s in sounds:
        annotations = s.annotations
        call_type = annotations["call_type"][:2].lower()
        s = Sound(s, manager=manager)
        s = utils.songfilt(s, frequency_range=[250*hertz, 8000*hertz])
        s = s.ramp(duration=.02 * second)
        s = s.set_level(70 * dB)
        s.annotate(**annotations)
        new_sounds.setdefault(call_type, list()).append(s)

    sounds = dict()
    nper = np.ceil(float(seq_length) / ntypes)
    for ct, sound_list in new_sounds.iteritems():
        sounds[ct] = sound_list * np.ceil(nper / len(sound_list))

    output_sounds = list()
    for ii in xrange(number):
        s = Sound.silence(duration)

        if random_types:
            current_types = np.random.choice(sounds.keys(), ntypes, replace=False)
            sequence = list()
            for ct in current_types:
                sequence.extend([(ct, sounds[ct][kk]) for kk in np.random.permutation(len(sounds[ct]))[:nper]])
            random.shuffle(sequence)
            sequence = sequence[:seq_length]
            seq_types, sequence = zip(*sequence)
            sequence = list(sequence)
            seq_types = list(seq_types)
            if len(np.unique(seq_types[:3])) == 1:
                diff_type = np.random.choice([ct for ct in seq_types if not ct == seq_types[0]])
                ind = seq_types.index(diff_type)
                diff_type = sequence.pop(ind)
                sequence.insert(2, diff_type)

        else:
            sequence = list()
            current_types = sounds.keys()
            for kk in xrange(seq_length):
                ct = current_types[kk % len(current_types)]
                sequence.append(random.choice(sounds[ct]))

        total_dur = np.sum([si.duration for si in sequence])
        isis = random_isi_values(seq_length - 1, .1, .75, float(duration) - total_dur)
        start = 0 * second
        output_str = list()
        for jj in xrange(seq_length):
            output_str.append("%s: %s" % (sequence[jj].annotations["bird"],
                                          sequence[jj].annotations["call_type"]))
            s = sequence[jj].embed(s, start=start)
            start += sequence[jj].duration
            if jj < seq_length - 1:
                start += isis[jj-1] * second

        print("%d: " %(ii) + ", ".join(output_str))
        output_sounds.append(s)

    return output_sounds