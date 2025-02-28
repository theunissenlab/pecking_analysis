"""Filtering functions to remove problematic trials due to hardware imperfections
"""
import logging

import numpy as np


logger = logging.getLogger(__name__)


def reject_double_pecks(df, rejection_threshold=200):
    """Remove trials that are interrupted too quickly and add a column named "Masked Stimulus" indicating the preceding one

    When this happens, it makes sense to take the latter trial because the audio
    of the first (interrupted) trial would have been suppressed

    If the underlying cause of this causes double pecks to be registered closer to 100-200ms
    apart, this will tend to allow the first rendition of the first stim through before
    quickly switching to the second stim.

    Double pecks seem to not be intentional so we leave them out of the analysis if a second
    trial happens within 200ms of the first. However, this often allows for a short bit, even a
    full distance call sometimes, to get through. Since this may affect the subject's response
    this filter adds a column that annotates trials that occured within 200ms of the start of
    the previous trial with the preceding stimulus filename.

    This will let us analyze whether that short bit has an effect on the subject's behavior.

    :param rejection_threshold: minimum intertrial duration in ms
    """
    if not len(df):
        return df

    bad_trial_index = np.where(
       np.diff(df["Time"]).astype('timedelta64[ms]') < np.timedelta64(rejection_threshold, "ms")
    )[0]

    df["Masked Stimulus"] = np.empty(len(df), dtype = np.str)
    df["Masked Class"] = np.empty(len(df), dtype = np.str)

    for idx in bad_trial_index:
        # Fill in the masked stim info
        df.at[df.index[idx + 1], "Masked Stimulus"] = df.iloc[idx]["Stimulus"]
        df.at[df.index[idx + 1], "Masked Class"] = df.iloc[idx]["Class"]

    good_trial_index = np.where(
        np.diff(df["Time"]).astype('timedelta64[ms]') >= np.timedelta64(rejection_threshold, "ms")
    )[0]
    good_trial_index = np.concatenate([good_trial_index, [len(df) - 1]])

    if len(good_trial_index) != len(df):
        logger.debug("Rejecting {} 'double pecks' out of {}".format(len(df) - len(good_trial_index), len(df)))

    return df.iloc[good_trial_index]


def fix_response_time_column(df):
    """
    """
    df = df.copy()
    for idx in range(len(df) - 1):
        if df.iloc[idx]["Response"]:
            dt = (df.iloc[idx + 1]["Time"] - df.iloc[idx]["Time"])
            df.at[df.index[idx], "RT"] = dt
    return df


def reject_stuck_pecks(df, iti=(6000, 6050), in_a_row=3):
    """Remove trials that are too close to the stimulus time

    This is the result of a hardware problem where the key can get stuck
    and continue to trigger trials right after a stimulus is finished.
    I don't think we've ever fixed this (as of Feb 2020) but the work around
    is to remove these specific trials by finding those trials with specific
    itis. The specific intervals happen in blocks, and itis range between
    6010ms and 6050ms.

    To minimize how often we grab such intervals, we look for strings of
    trials (>3) with itis within the iti range, and remove those stretches.

    :param iti: tuple of the start, stop times (in ms) to detect where potentially stuck trials
    :param in_a_row: int of how many trials need to fall in the above window in a row
        for us to consider the trials as "stuck" and reject those trials.
    """
    if not len(df):
        return df

    potentially_bad_trials = []
    potentially_bad_trials = np.where(
        (np.abs(np.diff(df["Time"]).astype("timedelta64[ms]")) > np.timedelta64(iti[0], "ms")) &
        (np.abs(np.diff(df["Time"]).astype("timedelta64[ms]")) < np.timedelta64(iti[1], "ms"))
    )[0]

    bad_trials = []
    current_run = []
    for trial_idx in potentially_bad_trials:
        # if its consecutive, keep adding
        if not len(current_run) or trial_idx == current_run[-1] + 1:
            current_run.append(trial_idx)
        else:
            if len(current_run) > 3:
                bad_trials += current_run
            current_run = [trial_idx]
    if len(current_run) >= 3:
        bad_trials += current_run
    bad_trials = np.array(bad_trials)

    good_trial_index = ~np.isin(np.arange(len(df.index)), bad_trials)

    if len(bad_trials):
        logger.debug("Rejecting {} bad 'trials' that were almost exactly 6s in a row".format(len(bad_trials)))

    return df.iloc[good_trial_index]
