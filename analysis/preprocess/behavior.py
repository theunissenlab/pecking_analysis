"""Preprocess behavioral data from pyoperant
"""
from __future__ import annotations

import datetime
import os

import numpy as np
import pandas as pd

import analysis.preprocess.trial_filters as trial_filters


COLUMN_NAMES = ["Session", "Trial", "Time", "Stimulus", "Class",
          "Response", "Correct", "RT", "Reward", "Max Wait"]


def rt_to_timedelta(rt: str) -> datetime.timedelta:
    """Converts a response time string from pyoperant csv to a datetime.timedelta

    I don't understand why it is allowed to return string nans
    """
    if rt in ("nan", ""):
        return None

    else:
        hours, minutes, seconds = [float(ss) for ss in rt.split(":")]
        return datetime.timedelta(
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )


def load_csv(csv_path: str, config, chunksize: int=None) -> pd.DataFrame:
    """Load pyoperant csv file

    Use chunksize to only read a small piece of the file
    """
    try:
        df = pd.read_csv(
            csv_path,
            header=0,
            names=config.csv_column_names,
            parse_dates=True,
            converters={
                "RT": rt_to_timedelta,
                "Time": pd.to_datetime,
            },
            chunksize=chunksize
        )

        if chunksize:
            return df.get_chunk(chunksize)
    except:
        raise IOError("Could not read csv {} using config. Perhaps the column names have changed?".format(csv_path))

    return df


def filter_trials(df: pd.DataFrame, filter_dicts: list) -> pd.DataFrame:
    """Apply filters from analysis.preprocess.trial_filters to a pyoperant dataframe"""
    for filter_dict in filter_dicts:
        filter_fn = getattr(trial_filters, filter_dict["name"])
        df = filter_fn(df, **filter_dict["kwargs"])

    return df


if __name__ == "__main__":
    from analysis.configs import project_lesions_2021 as config

    for subject in config.subjects:
        print(load_all_for_subject(subject, config))
