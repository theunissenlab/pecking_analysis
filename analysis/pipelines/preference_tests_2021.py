"""
Querying and loading preference test data
"""
import datetime
import glob
import logging
import os

import pandas as pd

from analysis.preprocess.behavior import load_csv
from analysis.pipelines.project_lesions_2021 import get_test_context


logger = logging.getLogger(__name__)


def load_preference_test_data(subject: str, config, from_date: datetime.date=None, to_date: datetime.date=None):
    """Loads all dataframes for a subject, filters the trials, and orders them by time

    Data frame has columns

    Session
    Trial
    Stimulus File

    This information should be sufficient to link to the audio file
    """
    raw_files = glob.glob(os.path.join(config.behavior_data_dir, subject, "raw", "*.csv"))
    dfs = []

    for csv in raw_files:
        # Read a preview of the file to get timestamp
        preview_df = load_csv(csv, config, chunksize=1, fast=True)
        if not len(preview_df):
            continue
        if from_date is not None and preview_df.iloc[0]["Time"].date() < from_date:
            continue
        if to_date is not None and preview_df.iloc[0]["Time"].date() > to_date:
            continue

        logger.debug("Loaded data from {}".format(csv))
        df = load_csv(csv, config)
        df = df[df["Stimulus"].apply(lambda x: get_test_context(x) == "preference")]
        df = df[["Trial", "Time", "Stimulus"]]
        df["Session"] = os.path.splitext(os.path.basename(csv))[0]
        dfs.append(df)

    df = pd.concat(dfs, axis=0).sort_values("Time").reset_index(drop=True)
    df["Subject"] = subject

    # df = df.reset_index(drop=True)
    return df


def query_preference_tests(subject: str, config, from_date: datetime.date=None, to_date: datetime.date=None):
    logger.debug("Querying preference test data")

    df = load_preference_test_data(subject, config, from_date, to_date)

    _cache = {}
    def get_audio_location(row):
        path = os.path.join(config.behavior_data_dir, row["Subject"], "preference_test_audio", row["Session"])
        if path not in _cache:
            _cache[path] = os.path.exists(path)
        exists = _cache[path]

        if exists:
            return os.path.join(path, "trial{}.wav".format(row["Trial"]))
        else:
            return ""

    if not len(df):
        return None
    else:
        df["Audio"] = df.apply(get_audio_location, axis=1)
        return df
