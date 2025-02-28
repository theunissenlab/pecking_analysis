"""
Data pipeline for Zebra Finch Memory Lesions project
"""
import datetime
import glob
import logging
import sys
import os

import re
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from analysis.preprocess.behavior import load_csv, filter_trials
from analysis.analysis import create_informative_trials_column


logger = logging.getLogger(__name__)


class NoDataForSubject(Exception):
    pass


def get_test_context(full_stim_path: str) -> str:
    """Return the 'test' the stim is from using the directory structure

    Stimuli were organized like ... stimuli/<group>/<TESTFOLDER>/box2/rewarded/ ...

    e.g.
    stimuli/ladder/SovsSo_4v4/box2/...
    stimuli/shaping/shapingSongID/...
    stimuli/preference_tests/<SubjectName>/...
    """
    x, y = os.path.split(full_stim_path)
    if os.path.split(x)[1] == "ladder":
        return y
    elif os.path.split(x)[1] == "shaping":
        return "shaping"
    elif os.path.split(x)[1] == "stimuli":
        return "preference"
    else:
        return get_test_context(x)


def read_metadata_from_stimulus_filename(stim_basename: str) -> Dict[str, str]:
    """Extract metadata about trials from the filename

    Should return a dict mapping the feature name to the
    value extracted.

    Expect: "Call Type", "Vocalizer"
    """
    sections = stim_basename.split("_")

    for i, section in enumerate(sections):
        if section.upper() in ["SO", "SONG"]:
            call_type = "SO"
            break
        elif section.upper() in ["DC"]:
            call_type = section.upper()
            break
    else:
        call_type = "SO"

    for i, section in enumerate(sections):
        if re.search("[a-zA-Z]{6}[a-zA-Z0-9]{4}", section):
            bird_name = section
            rendition = "_".join([section, sections[i-1]])
            break
        if re.search("[a-zA-Z]{3}[a-zA-Z0-9]{2}", section):
            bird_name = section
            rendition = "_".join([section, sections[i-1]])
            break
    else:
        bird_name = filename

    return {
        "Vocalizer": bird_name,
        "Call Type": call_type
    }


def add_stimulus_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing of stimuli information in block

    Adds new columns to block's dataframe, and creates a new separate dataframe
    with stimulus information. Each dataframe is given a new column called "Stim Key"
    that is sufficient for joining these dataframes later.

    Columns added to block:
        Test Context - the test block the stimulus is from (eg DCvsDC_1v1)
        Stim Key - a key (used to join stimuli to the corresponding rows in trials df)
    """
    df = df.copy()

    def _to_row(filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        return read_metadata_from_stimulus_filename(basename)

    _stim_rows = list(df["Stimulus"].apply(_to_row))
    _stim_rows = pd.DataFrame(_stim_rows)

    df = pd.concat([df, _stim_rows], axis=1)
    df["Stim Key"] = df.agg("{0[Vocalizer]} {0[Call Type]} {0[Class]}".format, axis=1)
    df["Test Context"] = df["Stimulus"].apply(get_test_context)
    return df


def split_shaping_preference_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_shaping = df[df["Test Context"] == "shaping"].reset_index()
    df_preference = df[df["Test Context"] == "preference"].reset_index()
    df_test = df[~np.isin(df["Test Context"], ("shaping", "preference"))].reset_index()

    return df_shaping, df_preference, df_test


def extract_stim_df(df: pd.DataFrame) -> pd.DataFrame:
    # Generate stimulus dataframe
    split_on_columns = ["Vocalizer", "Call Type", "Class"]
    grouped = df.groupby(split_on_columns)
    stim_data = []
    for (vocalizer, call_type, rewarded), _ in sorted(grouped.groups.items(), key=lambda x: (x[0][1], len(x[0][0]), x[0][0])):
        stim_key = "{} {} {}".format(vocalizer, call_type, rewarded)
        stim_data.append([stim_key, vocalizer, call_type, rewarded])

    stims_df = pd.DataFrame(stim_data, columns=["Stim Key", "Vocalizer", "Call Type", "Class"])

    return stims_df


def apply_column_names(df: pd.DataFrame, column_map: Dict[str, Tuple[str, Callable]]) -> pd.DataFrame:
    series = {}
    for new_key, (old_key, fn) in column_map.items():
        if old_key is None:
            series[new_key] = pd.Series([None] * len(df))
        elif fn is not None:
            series[new_key] = df[old_key].apply(fn)
        else:
            series[new_key] = df[old_key]

    return pd.DataFrame(series)


def load_all_for_subject(subject: str, config: dict, from_date: datetime.date=None, to_date: datetime.date=None):
    """Loads all dataframes for a subject, filters the trials, and orders them by time
    """
    raw_files = glob.glob(os.path.join(config.save_dir, subject, "*", "*.csv"))
    dfs = []

    for csv in raw_files:
        # Read a preview of the file to get timestamp
        try:
            preview_df = load_csv(csv, config, chunksize=1)
        except IOError as e:
            logger.warning(e)
            continue
            
        if not len(preview_df):
            continue
        if from_date is not None and preview_df.iloc[0]["Time"].date() < from_date:
            continue
        if to_date is not None and preview_df.iloc[0]["Time"].date() > to_date:
            continue

        df = load_csv(csv, config)

        if not len(df):
            continue

        df = filter_trials(df, filter_dicts=config.filters)
        dfs.append(df)

    if not len(dfs):
        raise NoDataForSubject("No data in the date range {} to {} found for subject {} in {}".format(
            from_date, to_date, subject, config.save_dir
        ))

    df = pd.concat(dfs, axis=0).sort_values("Time").reset_index()
    df["Subject"] = subject

    df = df.reset_index(drop=True)
    return df


def run_pipeline_subject(subject: str, config: dict, from_date: datetime.date=None, to_date: datetime.date=None, include_shaping: bool=False):
    logger.debug("Running data pipeline for {}".format(subject))
    df = load_all_for_subject(subject, config, from_date, to_date)

    # Here we have a specific step to correct the data from Jan 23
    df = add_stimulus_columns(df)
    
    shaping_df, _, df = split_shaping_preference_test(df)

    if not include_shaping and not len(df):
        raise Exception("No non-shaping data from {} to {}".format(
            from_date,
            to_date,
        ))
    elif include_shaping and not len(df) and not len(shaping_df):
        raise Exception("No data from {} to {}".format(
            from_date,
            to_date,
        ))
    elif include_shaping and not len(df):
        df = shaping_df

    stims_df = extract_stim_df(df)
    df = apply_column_names(df, config.column_mapping)

    # df = create_informative_trials_column(df, as_column="Informative Trials Seen")
    # logger.info("Calculated Informative Trials Columns")

    return df


if __name__ == "__main__":
    from configs import config
    from analysis.download_scripts.project_lesions_2021 import download

    try:
        download()
    except:
        pass

    for subject in config.subjects:
        # Preprocessing steps
        df = run_pipeline_subject(subject, config)
        print(df)
