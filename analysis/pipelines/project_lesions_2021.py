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


class NoDataForSubject(Exception):
    pass


def _get_test_context_old(full_stim_path: str) -> str:
    x, y = os.path.split(full_stim_path)

    if os.path.split(x)[1] == "stimuli":
        if y == "preference_tests":
            return "preference"
        elif y == "shapingSongID":
            return "shaping"
        else:
            return y
    else:
        return _get_test_context_old(x)


def _get_test_context_new(full_stim_path: str) -> str:
    x, y = os.path.split(full_stim_path)

    if os.path.split(x)[1] == "ladder":
        return y
    elif os.path.split(x)[1] == "shaping":
        return "shaping"
    elif os.path.split(x)[1] == "stimuli":
        return "preference"
    else:
        return get_test_context(x)


def get_test_context(full_stim_path: str) -> str:
    """Return the 'test' the stim is from using the directory structure

    This function disambiguates between two folder organizations we've used

    The new version, paths start with "/data/pecking_test/stimuli"
    Stimuli were organized like ... stimuli/<group>/<TESTFOLDER>/box2/rewarded/ ...

    e.g.
    stimuli/ladder/SovsSo_4v4/box2/...
    stimuli/shaping/shapingSongID/...
    stimuli/preference_tests/<SubjectName>/...

    The old style, paths started with "/home/fet/stimuli"

    stimuli/SovsSo_4v4/box2/...
    stimuli/preference_tests/box2/...
    stimuli/shapingSongID/...
    """
    if full_stim_path.startswith("/home/fet/stimuli"):
        return _get_test_context_old(full_stim_path)
    elif full_stim_path.startswith("/data/pecking_test/stimuli"):
        return _get_test_context_new(full_stim_path)
    else:
        raise ValueError("No handler for stimulus at {} found".format(full_stim_path))


def read_metadata_from_stimulus_filename(stim_basename: str, prefix: str=None) -> Dict[str, str]:
    """Extract metadata about trials from the filename

    Should return a dict mapping the feature name to the
    value extracted.

    Expect: "{prefix}Call Type", "{prefix}Vocalizer"
    """
    if not stim_basename:
        return {
            "{}Vocalizer".format(prefix or ""): None,
            "{}Call Type".format(prefix or ""): None
        }

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
        bird_name = stim_basename

    return {
        "{}Vocalizer".format(prefix or ""): bird_name,
        "{}Call Type".format(prefix or ""): call_type
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

    def _to_row(filename, prefix=None):
        if filename is not None:
            basename = os.path.splitext(os.path.basename(filename))[0]
            return read_metadata_from_stimulus_filename(basename, prefix=prefix)
        else:
            return read_metadata_from_stimulus_filename(None, prefix=prefix)

    _stim_rows = list(df["Stimulus"].apply(_to_row))
    _stim_rows = pd.DataFrame(_stim_rows)

    _masked_stim_rows = list(df["Masked Stimulus"].apply(lambda x: _to_row(x, prefix="Masked ")))
    _masked_stim_rows = pd.DataFrame(_masked_stim_rows)

    df = pd.concat([df, _stim_rows, _masked_stim_rows], axis=1)
    df["Stim Key"] = df.agg("{0[Vocalizer]} {0[Call Type]} {0[Class]}".format, axis=1)
    df["Test Context"] = df["Stimulus"].apply(get_test_context)
    return df


def split_shaping_preference_test(
        df: pd.DataFrame,
        config,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_shaping = df[df["Test Context"] == "shaping"].reset_index(drop=True)
    df_preference = df[df["Test Context"] == "preference"].reset_index(drop=True)
    df_test = df[np.isin(df["Test Context"], config.valid_test_contexts)].reset_index(drop=True)

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


def load_all_for_subject(subject: str, config, from_date: datetime.date=None, to_date: datetime.date=None):
    """Loads all dataframes for a subject, filters the trials, and orders them by time
    """
    raw_files = glob.glob(os.path.join(config.behavior_data_dir, subject, "*", "*.csv"))
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

    # df = df.reset_index(drop=True)
    return df


def join_subject_metadata(df: pd.DataFrame, subject_metadata: pd.Series):
    """Populate columns that require subject metadata information

    The columns needed are Condition, Subject Group, and Ladder Group.

    I would rename them to Condition, Treatment, and Ladder Group
    """
    df = df.copy()

    df["Treatment"] = subject_metadata["Treatment"]

    def get_ladder_group(row):
        is_prelesion = row["Date"] <= subject_metadata["Lesion Date"]
        is_set_2 = row["Test Context"].endswith("S2")
        if is_prelesion and not is_set_2:
            return "PrelesionSet1"
        elif not is_prelesion and not is_set_2:
            return "PostlesionSet1"
        elif not is_prelesion and is_set_2:
            return "PostlesionSet2"
        else:
            raise Exception("This dataset should not have any prelesion data for set 2")

    df["Ladder Group"] = df.apply(get_ladder_group, axis=1)

    return df


def renumber_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber the trial indexes on each day
    """
    df = df.copy()
    for _, day_subdf in df.groupby(["Subject", "Date"]):
        df.loc[day_subdf.index, "Trial"] = np.arange(len(day_subdf))
    return df


def remove_vocalizers_from_date(df: pd.DataFrame, date: datetime.date):
    """Removes all trials from the dataset matching the vocalizer/call type on a given date

    Use this function when data was spoiled due to experimental error and some data is unusable
    """
    df = df.copy()

    day_df = df[df["Date"] == date]

    to_remove = []
    for (vocalizer, call_type, reward), _ in day_df.groupby(["Stimulus Vocalizer", "Stimulus Call Type", "Stimulus Class"]):
        to_remove.append((vocalizer, call_type, reward))

    for (vocalizer, call_type, reward) in to_remove:
        df = df[~(
            (df["Stimulus Vocalizer"] == vocalizer) &
            (df["Stimulus Call Type"] == call_type) &
            (df["Stimulus Class"] == reward)
        )]

    return df


def run_pipeline_subject(subject: str, config: dict, from_date: datetime.date=None, to_date: datetime.date=None, include_shaping: bool=False):
    logger.debug("Running data pipeline for {}".format(subject))
    df = load_all_for_subject(subject, config, from_date, to_date)

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

    df = create_informative_trials_column(df, as_column="Informative Trials Seen")
    logger.info("Calculated Informative Trials Columns")
    df = renumber_trials(df)

    subjects_metadata = pd.read_csv(
        config.subject_metadata_path,
        parse_dates=True,
        converters={"Lesion Date": pd.to_datetime}
    )
    subjects_metadata["Lesion Date"] = subjects_metadata["Lesion Date"].apply(lambda x: x.date())
    subject_metadata = subjects_metadata.query("Subject == '{}'".format(subject)).iloc[0]

    df = join_subject_metadata(df, subject_metadata)

    # Here we have a specific step to correct and update the data from specific subjects
    if subject == "GreBla7410M":
        idx = df[
            (df["Subject"] == "GreBla7410M") &
            (df["Date"] >= datetime.date(2020, 1, 7)) &
            (df["Date"] <= datetime.date(2020, 1, 10))
        ].index
        df[idx, "Condition"] = "MonthLater"
    if subject == "GreBla5671F":
        idx = df[
            (df["Subject"] == "GreBla5671F") &
            (df["Date"] >= datetime.date(2020, 1, 7)) &
            (df["Date"] <= datetime.date(2020, 1, 10))
        ].index
        df[idx, "Condition"] = "MonthLater"

    if subject in ["XXXBlu0031M", "HpiGre0651M", "RedHpi0710F", "WhiBlu5805F"]:
        df = remove_vocalizers_from_date(df, datetime.date(2021, 1, 23))

    return df


if __name__ == "__main__":
    from configs.active_config import config
    from analysis.download_scripts.project_lesions_2021 import download

    try:
        download()
    except:
        pass

    subject_dfs = []
    for subject in config.subjects:
        # Preprocessing steps
        df = run_pipeline_subject(subject, config)
        subject_dfs.append(df)

    full_df = pd.concatenate(subject_dfs).reset_index(drop=True)
    full_df.to_csv(
        os.path.join(config.metadata_dir, "TrialsData.csv"),
        index=False
    )
