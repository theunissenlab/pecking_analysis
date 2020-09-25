import sys

sys.path.append("/auto/fhome/kevin/Projects/pecking_analysis")

import os
import datetime
import glob
import re
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact

from pecking_analysis.peck_data import (
    load_pecking_days,
    get_labels_by_combining_columns,
    plot_data,
    peck_data,
    color_by_reward,
    windows_by_reward,
    get_dates,
)

from subject_info import evaluation_days, brain_group, DATADIR


def cached_load(bird, date, date_end=None):
    """Load pecking test data over a date or date range

    Caches result for a given date range so it doesn't need to be reloaded
    """
    if (bird, date, date_end) not in cached_load._cache:
        if date_end is not None:
            blocks, stim_blocks = load_pecking_days(os.path.join(DATADIR, bird), date_range=(date, date_end))
            cached_load._cache[(bird, date, date_end)] = (blocks, stim_blocks)

        else:
            blocks, stim_blocks = load_pecking_days(os.path.join(DATADIR, bird, date.strftime("%d%m%y")))
            cached_load._cache[(bird, date, date_end)] = (blocks, stim_blocks)

    return cached_load._cache[(bird, date, date_end)]
cached_load._cache = {}



if __name__ == "__main__":

    if not len(sys.argv) > 1:
        raise Exception("""
            Must specify process mode
            ("behavior" or "lesions")
        """)

    mode = sys.argv[1]
    valid_modes = ["behavior", "lesions"]
    if mode not in valid_modes:
        raise Exception("Mode must be one of {}".format(valid_modes))

    BIRDS = [
        os.path.basename(bird)
        for bird in sorted(glob.glob(os.path.join(DATADIR, "*")), key=os.path.getmtime, reverse=True)
        if re.search(r"[a-zA-Z]{6}[0-9]{4}[MF]?$", os.path.basename(bird))
    ]


    # Create table with each subject and each day
    lesions = [thing for thing in evaluation_days]
    rows = []
    for lesion in lesions:
        for condition in evaluation_days[lesion]:
            for subject in evaluation_days[lesion][condition]:
                for date in evaluation_days[lesion][condition][subject]:
                    rows.append([lesion, condition, subject, date, brain_group[subject]])
    conditions = pd.DataFrame(rows, columns=["lesion", "test", "bird", "date", "lesion_group"])


    def get_subject_conditions(conditions, subject_name):
        """Read conditions table for one subject"""
        return conditions.where(
            conditions["bird"] == subject_name
        ).dropna().sort_values("date").reset_index(drop=True)


    def str2date(datestring):
        return datetime.datetime.strptime(datestring, "%Y-%m-%d")


    def investigate_stim_folders(bird):
        """Creating a massive new dataframe with all the trials for this bird

        Create columns for the following:
        - Consecutive
            How many consecutive test days has the stimulus been played in one ladder
        - Lesion
            Which lesion state is the subject in (prelesion or postlesion)
        - Lesion Group
            Which experimental group is the subject in (NCM, CM, CTRL, or other)
        - test_folder
            The stage of the ladder - e.g. DCvsDC_1v1, SovsSo_4v4_S2, SovsSo_8v8_d1_S2, etc
        - Subject
            Subject name
        - Overall Seen Before
            How many times this stimulus has been used before
        """
        bird_csv = "{}.csv".format(bird)
        subj_conditions = get_subject_conditions(conditions, bird_csv)
        
        is_prelesion = True
        seen_counts = {}
        overall_counts = {}
        
        new_dfs = []
        used_dates = []
        for idx in subj_conditions.index:
            row = subj_conditions.iloc[idx]
            if is_prelesion and row["lesion"] == "postlesion":
                is_prelesion = False
                seen_counts = {}
                
            if str2date(row["date"]).date() >= datetime.date.today():
                continue

            if str2date(row["date"]).date() in used_dates:
                continue

            used_dates.append(str2date(row["date"]).date())
                
            blocks, stim_blocks = cached_load(bird, str2date(row["date"]))

            if not len(blocks):
                continue

            block = blocks[0]

            if not len(block.data):
                continue
            
            def get_exact_condition(full_stim_path):
                x, y = os.path.split(full_stim_path)
                if os.path.split(x)[1] == "stimuli":
                    return y
                else:
                    return get_exact_condition(x)

            # Number Waited column: if the trial was not interrupted, how many times has this same stim been not interrupted previously
            #   first, initialize the column to zeros
            block.data["Wait Index"] = pd.Series(-1 * np.ones(len(block.data)), dtype=np.int32, index=block.data.index)
            #   trials without interruption have NaN as the response time column
            waited = block.data[pd.isnull(block.data["RT"]) | (block.data["RT"] == "nan")]
            #   Set those trials with ith time that the stimulus was waited for 
            block.data["Wait Index"].loc[waited.index] = waited.groupby(["Bird Name"]).cumcount()
            
            ###  Update the stimuli table ###
            # "Lesion": a column that says if its pre or post-lesion
            lesion_state = "prelesion" if is_prelesion else "postlesion"
            lesion_states = []
            for unique_stim in block.stimuli["Bird Name"]: ## The Bird Name column is the stimulus name column.
                lesion_states.append(lesion_state)
            block.stimuli["Lesion"] = pd.Series(lesion_states, index=block.stimuli.index)

            # "test_folder": a column for the stimulus folder used, e.g. SovsSo_8v8_d1 
            block.data["test_folder"] = block.data["Stimulus"].apply(get_exact_condition)
            block.data["Stim Key"] = pd.Series(
                [
                    "{} {} {}".format(bird_name, call_type, reward)
                    for bird_name, call_type, reward in zip(
                        block.data["Bird Name"], block.data["Call Type"], block.data["Class"]
                    )
                ],
                index=block.data.index
            )
            

            # "Consecutive" and "Overall Seen Before":
            #     Number of days the stimulus has been used in a row
            #     Number of days the stimulus has been used overall 
            for stim_idx in block.stimuli.index:
                stim = block.stimuli.loc[stim_idx, "Stim Key"]
                if stim not in seen_counts:
                    seen_counts[stim] = 0
                else:
                    seen_counts[stim] += 1
                
                if stim not in overall_counts:
                    overall_counts[stim] = 0
                else:
                    overall_counts[stim] += 1                
            block.stimuli["Consecutive"] = block.stimuli["Stim Key"].apply(lambda x: seen_counts.get(x, 0))
            block.stimuli["Overall Seen Before"] = block.stimuli["Stim Key"].apply(lambda x: overall_counts.get(x, 0))
            block.stimuli["New"] = block.stimuli["Consecutive"] == 0
            
            for key in list(seen_counts.keys()):
                if not np.isin(key, block.stimuli["Stim Key"]):
                    del seen_counts[key]
            
            # Include the stimulus metadata
            new_df = block.data.join(block.stimuli.set_index("Stim Key"), on="Stim Key", lsuffix="_")
        
            # Add a column naming the subject
            new_df["Subject"] = bird
            new_df["Lesion Group"] = brain_group["{}.csv".format(bird)]

            new_dfs.append(new_df)

        if len(new_dfs):
            return pd.concat(new_dfs)
        else:
            return None

    columns = [
        'Session', 
        'Trial', 
        'Stimulus', 
        'Response', 
        'Correct',
        'RT',
        'Reward',
        'Max Wait', 
        'Time', 
        'OverallTrial',
        'Filename',
        'Date',  
        'Stimulus Name',
        'Rendition', 
        'Stim Key',
        'Trial Number',
        'Wait Index', 
        'test_folder', 
        'Bird Name',
        'Call Type',
        'Class',
        'Trials',
        'Overall Seen Before',
        'Consecutive',
        'New',
        'Lesion',
        'Subject',
        'Lesion Group',
    ]

    all_dfs = []
    for bird in BIRDS:
        new_df = investigate_stim_folders(bird)
        if new_df is None:
            print("No df for {}".format(bird))
        else:
            all_dfs.append(new_df[columns])
    full_df = pd.concat(all_dfs)


    # The Wait Index per day is nice, but lets put in a wait index per LADDER:
    full_df["LadderGroup"] = None
    full_df["LadderGroup"] = None
    full_df["LadderGroup"].loc[full_df["Lesion"] == "prelesion"] = "PrelesionSet1"
    full_df["LadderGroup"].loc[(full_df["Lesion"] == "postlesion") & 
                                      (full_df["test_folder"].apply(lambda x: not x.endswith("S2")))] = "PostlesionSet1"
    full_df["LadderGroup"].loc[(full_df["Lesion"] == "postlesion") & 
                                      (full_df["test_folder"].apply(lambda x: x.endswith("S2")))] = "PostlesionSet2"

    full_df["Wait Count"] = np.nan
    for (_), sub_df in full_df.groupby(["Subject", "LadderGroup", "Bird Name", "Call Type", "Lesion"]):
        curr_waits = 0
        for idx in sub_df.index:
            full_df.loc[idx, "Wait Count"] = curr_waits

            if sub_df.loc[idx, "Response"] == False:
                curr_waits += 1

    full_df["TimeToNext"] = 0

    for _, subdf in full_df.groupby(["Subject", "Date"]):
        
        x = []
        curr_time = None
        for idx in subdf.index:
            if curr_time is not None:
                dt = subdf["Time"].loc[idx] - curr_time
                x.append(dt / np.timedelta64(1, 's'))
            curr_time = subdf["Time"].loc[idx]
        x.append(None)

        full_df.loc[subdf.index, "TimeToNext"] = x


    if mode == "behavior":
        copy_df = full_df.copy()
        copy_df = copy_df[(copy_df["LadderGroup"] == "PrelesionSet1") | (copy_df["LadderGroup"] == "PrelesionSet2")].copy()
        display(copy_df.head())
        new_df = copy_df[["Subject", "Trial", "Date", "RT", "Reward"]].copy()

        # time column is implicitly there as the index

        # Now fill in a column to distinguish the sessions that occured after a 1 month break
        conditions = pd.Series(["None"] * len(new_df), index=new_df.index)

        conditions[
            (new_df["Subject"] == "GreBla7410M") &
            (new_df["Date"] >= datetime.date(2020, 1, 7)) &
            (new_df["Date"] <= datetime.date(2020, 1, 10))
        ] = "MonthLater"

        conditions[
            (new_df["Subject"] == "GreBla5671F") &
            (new_df["Date"] >= datetime.date(2020, 1, 7)) &
            (new_df["Date"] <= datetime.date(2020, 1, 10))
        ] = "MonthLater"

        new_df["Condition"] = conditions
           
        new_df["Interrupt"] = copy_df["Response"]
        new_df["Stimulus File"] = copy_df["Stimulus"].apply(lambda x: os.path.basename(x))
        new_df["Stimulus Vocalizer"] = copy_df["Bird Name"]
        new_df["Stimulus Call Type"] = copy_df["Call Type"]
        new_df["Stimulus Class"] = copy_df["Class"].apply(lambda x: "Rewarded" if x == "Rewarded" else "Nonrewarded")
        new_df["Informative Trials Seen"] = copy_df["Wait Count"].astype(np.int)
        new_df["Test Context"] = copy_df["test_folder"]
        new_df["Subject Sex"] = copy_df["Subject"].apply(lambda x: x[-1])

        for _, day_subdf in new_df.groupby(["Subject", "Date"]):
            new_df.loc[day_subdf.index, "Trial"] = np.arange(len(day_subdf)) 
        
        new_df.to_csv("TrialData_{}.csv".format(mode))

    elif mode == "lesions":
        copy_df = full_df.copy()
        copy_df = copy_df[
            (copy_df["LadderGroup"] == "PrelesionSet1") |
            (copy_df["LadderGroup"] == "PostlesionSet1") |
            (copy_df["LadderGroup"] == "PostlesionSet2")
        ].copy()
        display(copy_df.head())
        new_df = copy_df[["Subject", "Trial", "Date", "RT", "Reward"]].copy()

        # time column is implicitly there as the index

        # Now fill in a column to distinguish the sessions that occured after a 1 month break
        conditions = pd.Series(["None"] * len(new_df), index=new_df.index)

        conditions[
            (new_df["Subject"] == "GreBla7410M") &
            (new_df["Date"] >= datetime.date(2020, 1, 7)) &
            (new_df["Date"] <= datetime.date(2020, 1, 10))
        ] = "MonthLater"

        conditions[
            (new_df["Subject"] == "GreBla5671F") &
            (new_df["Date"] >= datetime.date(2020, 1, 7)) &
            (new_df["Date"] <= datetime.date(2020, 1, 10))
        ] = "MonthLater"

        new_df["Condition"] = conditions

        new_df["Interrupt"] = copy_df["Response"]
        new_df["Stimulus File"] = copy_df["Stimulus"].apply(lambda x: os.path.basename(x))
        new_df["Stimulus Vocalizer"] = copy_df["Bird Name"]
        new_df["Stimulus Call Type"] = copy_df["Call Type"]
        new_df["Stimulus Class"] = copy_df["Class"].apply(lambda x: "Rewarded" if x == "Rewarded" else "Nonrewarded")
        new_df["Informative Trials Seen"] = copy_df["Wait Count"].astype(np.int)
        new_df["Test Context"] = copy_df["test_folder"]
        new_df["Subject Sex"] = copy_df["Subject"].apply(lambda x: x[-1])
        new_df["Subject Group"] = copy_df["Lesion Group"]
        new_df["Ladder Group"] = copy_df["LadderGroup"]

        for _, day_subdf in new_df.groupby(["Subject", "Date"]):
            new_df.loc[day_subdf.index, "Trial"] = np.arange(len(day_subdf))

        new_df.to_csv("TrialData_{}.csv".format(mode))
