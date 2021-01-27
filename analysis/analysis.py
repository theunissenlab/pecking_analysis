from typing import List

import numpy as np
import pandas as pd

from analysis.stats import compute_odds_ratio


def create_informative_trials_column(
        df: pd.DataFrame,
        as_column: str="Wait Index",
        group_by: List[str]=("Subject", "Stimulus Vocalizer", "Stimulus Call Type", "Stimulus Class"),
        in_place: bool=False,
        ):
    """Create a column showing the informative trial index of each stim relative to the start of the experiment

    By default, groups trials by Subject, Stimulus Vocalizer, Stimulus Call Type, and Stimulus Class (rewarded/nonrewarded)

    To do informative trial analysis for a given block (e.g. the informative trials since the start of the last day of testing)
    these need to be shifted by subtracting off the "Wait Index" of the first trial prior to the requested range.

    The normal procedure to do this would be to create a slice of the dataframe you would like to analyze and then
    call create_informative_trials_column() on it.
    """
    group_by = list(group_by)

    if not in_place:
        df = df.copy()

    df[as_column] = np.nan
    df[as_column] = df[as_column].astype(pd.Int64Dtype())
    # Loop over each stimulus, and count the number of non-interrupted trials that have been seen previously
    for _, sub_df in df.groupby(group_by):
        waits_seen = 0
        for idx in sub_df.index:
            df.loc[idx, as_column] = waits_seen
            if sub_df.loc[idx, "Interrupt"] == False:
                waits_seen += 1

    return df


def summarize_data(df: pd.DataFrame):
    """Summarizes data showing interrupts, feeds, and p-values
    """
    output = pd.DataFrame()
    for subject, subject_df in df.groupby("Subject"):
        all_trials = subject_df
        rewarded_trials = subject_df[subject_df["Stimulus Class"] == "Rewarded"]
        nonrewarded_trials = subject_df[subject_df["Stimulus Class"] == "Nonrewarded"]

        odds_ratio, odds_ratio_95_ci, p_value = compute_odds_ratio(
            nonrewarded_trials,
            rewarded_trials,
            "Interrupt",
            side="greater"
        )

        results = {}

        results[("Feeds", "Feeds")] = int(all_trials["Reward"].sum())
        results[("Trials", "Re")] = len(rewarded_trials)
        results[("Trials", "NonRe")] = len(nonrewarded_trials)
        results[("Trials", "All")] = len(all_trials)
        results[("Percent", "Re")] = len(rewarded_trials) / len(all_trials)
        results[("Percent", "NonRe")] = len(nonrewarded_trials) / len(all_trials)
        results[("Interrupt", "Re")] = rewarded_trials["Interrupt"].mean()
        results[("Interrupt", "NonRe")] = nonrewarded_trials["Interrupt"].mean()
        results[("Interrupt", "All")] = all_trials["Interrupt"].mean()
        results[("Stats", "OR")] = odds_ratio
        results[("Stats", "95% CI")] = "({:.1f}, {:.1f})".format(*odds_ratio_95_ci)
        results[("Stats", "P-Value")] = p_value
        results = pd.DataFrame(results, index=[0])

        results["Bird"] = subject
        results["Date"] = subject_df.iloc[0]["Date"]

        results = results.set_index(["Bird", "Date"])

        output = pd.concat([output, results])

    return output.sort_index()
