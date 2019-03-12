#!/usr/bin/env python
from __future__ import division
import os
import glob
import pandas as pd
import datetime
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import re

from pecking_analysis.stimuli import preprocess_stimuli, insert_stimulus_history
from pecking_analysis.objects import merge_daily_blocks
from pecking_analysis.importer import PythonCSV


def get_dates(directory):
    dates = [os.path.basename(date) for date in glob.glob(os.path.join(directory, "*"))]
    dates = [datetime.date(int(date[-2:]), int(date[2:-2]), int(date[:2])) for date in dates
            if re.search(r"^[0-9]{6}$", date)]

    return sorted(dates)


def load_pecking_days(directory, conditions=("Rewarded", "Unrewarded")):
    file_list = []

    if re.search("^[0-9]{6}$", os.path.basename(directory)):
        csvs = glob.glob(os.path.join(directory, "*.csv"))
        for csv_file in csvs:
            if os.path.getsize(csv_file) < 500:
                # don't load empty or tiny csv files
                continue
            file_list.append(csv_file)
    else:
        for date in get_dates(directory):
            date_folder = os.path.join(directory, date.strftime("%d%m%y"))
            csvs = glob.glob(os.path.join(date_folder, "*.csv"))

            for csv_file in csvs:
                if os.path.getsize(csv_file) < 500:
                    # don't load empty or tiny csv files
                    continue
                file_list.append(csv_file)

    blocks = PythonCSV.parse(file_list)
    blocks = merge_daily_blocks(blocks)

    for block in blocks:
        block.filter_conditions(conditions)

    stim_blocks = []
    for block in blocks:
        if not len(block.data):
            continue
        block, stims = preprocess_stimuli(block)
        stim_blocks.append(stims)

    stim_blocks = insert_stimulus_history(stim_blocks)

    for block, stim_df in zip(blocks, stim_blocks):
        if not len(block.data):
            continue
        block.reject_double_pecks(200)
        block.data["Trial Number"] = pd.Series(np.arange(len(block.data)))

        # join the finalized block to its stimuli
        new_df = block.data.join(
            stim_df.set_index(["Stim Key"])[["New"]],
            on=["Stim Key"],
            rsuffix="_stim"
        )
        block.data = new_df

    return blocks, stim_blocks


def get_labels_by_combining_columns(block, columns, combiner=None):
    if len(columns) > 1:
        labels = block.data[columns].apply(combiner, axis=1)
    elif len(columns) == 1:
        labels = block.data[columns[0]]

    return labels


class color_by_reward(object):
    @staticmethod
    def get(x):
        if "Rewarded" in x:
            return "Blue"
        else:
            return "Red"


def plot_data(block, labels, index_by="time", label_order=None, label_to_color=None):
    """Plot the data organized by given labels

    Parameters
    ----------
    block : pecking_analysis.objects.Block
        block object containing the data for one session
    labels : pandas.Series
        series of same length as block containing the string labels
        for each trial
    index_by : str ("time" or "trial")
        how to plot data, by trial time or by trial index
    label_order : function
        function to order the labels when plotting / assigning colors
    label_to_color : function
        dictionary or object mapping the label to color to plot,
        must implement a .get(label) method returning color
    """
    unique_labels = labels.unique()
    if label_order is not None:
        unique_labels = sorted(unique_labels, key=label_order)
    else:
        unique_labels = sorted(unique_labels)

    n_categories = len(unique_labels)
    
    fig = plt.figure(facecolor="white", edgecolor="white", figsize=(10, 4 + 0.5 * n_categories))
    
    events_ax = fig.gca()
    prob_ax = events_ax.twinx()
    
    events_ax.set_ylim(-0.2 - 0.1 * n_categories, 1.2 + 0.1 * n_categories)
    prob_ax.set_ylim(-0.2 - 0.1 * n_categories, 1.2 + 0.1 * n_categories)

    if label_to_color is None:
        label_to_color = {}

    old_index = block.data.index
    # if index_by == "time":
        # pass
    # else:
        # block.data.index = pd.Series(np.arange(len(block.data)))
    
    for label_idx, label in enumerate(unique_labels):
        label_df = block.data[labels == label]
        
        # Polarity signal (1 if pecked, -1 if not)
        flip = label_df["Response"].apply(lambda x: 1 if x else -1)
        # Binary signal (1 if pecked, 0 if not)
        binary = label_df["Response"].apply(lambda x: 1 if x else 0)
        
        # Plot events scatter
        scat = events_ax.scatter(label_df.index, 
            ((1.0 * binary) + (0.2 * flip)) + flip * 0.1 * label_idx * np.ones((len(label_df.index))),
            s=50, 
            marker="|",
            color=label_to_color.get(label),
            label=label
        )
        
        events_ax.vlines(x=0, ymin=1.1, ymax=1.3 + 0.1 * n_categories, color='black', linewidth=2)
        events_ax.vlines(x=0, ymin=-0.3 - 0.1 * n_categories, ymax=-0.1, color='black', linewidth=2)
        
        if len(label_df["Response"]) > 30:
            win_size = 20
        elif len(label_df["Response"]) > 15:
            win_size = 10
        else:
            win_size = 4
        win_size_half = win_size // 2
        rolled = label_df["Response"].rolling(win_size, center=True).mean()
        if len(rolled) > win_size_half:
            rolled.iloc[:win_size_half] = rolled.iloc[win_size_half]
            rolled.iloc[-win_size_half:] = rolled.iloc[-win_size_half - 1]

        prob_ax.plot(
            label_df.index,
            rolled,
            label=scat.get_label(),
            alpha=0.5,
            linewidth=3,
            color=scat.get_edgecolor()[0],
        )

    events_ax.hlines([-0.01, 1.01], *events_ax.get_xlim(), linewidth=2, linestyle=":", color="Grey")
    events_ax.fill_between(events_ax.get_xlim(), [-0.01, -0.01], [1.01, 1.01], color="0.95", zorder=0)

    # if index_by == "time":
        # prob_ax.set_xlim(block.data.index[0], block.data.index[-1])
    # else:
    prob_ax.set_xlim(0, len(block.data))
    events_ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.35 - 0.1 * n_categories),
        bbox_transform=events_ax.transData,
        fontsize=12, ncol=2)
    events_ax.xaxis.set_tick_params(labelsize=14)
    events_ax.set_yticks([0, 1])
    events_ax.set_yticklabels([0.0, 1.0], size=14)
    events_ax.set_ylabel("Prob.\ninterrupt", fontsize=14)
    events_ax.set_xticks([])
    events_ax.set_xlabel("Trial", fontsize=14)
    prob_ax.set_yticks([])
    events_ax.spines['top'].set_visible(False)
    events_ax.spines['right'].set_visible(False)
    events_ax.spines['bottom'].set_visible(False)
    events_ax.spines['left'].set_visible(False)
    prob_ax.spines['top'].set_visible(False)
    prob_ax.spines['right'].set_visible(False)
    prob_ax.spines['bottom'].set_visible(False)
    prob_ax.spines['left'].set_visible(False)

    events_ax.set_yticks([0.2, 0.4, 0.6, 0.8], minor=True)
    events_ax.grid(which='minor', alpha=0.8, linestyle=":")

    events_ax.text(0, 1.15 + 0.05 * n_categories, "Int.  ", fontsize=14, horizontalalignment="right", verticalalignment="center")
    events_ax.text(0, -0.15 - 0.05 * n_categories, "Wait  ",  fontsize=14, horizontalalignment="right", verticalalignment="center")

    events_ax.vlines(x=0, ymin=0, ymax=1, color='black', linewidth=2)
    
    block.data.index = old_index

    return fig


def peck_data_old(blk, group1="Rewarded", group2="Unrewarded"):

    if blk.data is None:
        return

    print("Computing statistics for %s" % blk.name)
    # Get peck information
    total_pecks = len(blk.data)
    grouped = blk.data.groupby("Class")
    total_reward = grouped.size()[group1]
    total_no_reward = grouped.size()[group2]
    total_feeds = blk.data["Reward"].sum()
    print("Bird underwent %d trials and fed %d times" % (total_pecks, total_feeds))

    # Get percentages
    percent_reward = total_reward / total_pecks
    percent_no_reward = total_no_reward / total_pecks
    print("Rewarded stimuli: %d (%2.1f%%), Unrewarded stimuli: %d (%2.1f%%)" % (total_reward, 100 * percent_reward, total_no_reward, 100 * percent_no_reward))

    # Get interruption information
    if total_no_reward > 0:
        interrupt_no_reward = grouped["Response"].sum()[group2]
    else:
        interrupt_no_reward = 0

    if total_reward > 0:
        interrupt_reward = grouped["Response"].sum()[group1]
    else:
        interrupt_reward = 0

    total_responses = interrupt_reward + interrupt_no_reward
    percent_interrupt = total_responses / total_pecks
    interrupt_no_reward = interrupt_no_reward / total_no_reward
    interrupt_reward = interrupt_reward / total_reward
    print("%d interruptions: %2.1f%% of rewarded, %2.1f%% of unrewarded" % (total_responses, 100 * interrupt_reward, 100 * interrupt_no_reward))

    if (total_reward > 0) and (total_no_reward > 0):
        mu = (interrupt_no_reward - interrupt_reward)
        sigma = np.sqrt(percent_interrupt * (1 - percent_interrupt) * (1 / total_reward + 1 / total_no_reward))
        zscore = mu / sigma
        binomial_pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(zscore)))
        is_significant = binomial_pvalue <= 0.05
    else:
        zscore = 0.0
        binomial_pvalue = 1.0
        is_significant = False

    print("ZScore = %3.2f, PValue = %3.2e, %s" % (zscore,
                                                  binomial_pvalue,
                                                  "Significant" if is_significant else "Not significant"))

    return dict(total_pecks=total_pecks,
                total_reward=total_reward,
                total_no_reward=total_no_reward,
                percent_reward=percent_reward,
                percent_no_reward=percent_no_reward,
                total_responses=total_responses,
                interrupt_no_reward=interrupt_no_reward,
                interrupt_reward=interrupt_reward,
                zscore=zscore,
                binomial_pvalue=binomial_pvalue,
                is_significant=is_significant)


def peck_data(blocks, group1="Rewarded", group2="Unrewarded"):
    """
    Computes some basic statistics for each block and compares groups 1 and 2 with a binomial test.
    :param blocks: a list of Block objects
    :param group1: First group to compare
    :param group2: Second group to compare
    :return: A pandas dataframe
    TODO: Add pecks and feeds in case where both groups were not seen
    """

    if not isinstance(blocks, list):
        blocks = [blocks]

    output = pd.DataFrame()
    for blk in blocks:
        blk.filter_conditions([group1, group2])

        # Initialize my variables
        results = dict()

        total_pecks = total_group1 = total_group2 = total_feeds = np.nan
        percent_group1 = percent_group2 = np.nan
        interrupt_group1 = interrupt_group2 = percent_interrupt = np.nan
        zscore = binomial_pvalue = np.nan

        if (blk.data is not None) and (len(blk.data) > 0):
            data = blk.data.copy()

            # Get peck information
            total_pecks = len(blk.data)
            total_feeds = blk.data["Reward"].sum()
            total_responses = blk.data["Response"].sum()

            # Collect group statistics
            if total_pecks > 0:
                percent_interrupt = total_responses / total_pecks

                group1_data = blk.data[blk.data["Class"] == group1]
                total_group1 = len(group1_data)
                percent_group1 = total_group1 / total_pecks
                if total_group1 > 0:
                    interrupt_group1 = group1_data["Response"].sum() / total_group1

                group2_data = blk.data[blk.data["Class"] == group2]
                total_group2 = len(group2_data)
                percent_group2 = total_group2 / total_pecks
                if total_group2 > 0:
                    interrupt_group2 = group2_data["Response"].sum() / total_group2

            # Compare the two groups
            if (total_group1 > 0) and (total_group2 > 0):
                mu = (interrupt_group2 - interrupt_group1)
                sigma = np.sqrt(percent_interrupt * (1 - percent_interrupt) * (1 / total_group1 + 1 / total_group2))
                zscore = mu / sigma
                binomial_pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(zscore)))
                is_significant = binomial_pvalue <= 0.05
            else:
                zscore = 0.0
                binomial_pvalue = 1.0
                is_significant = False

        results[("Total", group1)] = total_group1
        results[("Total", "Trials")] = total_pecks
        results[("Total", group2)] = total_group2
        results[("Total", "Feeds")] = total_feeds
        results[("Percent", group1)] = percent_group1
        results[("Percent", group2)] = percent_group2
        results[("Interrupt", "Total")] = percent_interrupt
        results[("Interrupt", group1)] = interrupt_group1
        results[("Interrupt", group2)] = interrupt_group2
        results[("Stats", "Z-Score")] = zscore
        results[("Stats", "P-Value")] = binomial_pvalue
        results = pd.DataFrame(results, index=[0])

        results["Bird"] = str(getattr(blk, "name", None))
        results["Date"] = str(getattr(blk, "date", None))
        results["Time"] = str(getattr(blk, "start", None))
        results = results.set_index(["Bird", "Date", "Time"])

        output = pd.concat([output, results])

    print(output.sort_index().to_string(float_format=lambda x: str(round(x, 3)), justify="left"))

    return output


def summarize_blocks(blocks):
    """ Get the number of blocks and the number of pecks per block """

    blocks = [blk for blk in blocks if len(blk.data) > 0]
    performance = peck_data(blocks).reset_index()
    grouped = performance.groupby("Bird")
    birds = grouped.groups.keys()
    summary = pd.DataFrame([], columns=["Blocks", "Pecks"], index=birds)
    for bird, group in grouped:
        summary.loc[bird]["Blocks"] = len(group)
        summary.loc[bird]["Pecks"] = group["Total"]["Trials"].mean()

    return summary.sort_index()


if __name__ == "__main__":
    import argparse
    import os
    from pecking_analysis.importer import PythonCSV

    parser = argparse.ArgumentParser(description="Run peck_data on a list of csv files")
    parser.add_argument("csv_files",
                        help="A list of CSV files separated by spaces",
                        nargs="+")

    args = parser.parse_args()
    csv_files = list()
    for cf in args.csv_files:
        filename = os.path.abspath(os.path.expanduser(cf))
        if not os.path.exists(filename):
            IOError("File %s does not exist!" % filename)
        csv_files.append(filename)

    blocks = PythonCSV.parse(csv_files)
    for blk in blocks:
        peck_data(blk)
