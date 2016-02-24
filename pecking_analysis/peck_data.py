#!/usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
import scipy.stats

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

        if (blk.data is None) or (len(blk.data) == 0):
            continue

        total_pecks = total_group1 = total_group2 = total_feeds = \
        percent_group1 = percent_group2 = interrupt_group1 = \
        interrupt_group2 = 0

        data = blk.data.copy()
        results = dict()

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

        output = pd.concat([results, output])

    output = output.sort_index()
    print(output.to_string(float_format=lambda x: str(round(x, 3)), justify="left"))

    return output


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
