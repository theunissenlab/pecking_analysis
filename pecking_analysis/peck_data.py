#!/usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
import scipy.stats
from pecking_analysis.objects import Block

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

    ORDER = ["Trials", "Feeds", "P-Value"]

    if isinstance(blocks, Block):
        blocks = [blocks]

    output = pd.DataFrame()
    for blk in blocks:

        if blk.data is None:
            continue

        data = blk.data.copy()
        results = dict()
        results["Bird"] = blk.name
        results["Date"] = str(blk.date)
        results["Time"] = str(blk.start)

        # Get peck information
        total_pecks = len(blk.data)
        results["Trials"] = total_pecks

        grouped = blk.data.groupby("Class")
        total_group1 = grouped.size()[group1]
        results[group1[:2]] = total_group1
        total_group2 = grouped.size()[group2]
        results[group2[:2]] = total_group2
        total_feeds = blk.data["Reward"].sum()
        results["Feeds"] = total_feeds

        # Get percentages
        percent_group1 = total_group1 / total_pecks
        percent_group2 = total_group2 / total_pecks

        # Get interruption information
        if total_group2 > 0:
            interrupt_group2 = grouped["Response"].sum()[group2]
        else:
            interrupt_group2 = 0

        if total_group1 > 0:
            interrupt_group1 = grouped["Response"].sum()[group1]
        else:
            interrupt_group1 = 0

        total_responses = interrupt_group1 + interrupt_group2
        percent_interrupt = total_responses / total_pecks
        interrupt_group1 = interrupt_group1 / total_group1
        results["Intrpt %s" % group1[:2]] = interrupt_group1
        interrupt_group2 = interrupt_group2 / total_group2
        results["Intrpt %s" % group2[:2]] = interrupt_group2

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

        results["P-Value"] = binomial_pvalue
        results = pd.DataFrame(results, index=[0]).set_index(["Bird", "Date", "Time"])
        output = pd.concat([results, output])

    output = output[ORDER + [ss for ss in output.keys() if ss not in ORDER]]

    print(output)

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