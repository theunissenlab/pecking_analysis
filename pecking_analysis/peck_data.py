
from __future__ import division
import pandas as pd
import numpy as np
import scipy.stats

def peck_data(blk, group1="Rewarded", group2="Unrewarded"):

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

def peck_data2(blk, group1="Rewarded", group2="Unrewarded"):

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