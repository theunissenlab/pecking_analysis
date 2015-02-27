from __future__ import division
import pandas as pd
import numpy as np
import scipy.stats

def peck_data(blk):

    if blk.data is None:
        return

    print("Computing statistics for block %s" % blk)
    # Get peck information
    total_pecks = len(blk.data)
    grouped = blk.data.groupby("Class")
    total_reward = grouped.size()["Rewarded"]
    total_no_reward = grouped.size()["Unrewarded"]
    total_feeds = blk.data["Reward"].sum()
    print("Bird underwent %d trials and fed %d times" % (total_pecks, total_feeds))

    # Get percentages
    percent_reward = total_reward / total_pecks
    percent_no_reward = total_no_reward / total_pecks
    print("Rewarded stimuli: %d (%2.1f%%), Unrewarded stimuli: %d (%2.1f%%)", (total_reward, percent_reward, total_no_reward, percent_no_reward))

    # Get interruption information
    if total_no_reward > 0:
        interrupt_no_reward = grouped["Response"].sum()["Unrewarded"]
    else:
        interrupt_no_reward = 0

    if total_reward > 0:
        interrupt_reward = grouped["Response"].sum()["Rewarded"]
    else:
        interrupt_reward = 0

    total_responses = interrupt_reward + interrupt_no_reward
    percent_interrupt = total_responses / total_pecks
    interrupt_no_reward = interrupt_no_reward / total_no_reward
    interrupt_reward = interrupt_reward / total_reward
    print("%d interruptions: %2.1f%% rewarded, %2.1f%% unrewarded" % (total_responses, interrupt_reward, interrupt_no_reward))

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
