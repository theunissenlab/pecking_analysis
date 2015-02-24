from __future__ import division, print_function
import numpy as np
import pandas as pd
import scipy.stats
from cached_property import cached_property

class BaseOperant(object):

    def __init__(self, **kwargs):
        '''
        Base data class. All additional keyword arguments are stored as
        annotations.
        :param kwargs: additional annotations
        :return:
        '''

        self.children = None
        self.annotations = kwargs

        self._format_options = dict(time_format="%H:%M:%S",
                                    date_format="%d/%m/%y",
                                    datetime_format="%m/%d/%y %H:%M:%S",
                                    )

    def annotate(self, **annotations):

        self.annotations.update(annotations)

    def save(self):
        pass

    def load(self):
        pass

    def plot(self):
        pass

    def summarize(self):
        '''
        Loops through all of the object's children and concatenates their
        summary data.
        :return: pandas DataFrame with concatenated summary data from all
        children
        '''
        if hasattr(self, self.children):
            value_list = list()
            for child in getattr(self, self.children):
                if hasattr(child, "_summary"):
                    columns, values = child._summary()
                    value_list.append(values)
            if len(value_list):
                return pd.DataFrame(value_list, columns=columns)

    def update(self):
        '''
        Delete
        '''
        cached = getattr(self, "_cached", [])
        for var in cached:
            if hasattr(self, var):
                delattr(self, var)

        parent = getattr(self, "_parent", None)
        if parent is not None:
            parent.update()

    def _sort_children(self):
        pass

    # This should go in the bird manager and handle objs of different types,
    # skipping objects that aren't Bird, Experiment, Session, Block, or Trial.
    # @staticmethod
    # def summarize(objs):
    #     # This is hacky and will fail if all objs are not of the same type
    #
    #     if len(objs):
    #         value_list = list()
    #         for obj in objs:
    #             columns, values = obj.summary()
    #             value_list.append(values)
    #         if len(value_list):
    #             return pd.DataFrame(value_list, columns=columns)
    #
    #     else:
    #         columns, values = objs.summary()
    #         return pd.DataFrame([values], columns=columns)


class Bird(BaseOperant):
    '''
    This class holds all data about a particular bird and should link
    together all experiments in which this bird participated.
    TODO: BaseOperant._sort_children should sort children and return the list of children

    '''

    _cached = ["num_experiments",
               "last_experiment",
               "average_weight",
               "average_peck_count",
               "average_performance",
               "significant_blocks",
               ]
    _children = "experiments"

    def __init__(self, name=None, **kwargs):

        super(Bird, self).__init__(**kwargs)
        self.experiments = list()

        # Initial variables
        self.name = name
        self.age = None
        self.sex = None
        self.nickname = None

        # Computed variables

    def __str__(self):

        return self.name

    def _summary(self):

        columns = ["Name", "Experiments",
                   "Avg. Pecks", "% Significant Blocks"]
        values = [self.name, self.num_experiments,
                  self.pecks_per_block, self.significant_blocks]

        return columns, values

    @cached_property
    def num_experiments(self):

        return len(self.experiments)

    @cached_property
    def last_experiment(self):

        if self.num_experiments > 0:
            self._sort_children()
            return self.experiments[-1]

    @cached_property
    def average_weight(self):

        return np.mean([exp.fast_weight for exp in self.experiments])

    @cached_property
    def average_peck_count(self):

        return np.mean([blk.num_trials for exp in self.experiments for
                        sess in exp.sessions for blk in sess.blocks])

    @cached_property
    def average_performance(self):

        return np.mean([blk.performance for exp in self.experiments for
                        sess in exp.sessions for blk in sess.blocks])

    @cached_property
    def significant_blocks(self):

        return 100 * np.mean([blk.is_significant for exp in
                              self.experiments for sess in exp.sessions
                              for blk in sess.blocks])

class Experiment(BaseOperant):
    '''
    This class contains all of the data for a single bird for a particular
    experiment. It summarizes details from all sessions of that experiment.
    '''

    # Properties
    num_sessions = property(fget=lambda x: len(x.sessions))

    def __init__(self, name= None, start=None,
                 end=None, weight=None, fast_start=None,
                 **kwargs):

        super(Experiment, self).__init__(**kwargs)
        self.children = "sessions"
        self.sessions = list()
        self.bird = None
        self.start_date = start
        self.end_date = end
        self.fast_start = fast_start
        self.weight = weight

    def summary(self):

        columns = ["Start", "End", "Weight", "Sessions", "Blocks",
                   "Avg. Pecks", "Signficant Blocks"]
        if self.end is not None:
            end = self.end.strftime(self.date_format)
        else:
            end = None
        values = [self.start.strftime(self.date_format),
                  end,
                  self.weight,
                  self.num_sessions, self.num_blocks,
                  self.total_pecks / float(self.num_blocks),
                  self.significant_blocks]

        return columns, values

    @property
    def num_blocks(self):

        num_blocks = 0
        for session in self.sessions:
            num_blocks += session.num_blocks

        return num_blocks

    @property
    def total_pecks(self):

        total_pecks = 0
        for session in self.sessions:
            total_pecks += session.total_pecks

        return total_pecks

    @property
    def significant_blocks(self):

        significant_blocks = 0
        for session in self.sessions:
            significant_blocks += session.significant_blocks

        return significant_blocks

    def show_weight(self, show_plot=True):

        weights = dict()
        if self.weight is not None:
            weights.setdefault("Time", list()).append(self.fast_start)
            weights.setdefault("Weight", list()).append(self.weight)
            weights.setdefault("Label", list()).append("Fast start")
            weights.setdefault("Pecks", list()).append(None)
            weights.setdefault("Num Reward", list()).append(None)
            weights.setdefault("Food Given", list()).append(None)

        for sess in self.sessions:
            if sess.start is not None:
                weights.setdefault("Time", list()).append(sess.start)
                weights.setdefault("Weight", list()).append(sess.weight)
                weights.setdefault("Label", list()).append("Session start")
                weights.setdefault("Pecks", list()).append(None)
                weights.setdefault("Num Reward", list()).append(None)
                weights.setdefault("Food Given", list()).append(None)

            if sess.end is not None:
                weights.setdefault("Time", list()).append(sess.end)
                weights.setdefault("Weight", list()).append(sess.post_weight)
                weights.setdefault("Label", list()).append("Session end")
                weights.setdefault("Pecks", list()).append(sess.total_pecks)
                weights.setdefault("Num Reward", list()).append(sess.total_reward)
                weights.setdefault("Food Given", list()).append(sess.seed_given)

        return weights

        index = weights.pop("Time")
        df = pd.DataFrame(weights, index=index)
        df["Weight"][df["Weight"] == 0] = None
        if show_plot:
            ts = df["Weight"]
            ts.index = df.index.format(formatter=lambda x: x.strftime(
                self.date_format))
            ax = ts.plot(rot=0)
            ax.set_xticks(range(len(ts)))
            ax.set_xticklabels(ts.index)
            ax.set_ylabel("Weight")
            ax.set_xlabel("Date")

            ax2 = ax.twinx()
            ts = df["Pecks"][~np.isnan(df["Pecks"])]
            ts.index = df.index.format(formatter=lambda x: x.strftime(
                self.date_format))
            ts.plot(rot=0, ax=ax2)


        return df


class Session(BaseOperant):
    '''
    This class groups together all data collected from an individual bird in
    a single session (usually one day). It summarizes data from multiple blocks.
    '''

    # Properties
    num_blocks = property(fget=lambda self: len(self.blocks))

    def __init__(self, name=None,
                 start=None, end=None,
                 weight=None, post_weight=None,
                 box=None, seed_given=None,
                 notes=None, labels=None,
                 **kwargs):

        super(Session, self).__init__(**kwargs)
        self.children = "blocks"

        self.name = name
        self.blocks = list()
        self.start = start
        self.end = end
        self.weight = weight
        self.box = box
        self.post_weight = post_weight
        self.seed_given = seed_given
        self.notes = notes
        self.labels = labels
        self.experiment = None

    def summary(self):

        columns = ["Start", "Initial Weight", "Final Weight",
                   "Stimulus Label", "Total Pecks", "# Blocks",
                   "Significant blocks"]
        values = [self.start.strftime(self.datetime_format),
                  self.weight,
                  self.post_weight,
                  self.labels,
                  self.total_pecks,
                  self.num_blocks,
                  self.significant_blocks]

        return columns, values

    @property
    def total_pecks(self):

        total_pecks = 0
        for blk in self.blocks:
            total_pecks += blk.total_pecks

        return total_pecks

    @property
    def significant_blocks(self):

        significant_blocks = 0
        for blk in self.blocks:
            if blk.is_significant:
                significant_blocks += 1

        return significant_blocks

    @property
    def total_reward(self):

        total_reward = 0
        for blk in self.blocks:
            try:
                total_reward += blk.total_reward
            except AttributeError:
                continue

        return total_reward

    @property
    def weight_loss(self):

        if self.experiment is not None:
            if (self.weight is not None) and (self.experiment.weight is not None):
                return 100 * (1 - float(self.weight) / float(self.experiment.weight))


    def to_csv(self, filename):

        pd.concat([blk.data for blk in self.blocks]).to_csv(filename)


class Block(BaseOperant):
    '''
    This class organizes data from a single block of trials.
    '''

    # Properties
    is_complete = property(fget=lambda self: self.end is not None)
    num_trials = property(fget=lambda self: len(self.trials))

    def __init__(self, name=None, start=None,
                 files=list(), end=None, first_peck=None,
                 **kwargs):

        super(Block, self).__init__(**kwargs)
        self.children = "trials"
        self.trials = list()
        self.name = name
        self.date = None
        self.files = files
        self.start = start
        self.first_peck = first_peck
        self.end = end
        self.data = None
        self.session = None

    def summary(self):

        if not hasattr(self, "total_pecks"):
            self.compute_statistics()

        columns = ["Start Time", "End Time", "Total Pecks",
                   "% No Re Interrupts", "% Re Interrupts",
                   "P-Value"]

        values = [self.start.strftime(self.time_format),
                  self.end.strftime(self.time_format),
                  self.total_pecks,
                  100 * self.percent_interrupt_no_reward,
                  100 * self.percent_interrupt_reward,
                  self.binomial_pvalue]

        return columns, values

    def compute_statistics(self):

        if self.data is None:
            return

        # Get peck information
        self.total_pecks = len(self.data)
        self.total_stimuli = self.total_pecks
        self.total_reward = self.data["Class"].sum()
        self.total_no_reward = self.total_pecks - self.total_reward

        # Get percentages
        self.percent_reward = self.to_percent(self.total_reward, self.total_stimuli)
        self.percent_no_reward = self.to_percent(self.total_no_reward, self.total_stimuli)

        # Add interval and interrupt data to table
        self.get_interrupts()

        grped = self.data.groupby("Class")

        # Get interruption information
        if self.total_no_reward > 0:
            self.interrupt_no_reward = grped["Interrupts"].sum()[0]
        else:
            self.interrupt_no_reward = 0
        self.percent_interrupt_no_reward = self.to_percent(self.interrupt_no_reward, self.total_no_reward)

        if self.total_reward > 0:
            self.interrupt_reward = grped["Interrupts"].sum()[1]
        else:
            self.interrupt_reward = 0
        self.percent_interrupt_reward = self.to_percent(self.interrupt_reward, self.total_reward)

        self.total_interrupt = self.interrupt_reward + self.interrupt_no_reward
        self.percent_interrupt = self.to_percent(self.total_interrupt, self.total_pecks)

        if (self.total_reward > 0) and (self.total_no_reward > 0):
            mu = (self.percent_interrupt_no_reward - self.percent_interrupt_reward)
            sigma = np.sqrt(self.percent_interrupt * (1 - self.percent_interrupt) * (1 / self.total_reward + 1 / self.total_no_reward))
            self.zscore = mu / sigma
            self.binomial_pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(self.zscore)))
            self.is_significant = self.binomial_pvalue <= 0.05
        else:
            self.zscore = 0.0
            self.binomial_pvalue = 1.0
            self.is_significant = False

    @staticmethod
    def to_percent(value, n):

        if n != 0:
            return value / float(n)
        else:
            return 0.0

    def get_interrupts(self):

        if "Interrupts" in self.data:
            return

        time_diff = np.hstack([np.diff([ind.value for ind in self.data.index]), 0]) / 10**9
        inds = (time_diff > 0.19) & (time_diff < 6) # This value should be based on the stimulus duration

        self.data["Interrupts"] = inds
        self.data["Intervals"] = time_diff


class Trial(BaseOperant):
    '''
    This is the most precise class. It organizes data for a single stimulus
    presentation.
    '''

    def __init__(self, **kwargs):

        super(Trial, self).__init__(**kwargs)
