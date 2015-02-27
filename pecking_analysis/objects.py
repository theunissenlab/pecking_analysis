from __future__ import division, print_function
import numpy as np
import pandas as pd
import scipy.stats

class BaseOperant(object):

    _children = None

    def __init__(self, **kwargs):
        '''
        Base data class. All additional keyword arguments are stored as
        annotations.
        :param kwargs: additional annotations
        :return:
        '''

        self.annotate(**kwargs)
        self._format_options = dict(time_format="%H:%M:%S",
                                    date_format="%d/%m/%y",
                                    datetime_format="%m/%d/%y %H:%M:%S",
                                    )

    def annotate(self, **annotations):

        if not hasattr(self, "annotations"):
            self.annotations = dict()
        self.annotations.update(annotations)

    def save(self):
        pass

    def load(self):
        pass

    def plot(self):
        pass


class Bird(BaseOperant):
    '''
    This class holds all data about a particular bird and should link
    together all experiments in which this bird participated.
    TODO: BaseOperant._sort_children should sort children and return the list of children

    '''

    _children = "experiments"

    def __init__(self, name=None, age=None, sex=None, nickname=None, **kwargs):

        super(Bird, self).__init__(**kwargs)
        self.experiments = list()

        # Initial variables
        self.name = name
        self.age = None
        self.birthday = None
        self.sex = None
        self.nickname = None

    def __str__(self):

        return self.name


class Experiment(BaseOperant):
    '''
    This class contains all of the data for a single bird for a particular
    experiment. It summarizes details from all sessions of that experiment.
    '''

    _children = "sessions"

    # Properties
    num_sessions = property(fget=lambda x: len(x.sessions))

    def __init__(self, name= None, start_date=None,
                 end_date=None, fast_weight=None, fast_start=None,
                 **kwargs):

        super(Experiment, self).__init__(**kwargs)
        self.sessions = list()
        self.bird = bird
        self.start_date = start_date
        self.end_date = end_date
        self.fast_start = fast_start
        self.fast_weight = fast_weight


class Session(BaseOperant):
    '''
    This class groups together all data collected from an individual bird in
    a single session (usually one day). It summarizes data from multiple blocks.
    '''

    _children = "blocks"
    # Properties
    num_blocks = property(fget=lambda self: len(self.blocks))

    def __init__(self, name=None,
                 start=None, end=None,
                 pre_weight=None, post_weight=None,
                 box=None, seed_given=None,
                 notes=None, labels=None,
                 experiment=None, **kwargs):

        super(Session, self).__init__(**kwargs)

        self.name = name
        self.blocks = list()
        self.start = start
        self.end = end
        self.pre_weight = pre_weight
        self.box = box
        self.post_weight = post_weight
        self.seed_given = seed_given
        self.notes = notes
        self.labels = labels
        self.experiment = experiment


class Block(BaseOperant):
    '''
    This class organizes data from a single block of trials.
    '''

    _children = "trials"

    def __init__(self, name=None, date=None, start=None,
                 files=None, end=None, first_peck=None, data=None,
                 session=None, **kwargs):

        super(Block, self).__init__(**kwargs)
        self.trials = list()
        self.name = name
        self.date = date
        self.files = files
        self.start = start
        self.first_peck = first_peck
        self.end = end
        self.data = data
        self.session = session

class Trial(BaseOperant):
    '''
    This is the most precise class. It organizes data for a single stimulus
    presentation.
    '''

    def __init__(self, **kwargs):

        super(Trial, self).__init__(**kwargs)
