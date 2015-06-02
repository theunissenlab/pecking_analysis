from __future__ import print_function, division
import logging
from neosound.sound import *

logging.basicConfig()
logger = logging.getLogger()

class SoundProtocol(object):
    """
    Basic sound protocol class. The overall structure is: Create (or select) a sound manager with all of the sounds
    you'd like to use. Generate a list of filters to apply to the manager to select only the sounds you want from the manager. Create a set of preprocessors to apply to each sound pulled from the manager. Do whatever combination of sounds you'd like to generate individual stimuli. Create a list
    of postprocessings to apply to each of the generated stimuli.
    As methods:
    filter -> preprocess -> create_stimuli -> postprocess
    """
    def __init__(self, manager, filters, preprocessors, postprocessors):

        self.manager = manager
        self.filters = filters
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

        self.ids = None
        self.inputs = None
        self.outputs = None

    def filter(self):

        self.ids = None
        for filt in self.filters:
            self.ids = filt.apply(self.manager, self.ids)

    def switch_manager(self, manager, recursive=True):

        manager.import_ids(self.manager, self.ids, recursive=recursive)
        self.manager = manager

    def preprocess(self):

        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'initialize'):
                preprocessor.initialize(self.inputs)

            for ii, sound in enumerate(self.inputs):
                self.inputs[ii] = preprocessor.apply(sound)

    def postprocess(self):

        for postprocessor in self.postprocessors:
            if hasattr(postprocessor, 'initialize'):
                postprocessor.initialize(self.outputs)

            for ii, sound in enumerate(self.outputs):
                self.outputs[ii] = postprocessor.apply(sound)

    def run(self):

        self.filter()
        self.preprocess()
        self.create_stimuli()
        self.postprocess()

    def create_stimuli(self):

        self.outputs = self.inputs

    def write(self):

        pass

    def write_protocol(self):

        pass


class SoundFilter(object):

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def apply(self, manager, ids):

        return ids


class BirdFilter(SoundFilter):

    def __init__(self, birds=None):
        """ Filter a list of ids by bird
    name. birds should be a list of names to be included. If birds is None, any
    sound with the bird annotation will be accepted.
        """
        if birds is None:
            birds = []

        self.birds = birds

    def apply(self, manager, ids):

        if len(self.birds):
            filter_func = lambda bn: bn in self.birds
        else:
            filter_func = lambda bn: True

        return manager.database.filter_by_func(bird=filter_func,
                                               ids=ids)


class CallTypeFilter(SoundFilter):

    def __init__(self, types=None):
        """ Filter a list of ids by call type. types should be a list of call types to
be included. If types is None, any sound with a call_type annotation will be
accepted.
        """

        if types is None:
            types = []

        self.types = types

    def apply(self, manager, ids):

        if len(self.types):
            filter_func = lambda ct: ct in self.types
        else:
            filter_func = lambda ct: True

        return manager.database.filter_by_func(call_type=filter_func,
                                               ids=ids)
