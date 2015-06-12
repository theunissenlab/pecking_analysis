from __future__ import print_function, division
import logging
from neosound.sound import *
from zeebeez.sound import utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

class SoundProtocol(object):
    """
    Basic sound protocol class. The overall structure is: Create (or select) a sound manager with all of the sounds
    you'd like to use. Generate a list of filters to apply to the manager to select only the sounds you want from the manager. Create a set of preprocessors to apply to each sound pulled from the manager. Do whatever combination of sounds you'd like to generate individual stimuli. Create a list
    of postprocessings to apply to each of the generated stimuli.
    As methods:
    filter -> preprocess -> create_stimuli -> postprocess
    """
    def __init__(self, manager,
                 filters=None,
                 preprocessors=None,
                 postprocessors=None,
                 output_manager=None,
                 log_file=None):

        self.manager = manager
        # Make sure that the original database is set to read_only
        self.manager.database.read_only = True

        if filters is None:
            filters = []
        self.filters = filters

        if preprocessors is None:
            preprocessors = []
        self.preprocessors = preprocessors

        if postprocessors is None:
            postprocessors = []
        self.postprocessors = postprocessors

        if output_manager is None:
            output_manager = SoundManager(DictStore)
        self.output_manager = output_manager

        if log_file is not None:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
            logger.addHandler(logging.FileHandler(log_file))

        self.ids = None
        self.inputs = None
        self.outputs = None

    def filter(self):

        ids = list()
        for filt in self.filters:
            logger.debug("Applying filter %s" % filt)
            ids.extend(filt.apply(self.manager))
        self.ids = dict((id_, None) for id_ in ids).keys() # Get only the unique ids

    def switch_manager(self, manager, recursive=True):

        # TODO: It might be best to only transfer those that are used in the create_stimuli method?
        logger.debug("Switching manager from %s to %s" % (self.manager,
                                                          manager))
        self.ids = manager.import_ids(self.manager, self.ids, recursive=recursive)
        self.manager = manager

    def preprocess(self):

        logger.debug("Reconstructing sounds from ids")
        self.inputs = [self.manager.reconstruct(id_) for id_ in self.ids]
        annotations = [s.annotations for s in self.inputs]
        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'initialize'):
                logger.debug("Initializing preprocessor %s" % preprocessor)
                preprocessor.initialize(self.inputs)

            logger.debug("Applying preprocessor %s" % preprocessor)
            for ii, sound in enumerate(self.inputs):
                self.inputs[ii] = preprocessor.apply(sound)

        logger.debug("Copying annotations to preprocessed inputs")
        for ii, annot in enumerate(annotations):
            self.inputs[ii].annotate(**annot)

    def postprocess(self):

        for postprocessor in self.postprocessors:
            if hasattr(postprocessor, 'initialize'):
                logger.debug("Initializing postprocessor %s" % postprocessor)
                postprocessor.initialize(self.outputs)

            logger.debug("Applying postprocessor %s" % postprocessor)
            for ii, sound in enumerate(self.outputs):
                self.outputs[ii] = postprocessor.apply(sound)

    def run(self, num_stimuli):

        self.filter()
        self.switch_manager(self.output_manager)
        self.preprocess()
        self.create_stimuli(num_stimuli)
        self.postprocess()

    def create_stimuli(self, num_stimuli):

        logger.debug("Copying inputs to outputs")
        self.outputs = self.inputs[:num_stimuli]

    def write_wavfiles(self, directory):

        if not os.path.exists(directory):
            os.mkdir(directory)

        for ii, ss in enumerate(self.outputs):
            filename = os.path.join(directory, "stimulus_%d.wav" % ii)
            logger.debug("Writing output wavefile to %s" % filename)
            ss.save(filename)

    def write(self):

        pass

    def write_protocol(self):

        pass


class SoundFilter(object):

    def __init__(self, *args, **kwargs):

        self.num_matches = kwargs.get("num_matches", None)
        self.matches = None

    def apply(self, manager, ids):

        if self.num_matches is not None:
            ids = ids[:self.num_matches]
        logger.debug("Found %d ids" % len(ids))
        self.matches = ids

        return self.matches


class BirdFilter(SoundFilter):

    def __init__(self, birds=None, **kwargs):
        """ Filter a list of ids by bird
    name. birds should be a list of names to be included. If birds is None, any
    sound with the bird annotation will be accepted.
        """
        super(BirdFilter, self).__init__(**kwargs)
        if birds is None:
            birds = []

        self.birds = birds

    def apply(self, manager, ids=None):

        if len(self.birds):
            filter_func = lambda bn: bn in self.birds
        else:
            filter_func = lambda bn: True

        ids = manager.database.filter_by_func(bird=filter_func,
                                              ids=ids)

        return super(BirdFilter, self).apply(manager, ids)

    def __str__(self):

        return self.__class__.__name__

class CallTypeFilter(SoundFilter):

    def __init__(self, types=None, **kwargs):
        """ Filter a list of ids by call type. types should be a list of call types to
be included. If types is None, any sound with a call_type annotation will be
accepted.
        """

        super(CallTypeFilter, self).__init__(**kwargs)
        if types is None:
            types = []

        self.types = types

    def apply(self, manager, ids=None):

        if len(self.types):
            filter_func = lambda ct: ct in self.types
        else:
            filter_func = lambda ct: True

        ids = manager.database.filter_by_func(call_type=filter_func,
                                              ids=ids)

        return super(CallTypeFilter, self).apply(manager, ids)

class MultiFilter(SoundFilter):

    def __init__(self, filters, **kwargs):

        super(MultiFilter, self).__init__(**kwargs)
        self.filters = filters
        self.num_matches = None
        for filt in self.filters:
            if filt.num_matches is not None:
                if self.num_matches is None:
                    self.num_matches = filt.num_matches
                else:
                    self.num_matches = min(self.num_matches, filt.num_matches)

    def apply(self, manager, ids=None):

        for ii, filt in enumerate(self.filters):
            if ii != len(self.filters) - 1:
                filt.num_matches = None
            else:
                filt.num_matches = self.num_matches

            ids = filt.apply(manager, ids)

        return super(MultiFilter, self).apply(manager, ids)


class SoundProcessor(object):

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def apply(self, sound):

        pass

    def __str__(self):

        return self.__class__.__name__


class SongFiltProcessor(SoundProcessor):

    __doc__ = "\n Wrapper for songfilt: %s" % utils.songfilt.__doc__

    def apply(self, sound):

        return utils.songfilt(sound, *self.args, **self.kwargs)


class RampProcessor(SoundProcessor):

    __doc__ = "\n Wrapper for Sound.ramp: %s" % Sound.ramp.__doc__

    def apply(self, sound):

        return sound.ramp(*self.args, **self.kwargs)


class FilterProcessor(SoundProcessor):

    __doc__ = "\n Wrapper for Sound.filter: %s" % Sound.filter.__doc__

    def apply(self, sound):

        return sound.filter(*self.args, **self.kwargs)


class ResampleProcessor(SoundProcessor):

    __doc__ = "\n Wrapper for Sound.resample: %s" % Sound.resample.__doc__

    def apply(self, sound):

        return sound.resample(*self.args, **self.kwargs)


class LevelProcessor(SoundProcessor):

    __doc__ = "\n Wrapper for Sound.set_level: %s" % Sound.set_level.__doc__

    def apply(self, sound):

        return sound.set_level(*self.args, **self.kwargs)


class NormalizeCallTypeProcessor(SoundProcessor):

    def __init__(self, *args, **kwargs):

        super(NormalizeCallTypeProcessor, self).__init__(*args, **kwargs)
        self.max = None

    def initialize(self, sounds):

        self.max = np.max([s.level for s in sounds])

    def apply(self, sound):

        pass

