from __future__ import print_function, division
from pecking_analysis.stimulus_protocol import *
from neosound.sound import *
from zeebeez.sound import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TylerBaseProtocol(SoundProtocol):

    preprocessors = [SongFiltProcessor(frequency_range=[250*hertz, 8000*hertz]),  # process with songfilt
                     ResampleProcessor(samplerate=22050*hertz),
                     RampProcessor(duration=.02*second),
                     LevelProcessor(70*dB)]
    postprocessors = [LevelProcessor(70*dB)]

    def __init__(self, manager, filters, *args, **kwargs):

        output_manager = kwargs.get("output_manager", None)
        if output_manager is None:
            output_manager = SoundManager(DictStore)

        super(TylerBaseProtocol, self).__init__(manager,
                                                filters,
                                                preprocessors=self.preprocessors,
                                                postprocessors=self.postprocessors,
                                                output_manager=output_manager
                                                )

class CallOperantSequenceProtocol(TylerBaseProtocol):

    def __init__(self, manager,
                 target_type,
                 nontarget_types,
                 birds=None,
                 output_manager=None,
                 additional_filters=None,
                 target_stimuli=None,
                 nontarget_stimuli=None):
        """
        Creates a set of sequence stimuli where each element of the sequence is a combination of a target_type call with some number of background calls
        :param manager: The original manager to pull calls from
        :param target_type: The target call type
        :param nontarget_types: A list of potential background call types
        :param birds: A list of birds to select from
        :param output_manager: The output sound manager for the protocol (default DictStore)
        :param additional_filters: A list of additional filters (default empty). These are applied as an AND with the call type and bird name filters.
        :param target_stimuli: Number of target stimuli to select from the sound manager (default None selects all matches)
        :param nontarget_stimuli: Number of nontarget stimuli to select from the sound manager (default None selects all matches)
        """

        filters = [CallTypeFilter(types=[target_type], num_matches=target_stimuli),
                   CallTypeFilter(types=nontarget_types, num_matches=nontarget_stimuli)]
        if birds is not None:
            if not isinstance(birds, list):
                birds = [birds]
            bird_filter = BirdFilter(birds=birds)
            filters = [MultiFilter([bird_filter, filt]) for filt in filters]

        if additional_filters is not None:
            filters = [MultiFilter(additional_filters + [filt]) for filt in filters]

        super(CallOperantSequenceProtocol, self).__init__(manager,
                                                          filters,
                                                          output_manager=output_manager)

        self.target = target_type
        self.nontargets = nontarget_types

    def create_stimuli(self, num_stimuli=100,
                       num_target=None,
                       min_isi=.25*second,
                       max_isi=1*second,
                       sequence_length=6,
                       sequence_duration=6*second,
                       annotations=None,
                       ):
        """
        Creates a set of sequence stimuli.
        :param num_stimuli: The total number of stimuli to generate (default 100).
        :param num_target: The total number of target stimuli to generate (default .2 * num_stimuli).
        :param min_isi: The minimum inter-stimulus interval for the sequence (default .25*second).
        :param max_isi: The maximum inter-stimulus interval for the sequence (default 1*second).
        :param sequence_length: The number of call combinations per sequence (default 6).
        :param sequence_duration: The duration of the desired sequence (default 6*second).
        :param annotations: Additional annotations to add to each output sound.
        """

        if num_target is None:
            num_target = int(.2 * num_stimuli)

        if annotations is None:
            annotations = dict()

        self.outputs = list()
        stims_by_type = dict()
        for s in self.inputs:
            stims_by_type.setdefault(s.annotations["call_type"], list()).append(s)
        for key, sounds in stims_by_type.iteritems():
            stims_by_type[key] = sounds * np.ceil(float(sequence_length) / len(sounds))

        target_inds = np.random.permutation(np.arange(num_stimuli))[:num_target]
        for ii in xrange(num_stimuli):
            # Choose a call_type
            if ii in target_inds:
                call_type = self.target
            else:
                call_type = random.choice(self.nontargets)

            logger.debug("%d) Creating sequence of %s" % (ii, call_type))
            components = np.random.permutation(stims_by_type[call_type])[:sequence_length]
            output = utils.nonoverlapping_sequence(components,
                                                   duration=sequence_duration,
                                                   min_isi=min_isi,
                                                   max_isi=max_isi)
            output.annotate(**annotations)
            output.annotate(call_type=call_type,
                            target=(ii in target_inds))

            self.outputs.append(output)


class SceneOperantSequenceProtocol(CallOperantSequenceProtocol):

    def __init__(self, *args, **kwargs):

        super(SceneOperantSequenceProtocol, self).__init__(*args, **kwargs)

    def create_stimuli(self, num_stimuli=100,
                       sequence_length=6,
                       sequence_duration=6*second,
                       min_isi=.25*second,
                       max_isi=1*second,
                       num_mixtures=None,
                       ratio=0*dB,
                       min_delay=.1*second,
                       num_per_mix=1,
                       annotations=None,
                       ):
        """
        Creates a set of sequence stimuli where each individual stimulus is either a combination of target stimuli and background stimuli or just background stimuli
        :param num_stimuli: The total number of stimuli to generate (default 100).
        :param sequence_length: The number of call combinations per sequence (default 6).
        :param sequence_duration: The duration of the desired sequence (default 6*second).
        :param min_isi: The minimum inter-stimulus interval for the sequence (default .25*second).
        :param max_isi: The maximum inter-stimulus interval for the sequence (default 1*second).
        :param num_mixtures: The number of mixed stimuli to generate (default .2 * num_stimuli).
        :param ratio: The total signal to background ratio in dB (default 0*dB).
        :param min_delay: The minimum delay between background onset and signal onset (default .1*second).
        :param num_per_mix: The number of background stimuli in a mixture (default 1).
        :param annotations: Additional annotations to add to each output sound.
        """

        if num_mixtures is None:
            num_mixtures = int(.2 * num_stimuli)

        if annotations is None:
            annotations = dict()

        self.outputs = list()
        stims_by_type = dict()
        for s in self.inputs:
            stims_by_type.setdefault(s.annotations["call_type"], list()).append(s)

        for key, sounds in stims_by_type.iteritems():
            if key == self.target:
                stims_by_type[key] = sounds * np.ceil(float(sequence_length) / len(sounds))
            else:
                stims_by_type[key] = sounds * np.ceil(float(sequence_length * num_per_mix) / len(sounds))

        target_inds = np.random.permutation(np.arange(num_stimuli))[:num_mixtures]
        for ii in xrange(num_stimuli):
            background_type = random.choice(self.nontargets)
            logger.debug("%d) Creating sequence of %s" % (ii, background_type))
            if ii in target_inds:
                logger.debug("\t Including target call")
                targets = np.random.permutation(stims_by_type[self.target])[:sequence_length]
            backgrounds = np.random.permutation(stims_by_type[background_type])[:sequence_length * num_per_mix]

            components = list()
            for jj in xrange(sequence_length):
                max_dur = 0*second
                for kk in xrange(num_per_mix):
                    max_dur = max(max_dur, backgrounds[jj * num_per_mix + kk].duration)
                if ii in target_inds:
                    max_dur = max(max_dur, targets[jj].duration + min_delay)

                component = backgrounds[jj * num_per_mix].pad(duration=max_dur, start=0*second)
                for kk in xrange(1, num_per_mix):
                    max_start = max_dur - backgrounds[jj * num_per_mix + kk].duration
                    component = backgrounds[jj * num_per_mix + kk].embed(component, max_start=max_start, ratio=0*dB)
                if ii in target_inds:
                    max_start = max_dur - targets[jj].duration
                    component = targets[jj].embed(component,
                                                  max_start=max_start,
                                                  min_start=min_delay,
                                                  ratio=ratio)
                components.append(component)

            output = utils.nonoverlapping_sequence(components,
                                                   duration=sequence_duration,
                                                   min_isi=min_isi,
                                                   max_isi=max_isi)
            output.annotate(**annotations)
            output.annotate(background_type=background_type,
                            target=(ii in target_inds))

            self.outputs.append(output)


if __name__ == "__main__":

    from pecking_analysis.scene_stimuli import *

    init_manager = SoundManager(HDF5Store,
                                "/auto/tdrive/tlee/julie_stimuli.h5",
                                read_only=True)
    output_filename = "/tmp/test_protocol.h5"
    if os.path.exists(output_filename):
        os.remove(output_filename)

    output_manager = SoundManager(HDF5Store,
                                  output_filename)
    target_type = "distance"
    background_types = ["whine"]

    p = SceneSequenceProtocol(init_manager,
                              target_type,
                              background_types,
                              output_manager=output_manager,
                              target_stimuli=30,
                              nontarget_stimuli=30)
    p.filter()
    p.switch_manager(p.output_manager)
    p.preprocess()

    p.create_stimuli(30, sequence_duration=4*second)

    p.postprocess()


