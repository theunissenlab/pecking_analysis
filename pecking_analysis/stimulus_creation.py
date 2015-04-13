# This module should house the basic steps to create a directory of stimuli
# according to most any specification. The basic steps are:
# 1. gather all files from some directory and, optionally, all subdirectories. This could be done by using a pre-loaded sound manager as well.
# 2. filter those files based on some filename criteria or extracted properties. If step 1 was done using a sound manager, then this should be done on the manager and step 3 need only create a new manager.
# 3. Load all files into a manager
# 4. Preprocess all individual stimuli. Preprocessing steps should be built into a pipeline.
# 5. Optionally combine multiple individual stimuli per trial
# 6. Create trials. This would involved randomization rules and maximum-of-a-type rules.
# 7. Postprocess each trial's stimuli. Postprocessing steps should be built into a pipeline
# Classes required:
#
# Functions required:
# load_wavs - should take a directory, recursive=False, file_pattern, filter_func
#          file_pattern would be a basic string (e.g. "*.wav")
#          filter_func would be a function called with filename, directory that returns True or False
#
# create_manager - should take a list of filenames (e.g. from load_wavs) and create a manager from them.
#
# switch_manager - should take a manager and a list of ids to add to the new manager
#
#
import logging


logger = logging.getLogger()

class SoundProtocol(object):

    def __init__(self, manager, filters, preprocessors, postprocessors):

        self.manager = manager
        self.filters = filters
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

    def filter(self):

        for filt in self.filters:
            filt.apply(manager, ids)

    def preprocess(self):

        for preprocessor in self.preprocessors:
            for ii, sound in enumerate(self.inputs):
                self.inputs[ii] = preprocessor.apply(sound)

    def postprocess(self):

        for postprocessor in self.postprocessors:
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

    
