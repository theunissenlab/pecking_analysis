import logging
import numpy as np
import lasp.sound import spectrogram as compute_spectrogram

logger = logging.getLogger(__name__)
logger.setLevel(logging.debug)

spectrogram_cache = dict()
def convolve_filters(spectrograms, filters):

    activations = list()
    for sound in sounds:
        if sound.id in spectrogram_cache:
            spec = spectrogram_cache[sound.id]
        else:
            spec = compute_spectrogram(sound, sound.samplerate, )


class FilteringModel(object):

    def __init__(self, manager,
                 training_ids,
                 testing_ids,
                 units_per_category,
                 freq_spacing=60,
                 spec_samplerate=1000,
                 filter_length=100):

        self.manager = manager
        self.training_ids = training_ids
        self.testing_ids = testing_ids
        self.units_per_category = units_per_category

        self.categories = self.get_categories()
        for ii in xrange(len(self.categories)):

        self.filters = self.initialize_filters()

    def initialize_filters(self, s):

        filters = list()
        for ii in xrange(len(self.categories)):
            cat_filters = list()
            for jj in xrange(self.units_per_category):
                cat_filter = np.random.uniform(-0.1, 0.1, (self.nfreqs, self.ntimepts))

    def get_categories(self):

        categories = list()
        for id_ in self.training_ids:
            annotations = self.manager.database.get_annotations(id_)
            if ("call_type" in annotations) and (annotations["call_type"] not in categories):
                categories.append(annotations["call_type"])

        return categories

    def train(self, num_iters=100):

        sounds = dict()
        logger.info("Beginning training")
        for iter in xrange(num_iters):
            logger.info("Iteration %d" % iter)
            for id_ in np.random.permutation(self.training_ids):
                logger.debug("\tTraining with sound id %s" % id_)
                if id_ not in sounds:
                    sound = self.manager.reconstruct(id_)
                    sounds[id_] = spectrogram(sound.asarray(), sound.samplerate, self.spec_samplerate, self.freq_spacing)[2]
                spec = sounds[id_]
