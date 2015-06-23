import h5py
import logging
import numpy as np
from scipy.io import loadmat
import lasp.sound import spectrogram as compute_spectrogram

logger = logging.getLogger(__name__)
logger.setLevel(logging.debug)

def convolve_filters(spectrograms, filters):

    activations = list()
    for sound in sounds:
        if sound.id in spectrogram_cache:
            spec = spectrogram_cache[sound.id]
        else:
            spec = compute_spectrogram(sound, sound.samplerate, )


class BaseModel(object):

    _default_spec_params = dict(spec_sample_rate=1000,
                                freq_spacing=125,
                                )


    def __init__(self, manager, output_file, spec_params=None):

        self.manager = manager
        self.output_file = output_file

        # Initialize spectrogram parameters
        self.spec_params = spec_params if spec_params else dict()
        for key, value in self._default_spec_params:
            if key not in self.spec_params:
                self.spec_params[key] = value

        # Ensure read-only manager
        self.manager.database.read_only = True

        self.create_output_file()

        self._filter_id = 0

        self.filters = dict()
        self.spectrograms = dict()
        self.activations = None

    def create_output_file(self):

        with h5py.File(self.output_file, "a") as hf:
            hf.attrs["manager"] = self.manager.database.filename
            hf.create_group("sounds")
            hf.create_group("filters")
            g = hf.create_group("spectrogram_parameters")
            for key, value in self.spec_params:
                g.attrs[key] = value

    def _get_group(self, hf, group_name):

        if group_name in hf:
            g = hf[group_name]
        else:
            g =  hf.create_group(group_name)

        return g

    def get_category(self, id_):

        annotations = self.manager.database.get_annotations(id_)
        if "call_type" in annotations:
            return annotations["call_type"]

    def get_spectrograms(self, ids):

        if not isinstance(ids, list):
            ids = [ids]

        spectrograms = list()
        for id_ in ids:
            spec = self.load_spectrogram(id)
            if spec is None:
                s = self.manager.reconstruct(id_)
                spec = compute_spectrogram(s, s.samplerate, **spec_params)[2]
                self.store_spectrogram(id, spec)
            spectrograms.append(spec)

        return spectrograms

    def load_spectrogram(self, id_):

        id_ = unicode(id_)
        with h5py.File(self.output_file, "r") as hf:
            g = hf["sounds"]
            if id_ in g:
                return g[id_]["spectrogram"]

        return None

    def store_spectrogram(self, id_, spec):

        id_ = unicode(id_)
        with h5py.File(self.output_file, "a") as hf:
            g = hf["sounds"]
            g = self._get_group(g, id_)
            if "spectrogram" in g:
                del g["spectrogram"]

            g.create_dataset("spectrogram", data=spec)

    def store_filter(self, filt, category, id_=None):

        if id_ is None:
            id_ = self._filter_id
            self._filter_id += 1

        id_ = unicode(id_)
        with h5py.File(self.output_file, "a") as hf:
            g = hf["filters"]
            g = self._get_group(g, id_)
            if "filter" in g:
                del g["filter"]

            g.create_dataset("filter", data=filt)
            g.attrs["category"] = category

    def load_filter(self, id_):

        id_ = unicode(id_)
        with h5py.File(self.output_file, "r") as hf:
            g = hf["filters"]
            if id_ in g:
                return (g[id_]["filter"], g[id_].attrs["category"])


class FilteringModel(BaseModel):

    def __init__(self, manager,
                 output_file,
                 spec_params=None,
                 num_categories=9,
                 units_per_category=1,
                 nfreqs=100,
                 ntimepts=100):

        super(FilteringModel, self).__init__(self, manager, output_file, spec_params=spec_params)
        self.initialize_filters(num_categories, units_per_category, nfreqs, ntimepts)

    def initialize_filters(self, num_categories, units_per_category, nfreqs, ntimepts):

        for ii in xrange(num_categories):
            for jj in xrange(units_per_category):
                cat_filter = np.random.uniform(-0.1, 0.1, (nfreqs, ntimepts))
                self.filters.setdefault(jj, list()).append(cat_filter)
                self.store_filter(cat_filter, jj)

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


class FilteringDiscriminantModel(BaseModel):

    def __init__(self,
                 manager,
                 output_file,
                 discriminant_file):

        super(FilteringDiscriminantModel, self).__init__(manager, output_file)
        self.initialize_filters(discriminant_file)

    def initialize_filters(self, discriminant_file):

        vars = loadmat(discriminant_file)

