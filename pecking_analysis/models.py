import h5py
import os
import logging
import numpy as np
from scipy.io import loadmat
from lasp.sound import spectrogram as compute_spectrogram
from lasp.strfs import conv_activations_2d

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseModel(object):

    _default_spec_params = dict(spec_sample_rate=1000,
                                freq_spacing=125,
                                )


    def __init__(self, manager, output_file, spec_params=None):

        self.manager = manager
        self.output_file = output_file

        # Initialize spectrogram parameters
        self.spec_params = spec_params if spec_params else dict()
        for key, value in self._default_spec_params.iteritems():
            if key not in self.spec_params:
                self.spec_params[key] = value

        # Ensure read-only manager
        self.manager.database.read_only = True

        self.create_output_file()

        self._filter_id = 0

        self.filters = dict()
        self.biases = dict()
        self.spectrograms = dict()

    def create_output_file(self):

        if not os.path.exists(self.output_file):
            with h5py.File(self.output_file, "a") as hf:
                hf.attrs["manager"] = self.manager.database.filename
                hf.create_group("sounds")
                hf.create_group("filters")
                g = hf.create_group("spectrogram_parameters")
                for key, value in self.spec_params.iteritems():
                    g.attrs[key] = value

    def _get_group(self, hf, group_name):

        if group_name in hf:
            g = hf[group_name]
        else:
            g =  hf.create_group(group_name)

        return g

    def get_category(self, id_, cat_name="call_type"):

        annotations = self.manager.database.get_annotations(id_)
        if cat_name in annotations:
            return self.categories.index(annotations[cat_name])

    def get_spectrograms(self, ids):

        if not isinstance(ids, list):
            ids = [ids]

        spec_params = self.spec_params.copy()
        spec_sample_rate = spec_params.pop("spec_sample_rate")
        freq_spacing = spec_params.pop("freq_spacing")

        spectrograms = list()
        for id_ in ids:
            if id_ in self.spectrograms:
                spec = self.spectrograms[id_]
            else:
                spec = self.load_spectrogram(id_)
                if spec is None:
                    s = self.manager.reconstruct(id_)
                    spec = compute_spectrogram(s.asarray(), s.samplerate, spec_sample_rate, freq_spacing, **spec_params)[2]
                    self.store_spectrogram(id_, spec)
                self.spectrograms[id_] = spec
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

    def store_filter(self, filt, category, bias=0, id_=None):

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
            g.attrs["bias"] = bias

    def load_filter(self, id_):

        id_ = unicode(id_)
        with h5py.File(self.output_file, "r") as hf:
            g = hf["filters"]
            if id_ in g:
                return (g[id_]["filter"], g[id_].attrs["category"])

    def compute_activations(self, ids):

        activations = dict()
        spectrograms = self.get_spectrograms(ids)
        cat_inds = [self.get_category(id_) for id_ in ids]
        for spec in spectrograms:
            for ci, filters in self.filters.iteritems():
                act = conv_activations_2d([spec], filters)
                activations.setdefault(ci, list()).extend(act)

        return activations, spectrograms, cat_inds



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

    _type_translations = dict(Ag="aggressive",
                              Be="begging",
                              DC="distance",
                              Di="distress",
                              LT="long tonal",
                              Ne="nesting",
                              So="song",
                              Te="tet",
                              Th="thuck",
                              Tu="tuk",
                              Wh="whine",
                              )

    def __init__(self,
                 manager,
                 output_file,
                 discriminant_file):

        super(FilteringDiscriminantModel, self).__init__(manager, output_file)
        self.initialize_filters(discriminant_file)
        vars = loadmat(discriminant_file, squeeze_me=True, variable_names=["fo", "to", "vocTypes"])
        self.spec_params["spec_sample_rate"] = 1 / float(vars["to"][1] - vars["to"][0])
        self.spec_params["freq_spacing"] = int(float(vars["fo"][1] - vars["fo"][0]) * 6.0 / (2 * np.pi)) # This is a hack. It seems to give the right number of values but not exact frequency values...
        self.spec_params["min_freq"] = min(vars["fo"])
        self.spec_params["max_freq"] = max(vars["fo"])
        self.categories = [self._type_translations[str(vt)] for vt in vars["vocTypes"]]

    def initialize_filters(self, discriminant_file):

        vars = loadmat(discriminant_file, squeeze_me=True, variable_names=["nf", "nt", "PC_LR", "PC_LR_Bias"])
        nf, nt = vars["nf"], vars["nt"]
        for ii in xrange(vars["PC_LR"].shape[1]):
            cat_filter = np.reshape(vars["PC_LR"][:, ii], (nt, nf)).T
            self.filters.setdefault(ii, list()).append(cat_filter)
            self.biases.setdefault(ii, list()).append(vars["PC_LR_Bias"][ii])
            self.store_filter(cat_filter, ii, bias=vars["PC_LR_Bias"][ii])






if __name__ == "__main__":
    from neosound.sound import *

    sm = SoundManager(HDF5Store, "/home/tlee/data/isolated_call_sequences.h5", read_only=True)
    dcs = sm.database.filter_by_func(call_type=lambda x: x == "distance", output=lambda b: True, num_matches=20)
    output_file = "/home/tlee/data/test_discriminant_model.h5"
    discriminant_file = "/home/tlee/Downloads/LogisticForTyler.mat"
    fdm = FilteringDiscriminantModel(sm, output_file, discriminant_file)