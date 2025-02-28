import glob
import os

from configs.base_pecking_test_2019 import BaseConfig


class LocalConfig:

    behavior_data_dir = "/data/pecking_test/behavior"
    gdrive_credentials_dir = None

    subjects = [
        os.path.basename(foldername)
        for foldername
        in glob.glob(os.path.join(behavior_data_dir, "*"))
    ]

    @property
    def metadata_dir(self):
        raise NotImplementedError
