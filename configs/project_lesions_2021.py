import os

from configs.base_pecking_test_2019 import BaseConfig

# Change this on different computers
from configs.kevin_local import LocalConfig


class ProjectConfig(LocalConfig, BaseConfig):

    subjects = [
        "BluYel2571F",
        "GreBlu5039F",
        "GreBla3404M",
        "GraWhi4040F",
        "BlaGre1349M",
        "XXXHpi0038M",
        "XXXOra0037F",
        "YelPur7906M",
        "WhiWhi2526M",
        "XXXBlu0031M",
        "HpiGre0651M",
        "RedHpi0710F",
        "WhiBlu5805F",
    ]

    @property
    def subject_metadata_path(self):
        return os.path.join(self.metadata_dir, "lesion_metadata.csv")
