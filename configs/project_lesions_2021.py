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

        "BluWhi0398F",
        "BluWhi3230M",
        "GraYel7337F",
        "GreWhi2703M",

        "HpiGre8613M",
        "BluGre4315M",
        "RedGra7912M",
        "BluRed8773M",
    ]

    valid_test_contexts = [
        "DCvsDC_1v1",
        "DCvsDC_1v1_S2",
        "DCvsDC_4v4",
        "DCvsDC_4v4_S2",
        "DCvsDC_6v6_d1",
        "DCvsDC_6v6_d1_S2",
        "DCvsDC_6v6_d2",
        "DCvsDC_6v6_d2_S2",
        "SovsSo_1v1",
        "SovsSo_1v1_S2",
        "SovsSo_4v4",
        "SovsSo_4v4_S2",
        "SovsSo_8v8_d1",
        "SovsSo_8v8_d1_S2",
        "SovsSo_8v8_d2",
        "SovsSo_8v8_d2_S2",
    ]

    @property
    def subject_metadata_path(self):
        return os.path.join(self.metadata_dir, "lesion_metadata.csv")
