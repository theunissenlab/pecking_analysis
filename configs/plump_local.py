import glob
import os

from configs.base_pecking_test_2019 import (
    column_mapping,
    filters,
    csv_column_names,
)


save_dir = "/data/pecking_test/behavior"
gdrive_access_credentials_dir = None

subjects = [
    os.path.basename(foldername)
    for foldername
    in glob.glob(os.path.join(save_dir, "*"))
]
