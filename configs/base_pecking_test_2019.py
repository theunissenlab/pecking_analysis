import os


class BaseConfig:
    csv_column_names = ["Session", "Trial", "Time", "Stimulus", "Class",
              "Response", "Correct", "RT", "Reward", "Max Wait"]

    column_mapping = {
        "Trial": ("Trial", None),
        "Time": ("Time", None),
        "Date": ("Time", lambda time_: time_.date()),
        "RT": ("RT", None),
        "Reward": ("Reward", None),
        "Interrupt": ("Response", None),
        "Masked Stimulus File": ("Masked Stimulus", lambda filename: os.path.splitext(os.path.basename(filename))[0] if filename else None),
        "Masked Stimulus Vocalizer": ("Masked Vocalizer", None),
        "Masked Stimulus Call Type": ("Masked Call Type", None),
        "Masked Stimulus Class": ("Masked Class", lambda rewarded: None if not rewarded else "Rewarded" if rewarded == "Rewarded" else "Nonrewarded"),  # Remap Unrewarded to Nonrewarded
        "Stimulus File": ("Stimulus", lambda filename: os.path.splitext(os.path.basename(filename))[0]),
        "Stimulus Vocalizer": ("Vocalizer", None),
        "Stimulus Call Type": ("Call Type", None),
        "Stimulus Class": ("Class", lambda rewarded: "Rewarded" if rewarded == "Rewarded" else "Nonrewarded"),  # Remap Unrewarded to Nonrewarded
        "Test Context": ("Test Context", None),
        "Subject": ("Subject", None),
        "Subject Sex": ("Subject", lambda subject: subject[-1]),
        "Condition": (None, None),       # None or MonthLater
        "Treatment": (None, None),   # NCM, CTRL, etc
        "Ladder Group": (None, None),    # PrelesionSet1, etc
    }

    filters = [
        {
            "name": "fix_response_time_column",
            "kwargs": {}
        },
        {
            "name": "reject_double_pecks",
            "kwargs": {
                "rejection_threshold": 200
            }
        },
        {
            "name": "reject_stuck_pecks",
            "kwargs": {
                "iti": (6000, 6050),
                "in_a_row": 3
            }
        },
    ]

    @property
    def subject_metadata_path(self):
        raise NotImplementedError

    @property
    def behavior_data_dir(self):
        raise NotImplementedError

    @property
    def gdrive_credentials_dir(self):
        raise NotImplementedError
