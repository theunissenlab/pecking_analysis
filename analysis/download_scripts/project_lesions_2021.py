import os

from gdrive_access import GDriveCommands
from pydrive2.files import GoogleDriveFile

from configs.active_config import config


try:
    g = GDriveCommands(os.path.join(config.gdrive_credentials_dir, "settings.yaml"))
except Exception as e:
    g = None
    print(type(e), e)
    print("gdrive_access credentials not found")


def get_subject_folders():
    root = g.get_root("TLab Shared Folders")
    ongoing = g.ls(root, "pecking_test_data", "plump_synced", "behavior")
    previous = g.ls(root, "pecking_test_data", "behavior_archive", "Ladder Memory Capacity 2019-2021")
    return ongoing + previous


def get_subject_dates(subject_folder: GoogleDriveFile):
    return g.ls(subject_folder)


def download_subject_csvs(subject_folder: GoogleDriveFile, save_dir: str):
    subject_name = subject_folder["title"]
    subject_save_dir = os.path.join(save_dir, subject_name, "raw")

    date_folders = get_subject_dates(subject_folder)
    csv_files = []
    for date_folder in date_folders:
        csv_files += [file_ for file_ in g.ls(date_folder) if file_["title"].endswith(".csv")]

    if not os.path.exists(subject_save_dir):
        os.makedirs(subject_save_dir)


    print("Downloading {}".format("\n".join([c["title"] for c in csv_files])))

    g.download_files(csv_files, subject_save_dir, overwrite=g.Overwrite.ON_MD5_CHECKSUM_CHANGE)


def download():
    subjects = filter(lambda a: (a["title"] in config.subjects), get_subject_folders())
    subjects = list(subjects)
    for subject in subjects:
        download_subject_csvs(subject, save_dir=config.behavior_data_dir)
