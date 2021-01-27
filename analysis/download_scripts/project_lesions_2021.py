import os

from gdrive_access import GDriveCommands
from pydrive2.files import GoogleDriveFile

from configs import config

try:
    g = GDriveCommands(os.path.join(config.gdrive_credentials_dir, "settings.yaml"))
except:
    g = None
    print("gdrive_access credentials not found")


def get_subject_folders():
    root = g.get_root("TLab Shared Folders")
    return g.ls(root, "pecking_test_data", "plump_synced", "behavior")


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

    g.download_files(csv_files, subject_save_dir, overwrite=g.Overwrite.ON_MD5_CHECKSUM_CHANGE)


def download():
    subjects = filter(lambda a: (a["title"] in config.subjects), get_subject_folders())
    subjects = list(subjects)
    for subject in subjects:
        download_subject_csvs(subject, save_dir=config.save_dir)
