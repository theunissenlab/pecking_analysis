# Pecking Analysis

pecking test preprocessing

## Setup

```
pip install -r requirements.txt
```

Setup file save paths:

* copy the file `configs/plump_config.py` and name it `configs/XXX_config.py`.

* edit the new file and fill in `behavior_data_dir` to be the location of the trialdata csvs, `metadata_dir` to be a place where metadata will live about lesions (you need to set up that file).

* edit `configs/project_lesions_2021.py` to import your new config

If you need to download data, see the Downloading Data section at the bottom.

## Steps

1. Download data OR point `behavior_data_dir` to where your data is

2. Run `python cli.py package-data OUTPUT_PATH`. a single csv with all trials will be saved at `OUTPUT_PATH`.

## Lesion metadata file

Should look something like this

```
Subject,Lesion Date,Treatment,Drug
BluYel2571F,5/2/2019,CTRL,
GreBlu5039F,5/2/2019,NCM,NMA
...
```

## Downloading Data

To download data, install [gdrive_access](https://github.com/theunissenlab/gdrive_python), easiest way will be to do `pip install git+https://github.com/theunissenlab/gdrive_python.git@main`.

Follow the setup instructions in that project's readme to get google drive API credentials. In the config file, set up `gdrive_credentials_dir` to be the folder containing your settings.yaml file.

Make sure to share the folder `pecking_test_data` (from fetlab) with yourself. You should see it at the top level of your "sharedWithMe" folder on Google Drive.

Note: it is important to specify `main` branch of gdrive_access which has a fix that lets you see shared folders.

Run `python cli.py locate-data` to see if the subject folders are being found on google drive properly.


