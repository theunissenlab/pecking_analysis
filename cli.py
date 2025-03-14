import datetime
import logging
import os
import sys
import click


logging.basicConfig()


@click.group()
def cli():
    pass


@click.command()
@click.option("--trials/--no-trials", default=True, help="Download the trial data")
@click.option("--preference", is_flag=True, help="Download the preference test data")
def download_data(trials, preference):
    from analysis.download_scripts.project_lesions_2021 import (
        download,
        download_preference_test_data
    )

    if trials:
        download()

    if preference:
        download_preference_test_data()


@click.command()
def locate_data():
    from analysis.download_scripts.project_lesions_2021 import get_subject_folders
    print(get_subject_folders())


@click.command("data")
@click.argument("subjects", type=str, nargs=-1)
@click.option("-d", "--date", "--from", type=click.DateTime(formats=("%d-%m-%y", "%d%m%y", "%d-%m-%Y", "%d%m%Y")))
@click.option("--until", type=click.DateTime(formats=("%d-%m-%y", "%d%m%y", "%d-%m-%Y", "%d%m%Y")))
@click.option("--include-shaping", is_flag=True)
@click.option("--debug", is_flag=True)
def summarize_data(subjects, date, until, include_shaping, debug):
    import pandas as pd

    from configs.active_config import config
    from analysis.analysis import create_informative_trials_column, summarize_data
    from analysis.pipelines.project_lesions_2021 import NoDataForSubject, load_all_for_subject, run_pipeline_subject

    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    from_day = date.date() if date else datetime.date.today()
    until_day = until.date() if until else from_day

    if not len(subjects):
        subjects = config.subjects or []

    for day_idx in range((until_day - from_day).days + 1):
        day = from_day + datetime.timedelta(days=day_idx)

        results = []
        for subject in subjects:
            try:
                df = run_pipeline_subject(
                    subject,
                    config,
                    from_date=day,
                    to_date=day,
                    include_shaping=include_shaping,
                )
            except NoDataForSubject as e:
                logger.debug(e)
            else:
                if len(df):
                    results.append(df)

        if not len(results):
            continue

        print(summarize_data(pd.concat(results)).to_string(
            float_format=lambda x: str(round(x, 2)),
            justify="right",
        ))

@click.command()
@click.argument("output_path", type=click.Path())
@click.option("--debug", is_flag=True)
def package_data(output_path, debug):
    import pandas as pd

    from configs.active_config import config
    # from analysis.download_scripts.project_lesions_2021 import download
    from analysis.pipelines.project_lesions_2021 import run_pipeline_subject

    # try:
    #     download()
    # except:
    #     pass
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    subject_dfs = []
    for subject in config.subjects:
        # Preprocessing steps
        df = run_pipeline_subject(subject, config)
        subject_dfs.append(df)

    full_df = pd.concat(subject_dfs).reset_index(drop=True)
    full_df.to_csv(
        output_path,
        index=False
    )


@click.command()
@click.option("--debug", is_flag=True)
def missing_trials(debug):
    """Count # of missing trials in original data by (by skipped numbers in 'trial' column of csvs)

    The original recording sometimes missed errors in the csv files (missing trial number)

    1. Count missing trials by # trials where trial(n)+1 != trial(n+1)  [before pipelines applied]
    2. Count total trials [after pipelines applied]
    3. Compute (output of 1 / output of 2)
    """
    import pandas as pd
    import numpy as np

    from configs.active_config import config
    from analysis.pipelines.project_lesions_2021 import load_all_for_subject

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Patch config so that there are no filters applied
    config.filters = []

    n_trials = 0
    n_missing = 0
    n_affected = 0
    for subject in config.subjects:
        # Preprocessing steps
        df = load_all_for_subject(subject, config)

        # Count missing trials by trials where the trial count increases by an amount
        # more than 1
        trial_diff = np.diff(df["Trial"])

        subj_n_trials = len(df)
        subj_n_missing = int(np.sum(trial_diff > 1))
        missing_trials = np.where(trial_diff > 1)[0]

        n_trials += subj_n_trials
        n_missing += subj_n_missing

        print("{subject}: Identified {n_missing} trials missing out of {n_trials}".format(
            subject=subject,
            n_missing=subj_n_missing,
            n_trials=subj_n_trials
        ))

        time_diff = np.array([pd.Timedelta(x).total_seconds() for x in np.diff(df["Time"])]).astype(np.float32)
        rts = np.array([pd.Timedelta(x).total_seconds() for x in df["RT"].iloc[:-1]]).astype(np.float32)

        subj_n_affected = np.sum(
            (time_diff > 0)
            & (time_diff - rts > 0.5)
        )

        print("{n_affected} trials with time difference greater than 0.5s + RT".format(n_affected=subj_n_affected))
        n_affected += subj_n_affected

        # Count how often RT - rt is > 0.5s

    print("-" * 40)
    print("Identified {n_missing} trials missing out of {n_trials}".format(
        n_missing=n_missing,
        n_trials=n_trials
    ))
    print("{n_affected} trials with time difference greater than 0.5s + RT".format(n_affected=n_affected))

    print("Note: run `python cli.py data` and sum the trials column to get the total number of trials after filtering")



cli.add_command(download_data)
cli.add_command(summarize_data)
cli.add_command(locate_data)
cli.add_command(package_data)
cli.add_command(missing_trials)


@click.group()
def preference():
    pass


@click.command("list")
@click.argument("subjects", type=str, nargs=-1)
@click.option("-d", "--date", "--from", type=click.DateTime(formats=("%d-%m-%y", "%d%m%y", "%d-%m-%Y", "%d%m%Y")))
@click.option("--until", type=click.DateTime(formats=("%d-%m-%y", "%d%m%y", "%d-%m-%Y", "%d%m%Y")))
@click.option("--debug", is_flag=True)
def list_preference_test_data(subjects, date, until, debug):
    import pandas as pd

    from configs.active_config import config
    from analysis.pipelines.preference_tests_2021 import query_preference_tests

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    from_day = date.date() if date else None
    until_day = until.date() if until else from_day

    if not len(subjects):
        subjects = config.subjects or []

    results = []
    for subject in subjects:
        df = query_preference_tests(
            subject,
            config,
            from_date=from_day or None,
            to_date=until_day or None,
        )

        if df is not None:
            print("\n".join(df["Audio"]))


preference.add_command(list_preference_test_data)
cli.add_command(preference)


if __name__ == "__main__":
    cli()
