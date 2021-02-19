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
@click.option("--debug", is_flag=True)
def summarize_data(subjects, date, until, debug):
    import pandas as pd

    from configs.active_config import config
    from analysis.analysis import create_informative_trials_column, summarize_data
    from analysis.pipelines.project_lesions_2021 import load_all_for_subject, run_pipeline_subject

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
        df = run_pipeline_subject(
            subject,
            config,
            from_date=from_day or None,
            to_date=until_day or None,
        )
        if len(df):
            results.append(df)

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


cli.add_command(download_data)
cli.add_command(summarize_data)
cli.add_command(locate_data)
cli.add_command(package_data)


if __name__ == "__main__":
    cli()
