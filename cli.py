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
def download_data():
    from analysis.download_scripts.project_lesions_2021 import download
    download()


@click.command("data")
@click.argument("subjects", type=str, nargs=-1)
@click.option("-d", "--date", "--from", type=click.DateTime(formats=("%d-%m-%y", "%d%m%y", "%d-%m-%Y", "%d%m%Y")))
@click.option("--until", type=click.DateTime(formats=("%d-%m-%y", "%d%m%y", "%d-%m-%Y", "%d%m%Y")))
@click.option("--include-shaping", is_flag=True)
@click.option("--debug", is_flag=True)
def summarize_data(subjects, date, until, include_shaping, debug):
    import pandas as pd

    from configs import config
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


cli.add_command(download_data)
cli.add_command(summarize_data)


if __name__ == "__main__":
    cli()
