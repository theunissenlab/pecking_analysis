from __future__ import print_function, division
import os
import re

import numpy as np
import pandas as pd

from pecking.bird import Bird
from pecking.session import Session
from pecking.experiment import Experiment
from pecking.block import Block
from pecking.importer import MatlabTxt


class BirdManager(object):

    def __init__(self, filename=None, fasting_csv=None, session_csv=None):

        self.store = BirdStore(filename)
        self.fasting_csv = fasting_csv
        self.session_csv = session_csv
        if self.fasting_csv is not None:
            self.fasting_data = self.import_fasting_data()
        else:
            self.fasting_data = None

        if self.session_csv is not None:
            self.session_data = self.import_session_data()
        else:
            self.session_data = None

    def load(self, filename):

        pass

    def save(self, filename):

        pass

    def import_fasting_data(self):

        fasting_data = pd.read_csv(self.fasting_csv,
                                   header=0,
                                   index_col="Timestamp",
                                   parse_dates=["Timestamp", "Fast Date and Time"])
        fasting_data = fasting_data.dropna(how="all",
                                           subset=["Bird Name"])

        return fasting_data

    def import_session_data(self):

        session_data = pd.read_csv(self.session_csv,
                                   header=0,
                                   index_col="Timestamp",
                                   parse_dates=["Timestamp", "Date"])
        session_data = session_data.dropna(how="all",
                                           subset=["Bird Name"])
        session_data["Notes"].fillna("", inplace=True)

        return session_data

    def parse(self, files, Importer=MatlabTxt, overwrite=False):

        if not overwrite:
            existing_files = self.store.list_files()
            files = [fname for fname in self._get_files(files) if fname not in existing_files]

        imp = Importer()
        imp.parse(files)
        blocks = imp.blocks
        sessions = self._get_sessions(blocks)
        experiments = self._get_experiments(sessions)
        birds = self._get_birds(experiments)

        return birds


    def _get_sessions(self, blocks):

        sessions = dict()
        for blk in blocks:
            sessions.setdefault((blk.name, blk.start.date()), list()).append(blk)
        session_list = list()
        for block_list in sessions.values():
            name = block_list[0].name
            start = block_list[0].start
            if block_list[-1].end is not None:
                end = block_list[-1].end
            else:
                end = block_list[-1].start

            sess = Session(name=name,
                           start=start,
                           end=end)
            sess.blocks = block_list
            for blk in sess.blocks:
                blk.session = sess

            if self.session_data is not None:
                df = self.session_data[self.session_data["Bird Name"] == sess.name]
                df = df[df["Date"] == sess.start.date()]
                if len(df):
                    df = df.ix[0]
                    sess.weight = df["Initial Weight"]
                    sess.box = int(df["Box #"])
                    sess.post_weight = df["Final Weight"]
                    sess.notes = df["Notes"]
                    sess.labels = df["Stimulus Labels"]
                    sess.seed_given = df["Seed Amount"]

            session_list.append(sess)

        return session_list

    def _get_experiments(self, sessions):

        offset = 10 * pd.datetools.day
        same_experiment = lambda sess, e: sess.start < offset.apply(e.start)
        sessions.sort(key=lambda sess: (sess.name, sess.start.date()))
        dates = [sess.start.date() for sess in sessions]
        experiments = list()
        prev_name = ""
        for ii, sess in enumerate(sessions):
            if (ii == 0) or (sess.name != prev_name) or (not same_experiment(sess, e)):
                e = Experiment()
                e.name = sess.name
                e.start = sess.start
                e.end = sess.end
                if self.fasting_data is not None:
                    df = self.fasting_data[self.fasting_data["Bird Name"] == sess.name]
                    df = df[df["Fast Date and Time"] >= (e.start - pd.datetools.Day(10))]
                    if len(df):
                        df = df.ix[0]
                        e.weight = df["Weight"]
                        e.fast_start = df["Fast Date and Time"]
                prev_name = e.name
                experiments.append(e)

            if e.end < sess.end:
                e.end = sess.end
            e.sessions.append(sess)
            sess.experiment = e


        return experiments

    def _get_birds(self, experiments):

        bird_dict = dict()
        for e in experiments:
            if e.name in bird_dict:
                birdy = bird_dict[e.name]
            else:
                birdy = Bird(e.name)
                bird_dict[e.name] = birdy
            birdy.experiments.append(e)
            e.bird = birdy

        return bird_dict.values()


    def get_bird(self, bird_name):

        pass

    def get_all_birds(self):

        pass

    def _get_files(self, files):

        if isinstance(files, str):
            files = [files]
        file_list = list()
        for fname in files:
            if os.path.exists(fname):
                if os.path.isdir(fname):
                    file_list.extend(
                        self._get_files(map(lambda ss: os.path.join(fname, ss),
                                                         os.listdir(fname)))
                                    )
                else:
                    file_list.append(fname)

        return file_list


class BirdStore(object):

    def __init__(self, filename):

        self.filename = filename

    def list_files(self):

        return list()


if __name__ == "__main__":

    # from pecking.bird_manager import *
    directory = "/Users/tylerlee/Dropbox/pecking_test"
    session_csv = os.path.join(directory, "Pecking data - Session Data.csv")
    fasting_csv = os.path.join(directory, "Pecking data - Fasting Data.csv")
    bm = BirdManager("/tmp/test", fasting_csv, session_csv)
    birds = bm.parse(directory)

