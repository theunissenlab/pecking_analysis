#!/usr/bin/env python
import datetime
import os
import re
import csv
import numpy as np
import pandas as pd
from pecking_analysis import objects

BIRDS = {"BlaBlu1387F": "BlaLbu1387F",
         "GraGre1401M": "GraGre4401M",
         "BlaYel0208": "BlaYel0208M",
         "WhiXXX4100F": "WhiRas4100F"}

# normally when importing headers we will capitalize the words
# If the conversion is more than a capitalization, include them in this
# dictionary
COL_LABELS = {
    "index": "Trial",
    "stimulus_name": "Stimulus",
    "condition_name": "Class",
    "rt": "RT",
    "max_wait": "Max Wait",
    "delay_period_pecks": "Delay Pecks"
}

class Importer(object):

    def __init__(self):

        pass

    @classmethod
    def parse(cls, files):

        pass

    @staticmethod
    def get_name(bird_name):

        if bird_name in BIRDS:
            return BIRDS[bird_name]
        else:
            return bird_name


class PythonCSV(Importer):

    # pattern = "_".join(["(?P<name>(?:[A-Za-z]{3}){1,2}(?:[0-9]{2}){1,2}[MF]?)",
    #                     "trialdata",
    #                     "(?P<datestr>[0-9]*)\.csv"])

    # We gave a single banded subject the name XXXBluXX31M instead of Blu31M.
    # Need to allow for X in replacement of numbers in second half of name
    pattern = "_".join(["(?P<name>(?:[A-Za-z]{3}){1,2}(?:[X0-9]{2}){1,2}[MF]?)",
                        "trialdata",
                        "(?P<datestr>[0-9]*)\.csv"])
    pattern2 = "_".join(["(?P<name>(?:[A-Za-z]{3}){1,2}(?:[X0-9]{2}){1,2}[MF]?)",
                        "(?P<datestr>[0-9]*)",
                        "trialdata.csv"])
    pattern3 = "_".join(["(?P<name>(?:[A-Za-z]{3}){1,2}(?:[X0-9]{3}){1,2}[MF]?)",
                        "(?P<datestr>[0-9]*)",
                        "trialdata.csv"])
    @classmethod
    def parse(cls, files):

        blocks = list()
        for file in files:

            fname = os.path.split(file)[1]
            m = cls.parse_filename(fname)
            if m is not None:
                datetime = pd.to_datetime(m["datestr"])
                blk = objects.Block(name=cls.get_name(m["name"]),
                                    date=datetime.date(),
                                    start=datetime.time(),
                                    filename=file,
                                    data=cls.get_block_data(file))

                blocks.append(blk)
            else:
                print("Could not parse filename %s. Skipping" % file)

        return blocks

    @classmethod
    def parse_filename(cls, fname):

        m = re.match(cls.pattern, fname, re.IGNORECASE)
        if m is not None:
            m = m.groupdict()
            if m["name"] is None:
                m["name"] = "Unknown"
            if m["datestr"] is None:
                m["datestr"] = "Unknown"

            return m
        m = re.match(cls.pattern2, fname, re.IGNORECASE)
        if m is not None:
            m = m.groupdict()
            if m["name"] is None:
                m["name"] = "Unknown"
            if m["datestr"] is None:
                m["datestr"] = "Unknown"
            return m

        m = re.match(cls.pattern3, fname, re.IGNORECASE)
        if m is not None:
            m = m.groupdict()
            if m["name"] is None:
                m["name"] = "Unknown"
            if m["datestr"] is None:
                m["datestr"] = "Unknown"
            return m
    @classmethod
    def get_block_data(cls, csv_file):

        def rt_to_timedelta(rt):

            if rt != "nan":
                if rt == "":
                    return "nan"
                hours, minutes, seconds = [float(ss) for ss in rt.split(":")]
                deltadict = dict(hours=hours,
                                 minutes=minutes,
                                 seconds=seconds)
                return datetime.timedelta(**deltadict)
            else:
                return rt
        with open(csv_file) as in_file:
            csv_reader = csv.reader(in_file)
            labels = next(csv_reader)
        labels = [COL_LABELS.get(l,l.capitalize()) for l in labels]            
        custom_date_parser = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

        data = pd.read_csv(csv_file,
                           header=0,
                           names=labels,
                           index_col="Time",
                           converters={"RT": rt_to_timedelta,
                                       'Time': custom_date_parser})
        
        data["Time"] = data.index

        return data


class MatlabTxt(Importer):

    pattern = "".join(["(?:(?P<name>(?:[a-z]{2,3})+(?:[0-9]{2})+[mf]?)|.*)",
                       "_",
                       "(?:(?P<date>[\d\s]{6})|.*)",
                       "_",
                       "(?:(?P<time>[\d\s]{6})|.*)",
                       "_.*_",
                       "(?:(?P<file>[a-z]+)|.*)",
                       "\.txt"])

    def __init__(self):

        super(MatlabTxt, self).__init__()

    def parse(self, files):
        """ Takes in a list of files and returns a list of Block objects
        """
        block_groups = self.group_files(files)

        for file_grp in block_groups.values():
            files, mdicts = zip(*file_grp)
            blk = objects.Block()
            blk.name = self.get_name(mdicts[0]["name"])
            date = pd.to_datetime(mdicts[0]["date"], format="%y%m%d").date()
            time = pd.to_datetime(mdicts[0]["time"], format="%H%M%S").time()
            file_types = [m["file"] for m in mdicts]
            if "parameters" in file_types:
                fname = files[file_types.index("parameters")]
                blk.start_time, blk.first_peck, blk.end_time = self.parse_time_file(fname, date, time)
            else:
                blk.start_time = pd.Timestamp(pd.datetime.combine(date, time))

            if "timestamp" in file_types:
                fname = files[file_types.index("timestamp")]
                blk.data = self.get_block_data(fname, start=blk.start_time)
                if (blk.data is None) or (len(blk.data) <= 1):
                    continue

            blk.files = files
            self.blocks.append(blk)

        return self.blocks

    def parse_time_file(self, fname, date, default):
        """ Parses the file with "parameters" in its name to extract the session start, stop and first_peck times
        """

        with open(fname, "r") as f:
            contents = f.read()

        timestr = "\d{,2}\:\d{,2}\:\d{,2}"
        as_datetime = lambda ss: pd.Timestamp(pd.datetime.combine(date, pd.to_datetime(ss).time()))
        # Start time
        m = re.search("protocol.*?(%s)" % timestr, contents, re.IGNORECASE)
        if m is not None:
            start = as_datetime(m.groups()[0])
        else:
            start = pd.Timestamp(pd.datetime.combine(date, default))

        # Time of first peck
        m = re.search("first\speck.*?(%s)" % timestr, contents, re.IGNORECASE)
        if m is not None:
            first_peck = as_datetime(m.groups()[0])
        else:
            first_peck = None

        m = re.search("trial\sstopped.*?(%s)" % timestr, contents, re.IGNORECASE)
        if m is not None:
            stop = as_datetime(m.groups()[0])
        else:
            stop = None

        return start, first_peck, stop

    def get_block_data(self, fname, start):
        """ Gets the block data for file with "timestamp" in its name
        """

        data_labels = ["Timestamp", "Class", "Number"]
        start_value = start.value
        to_timestamp = lambda nsec: pd.Timestamp(start_value + nsec * 10 ** 9)
        to_class = lambda label: label == "GoStim"
        try:
            data = pd.read_csv(fname,
                                    delimiter="\t",
                                    header=0,
                                    names=data_labels,
                                    index_col="Timestamp",
                                    converters={"Class": to_class})
            data.index = map(to_timestamp, data.index)

        except pd.parser.CParserError:
            return

        return data

    def parse_filename(self, fname):
        """ Parses fname according to the regular expression MatlabTxt.pattern
        """

        m = re.match(self.pattern, fname, re.IGNORECASE)
        if m is not None:
            m = m.groupdict()
            if m["name"] is None:
                m["name"] = "Unknown"
            if m["date"] is None:
                m["date"] = "000101"
            else:
                m["date"] = m["date"].replace(" ", "0")
            if m["time"] is None:
                m["time"] = "000000"
            else:
                m["time"] = m["time"].replace(" ", "0")
            if m["file"] is None:
                m["file"] = "unknown"
            else:
                m["file"] = m["file"].lower()

            return m

    def group_files(self, files):
        """ Takes in a list of files and groups them according to their bird, date and timestamp
        Outputs a dictionary where the keys are the grouping and the values are a tuple of (filename, full regex match)
        """
        block_groups = dict()
        for fname in files:
            if os.path.exists(fname):
                if os.path.isdir(fname):
                    block_groups.update(**self.group_files(map(lambda x: os.path.join(fname, x), os.listdir(fname))))
                else:
                    m = self.parse_filename(os.path.basename(fname))
                    if m is None:
                        print("File does not match regex pattern: %s" % fname)
                        continue
                    key = "%s_%s_%s" % (m["name"], m["date"], m["time"])
                    block_groups.setdefault(key, list()).append((fname, m))
            else:
                print("File does not exist! %s" % fname)
                continue

        return block_groups


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse CSV files and store them in an hdf5 file")
    parser.add_argument("csv_files",
                         help="A list of CSV files separated by spaces",
                         nargs="+")
    parser.add_argument("output_file",
                        help="An hdf5 file where block data will be stored",
                        nargs=1,
                        type=str)
    args = parser.parse_args()
    csv_files = list()
    for cf in args.csv_files:
        csv_files.append(os.path.abspath(os.path.expanduser(cf)))

    blocks = PythonCSV.parse(args.csv_files)
    for blk in blocks:
        blk.save(os.path.abspath(os.path.expanduser(args.output_file[0])))
