import os
import re
import ipdb

import numpy as np
import pandas as pd

from pecking.block import Block


bird_dict = {"BlaBlu1387F": "BlaLbu1387F",
             "GraGre1401M": "GraGre4401M",
             "BlaYel0208": "BlaYel0208M",
             "WhiXXX4100F": "WhiRas4100F"}

class Importer(object):

    def parse(self, files):

        pass

    def get_name(self, bird_name):

        if bird_name in bird_dict:
            return bird_dict[bird_name]
        else:
            return bird_name

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

        self.blocks = list()

    def parse(self, files):

        block_groups = self.group_files(files)

        for file_grp in block_groups.values():
            files, mdicts = zip(*file_grp)
            blk = Block()
            blk.name = self.get_name(mdicts[0]["name"])
            date = pd.to_datetime(mdicts[0]["date"], format="%y%m%d").date()
            time = pd.to_datetime(mdicts[0]["time"], format="%H%M%S").time()
            file_types = [m["file"] for m in mdicts]
            if "parameters" in file_types:
                fname = files[file_types.index("parameters")]
                blk.start, blk.first_peck, blk.end = self.parse_time_file(fname, date, time)
            else:
                blk.start = pd.Timestamp(pd.datetime.combine(date, time))

            if not blk.is_complete:
                continue

            if "timestamp" in file_types:
                fname = files[file_types.index("timestamp")]
                blk.data = self.get_block_data(fname, start=blk.start)
                if (blk.data is None) or (len(blk.data) <= 1):
                    continue
                blk.compute_statistics()

            blk.files = files
            self.blocks.append(blk)

        return self.blocks

    def parse_time_file(self, fname, date, default):
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

        if isinstance(files, str):
            files = [files]

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