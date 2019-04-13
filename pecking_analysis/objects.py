from __future__ import division, print_function
from itertools import product, groupby
import h5py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def merge_daily_blocks(blocks, date_range=None):
    blocks = list(sorted(blocks, key=lambda b: b.data.index[0]))

    new_blocks = []
    if date_range is not None:
        group = []
        for block in blocks:
            if (date_range[0] <= block.data.index[0].date() <= date_range[1]):
                group.append(block)

        if len(group) == 0:
            pass
        elif len(group) > 1:
            new_blocks.append(Block.merge(group))
        else:
            new_blocks.append(group[0])
    else:
        for _, group in groupby(blocks, lambda b: b.data.index[0].date()):
            group = list(group)
            if len(group) > 1:
                new_blocks.append(Block.merge(group))
            else:
                new_blocks.append(group[0])

    return new_blocks


class Block(object):
    '''
    This class organizes data from a single block of trials. It has attributes of:
    - name: the subject's name
    - date: the starting date of the block
    - start: the starting time of the block
    - data: a pandas dataframe of the block data
    - annotations: a dictionary of annotations for the block
    - first_peck: timestamp of the first peck
    - last_peck: timestamp of the last peck

    It also has useful methods:
    - save: Save the block (only hdf5 files are currently supported)
    - load: Load a block from the specified location
    - plot: Plot a quick representation of the data throughout the block
    - reject_double_pecks: Birds typically peck faster than the software responds and some must be rejected
 
    TODO: 
    - Add a method to reject the first one or two pecks of the day (testing pecks by human)
    - Add better plotting functions
    - Add organize_and_merge_blocks (?)
    '''

    first_peck = property(fget=lambda self: self.data.index[0], doc="The timestamp of the first peck")
    last_peck = property(fget=lambda self: self.data.index[-1], doc="The timestamp of the last peck")

    def __init__(self,
                 name=None,
                 date=None,
                 start=None,
                 data=None,
                 store=None,
                 **kwargs):
        """
        Creates a Block object that stores data about a single chunk of trials for the pecking test
        :param name: The bird's name
        :param date: The date of the block - A datetime.date
        :param start: A start time of the block - A datetime.time
        :param filename: The CSV filename where the data came from
        :param data: The imported pandas DataFrame
        :param store: An HDF5Store instance.
        :param kwargs: Any additional keyword arguments will be added as annotations
        """
        self.name = name
        self.date = date
        self.start = start
        if "OverallTrial" not in data:
            data["OverallTrial"] = data["Trial"]
        if "filename" in kwargs:
            data["Filename"] = pd.Series(
                [os.path.basename(kwargs["filename"])] * len(data),
                index=data.index
            )
            data["Date"] = pd.Series(
                [self.date] * len(data),
                index=data.index
            )
        self.data = data
        self.store = store

        self.annotations = dict()
        self.annotate(**kwargs)

    def __str__(self):

        output = ["%s Date: %s" % (self.name, self.date.isoformat())]
        output.append("Time: %s" % (self.start.isoformat()))
        if "filename" in self.annotations:
            output.append("Filename: %s" % (self.annotations["filename"]))

        g = self.data[["Response", "Class"]].groupby("Class")
        c = g["Response"].count().to_frame().transpose().rename({"Response": "Played"})
        m = g["Response"].mean().to_frame().transpose().rename({"Response": "Fraction Interrupt"})
        output.extend(("%s" % c.append(m)).splitlines())

        if len(self.annotations):
            output.append("Annotations:")
            for key, val in self.annotations.items():
                output.append("\t%s = %s" % (str(key), str(val)))

        return "\n".ljust(13).join(output)

    def annotate(self, **annotations):
        """
        Add an annotation to the block
        :param annotations:
        :return:
        """

        self.annotations.update(annotations)
        if self.store is not None:
            return self.store.annotate_block(self, **self.annotations)

        return True

    def filter_conditions(self, conditions):
        self.data = self.data.iloc[np.in1d(self.data["Class"], conditions)]

    @classmethod
    def merge(cls, blocks):
        """
        Merges all of the blocks into a single Block object. Useful if multiple runs of the same condition got
        accidentally separated (e.g. hardware malfunction causing you to run it twice).
        The merging requires that all blocks have the same name attribute (or None). It will take the earliest date
        and start time as the date and start attributes. The filename attribute is set to None, but the resulting
        block will have a "filenames" annotation that is a list of all merged filename attributes.
        :param blocks: a list of Block objects for each individual CSV that you want merged.
        :return: A single Block object instance
        """

        earliest = None
        filenames = list()
        name = None
        data = pd.DataFrame()
        for blk in blocks:
            datetime = pd.datetime.combine(blk.date, blk.start)
            if earliest is not None:
                if datetime < earliest:
                    earliest = datetime
            else:
                earliest = datetime

            if name is not None:
                if (blk.name is not None) and (blk.name != name):
                    ValueError("Blocks do not come from the same bird. Make sure all of the names are the same!")
            else:
                name = blk.name

            filenames.append(blk.annotations.get("filename", blk.annotations.get("filenames")))
            data = pd.concat([data, blk.data])

        data["OverallTrial"] = pd.Series(np.arange(len(data)), index=data.index)

        return cls(name=name,
                   date=earliest.date(),
                   start=earliest.time(),
                   data=data,
                   filenames=filenames)

    def save(self, filename=None, overwrite=False):
        """
        Save the block. If the store attribute is not set, then you must provide a filename.
        :param filename: hdf5 file
        :param overwrite: Whether or not to overwrite if the data already exists (default False)
        :return: True if saving was successful
        """

        if filename is not None:
            if filename.endswith((".h5", ".hdf5", ".hdf")):
                self.store = HDF5Store(filename)
            else:
                print("Only .h5 files are currently supported")

        if self.store is not None:
            return self.store.save_block(self, overwrite=overwrite)
        else:
            return False

    @classmethod
    def load(cls, store, path):
        """
        Loads a block object from the specified storage file at the specified path
        :param store: the store or name of the store file.
        :param path: the path to the group within the hdf5 file where the block is stored
        :return: a Block instance
        """

        if isinstance(store, str):
            if store.endswith((".h5", ".hdf5", ".hdf")):
                store = HDF5Store(store)
            else:
                print("Only .h5 files are currently supported")
                return

        block = store.load_block(path)
        block.store = store

        return block

    def reject_double_pecks(self, rejection_threshold=200):
        """Remove trials that are interrupted too quickly

        :param rejection_threshold: minimum intertrial duration in ms
        """
        good_trials = np.where(
            np.diff(self.data.index).astype('timedelta64[ms]') >= np.timedelta64(rejection_threshold, "ms")
        )[0]
        good_trials = np.concatenate([good_trials, [len(self.data) - 1]])
        self.data = self.data.iloc[good_trials]

    def plot(self,
            window_size=20,
            filename=None,
            split_on_columns=None
        ):
        """

        split_on_columns:
            list of column(s) to group data by to plot,
            if left blank, defaults to ["Class"]
        """

        fig = plt.figure(facecolor="white", edgecolor="white", figsize=(15, 3))
        ax = fig.gca()

        # class_names = self.data["Class"].unique().tolist()
        # call_names = self.data["Call Type"].unique().tolist()

        if split_on_columns is None:
            split_on_columns = ["Class"]

        categories = sorted(list(product(*[self.data[col].unique().tolist() for col in split_on_columns])))


        # class_call_pairings = list(product(class_names, call_names))
        # convert_rt = lambda x: x.total_seconds() if x != "nan" else np.nan
        grouped = self.data.groupby(split_on_columns)

        try:
            import palettable
            colors = palettable.tableau.ColorBlind_10.mpl_colors
        except ImportError:
            colors = None

        if colors is None or len(categories) > len(colors):
            colors = plt.get_cmap("gist_ncar")
            colors = [colors(ff) for ff in np.linspace(0, 1, len(categories))]

        for ii, cn in enumerate(categories):
            try:
                if len(cn) == 1:
                    g = grouped.get_group(cn[0])
                else:
                    g = grouped.get_group(cn)
            except KeyError:
                continue

            print(cn, len(g))

            pd.rolling_mean(g["Response"],
                            window_size,
                            center=True).plot(ax=ax,
                                              color=colors[categories.index(cn)],
                                              linewidth=2,
                                              label=cn)
            # pd.rolling_mean(g[g["Response"] == 1]["RT"].apply(convert_rt),
            #                 window_size, center=True).plot(ax=ax2,
            #                                                color=colors[ii],
            #                                                linewidth=2,
            #                                                linestyle="--",
            #                                                label="%s RT" % cn)

        # inds = self.data[self.data["Response"] == 1].index
        # c = [colors[class_names.index(cn)] for cn in self.data.loc[inds]["Class"].values]

            inds = g.index
            ax.scatter(inds, np.ones((len(inds),)), s=100, color=colors[categories.index(cn)], marker="|", edgecolor="face")
        # inds = self.data.index
        # c = [colors[class_call_pairings.index(cn)] for cn in class_call_pairings] #self.data["Class"].values]
        # ax.scatter(inds, np.ones((len(inds),)), s=100, c=c, marker="|", edgecolor="face")
        ax.set_ylim((-0.1, 1.1))

        for loc in ["right", "top"]:
            ax.spines[loc].set_visible(False)
        # for loc in ["left", "top"]:
        #     ax2.spines[loc].set_visible(False)

        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.grid(False)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.grid(False)
        ax.set_ylabel("Fraction Interrupt")
        ax.set_title("%s - %s" % (self.name, self.date.strftime("%a, %B %d %Y")))

        # ax2.yaxis.grid(False)
        # ax2.set_ylabel("Reaction Time (s)")

        ax.legend(loc="upper right", bbox_to_anchor=(0.0, 1.0), frameon=False)
        # ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)

        if filename is not None:
            fig.savefig(filename, dpi=450, facecolor="white", edgecolor="white")


class HDF5Store(object):

    def __init__(self, filename):
        """ Implements storing Block object data in an hdf5 file.
        :param filename: path to an hdf5 file
        """

        self.filename = filename

        # Ensure the file exists
        if not os.path.isfile(filename):
            with h5py.File(filename, "w") as hf:
                pass

    def annotate_block(self, block, **kwargs):
        """ Annotate the block with key-value pairs in kwargs
        :param block: a Block instance
        :param kwargs: key-value pairs to store as annotations
        """

        # Store None values as strings
        or_none = lambda val: val if val is not None else "none"

        group_name = self._group_name(block)
        with h5py.File(filename, "a") as hf:
            g = hf.get(group_name)
            if g is not None:
                for key, val in kwargs.items():
                    g.attrs[key] = or_none(val)

    def save_block(self, block, overwrite=True):
        """ Save the block in the hdf5 file
        :param block: Block instance to save
        :param overwrite: whether or not to overwrite the existing data if the block has already been stored. (default True)
        """

        # Store None values as strings
        or_none = lambda val: val if val is not None else "none"

        # Store all annotations, as well as top-level attributes as attributes on the group for the block
        with h5py.File(self.filename, mode="a") as hf:
            # File is structured /bird_name
            group_name = self._group_name(block)
            g = self._create_group_recursive(hf, group_name, overwrite)

            g.attrs["name"] = block.name
            g.attrs["date"] = block.date.strftime("%d%m%Y")
            g.attrs["start"] = block.start.strftime("%H%M%S")
            for key, val in block.annotations.items():
                g.attrs[key] = or_none(val)

        # Store the data using pandas built-in to_hdf method
        block.data.to_hdf(self.filename, group_name + "/data")

        # Records of which data is where in the file are kept in a table called "values" at the root of the file
        # Load that table if it exists
        try:
            values = pd.read_hdf(self.filename, "/values")
        except KeyError:
            values = None

        # Add the table entry if it doesn't yet exist
        if (values is None) or (str(group_name) not in values["Path"].values):
            df = pd.DataFrame({"Name": block.name,
                               "Timestamp": pd.Timestamp(pd.datetime.combine(block.date, block.start)),
                               "Path": str(group_name)},
                               index=[0])
            df = df.set_index("Timestamp")
            df.to_hdf(self.filename, "/values", format="table", append=True)

        return True

    def load_block(self, path):
        """ Load the block at the specified path
        :param path: hdf5 group path to load
        """

        # Return "none" strings as None
        or_none = lambda val: val if (not isinstance(val, str) or (val != "none")) else None

        # Load the data
        data = pd.read_hdf(self.filename, path + "/data")

        # Load the annotatons and top-level attributes
        with h5py.File(self.filename, "r") as hf:
            g = hf.get(path)
            annotations = dict(g.attrs.items())
            name = annotations.pop("name")
            date = pd.datetime.strptime(annotations.pop("date"), "%d%m%Y").date()
            start = pd.datetime.strptime(annotations.pop("start"), "%H%M%S").time()

            for key, val in annotations.items():
                annotations[key] = or_none(val)

        return Block(name=name,
                     date=date,
                     start=start,
                     data=data,
                     **annotations)


    @staticmethod
    def _group_name(block):

        if block.name is None:
            ValueError("Cannot save to hdf5 file when block.name is None")
        if block.date is None:
            ValueError("Cannot save to hdf5 file when block.date is None")
        if block.start is None:
            ValueError("Cannot save to hdf5 file when block.start is None")

        return "/" + "/".join([block.name, block.date.strftime("%d%m%Y"), block.start.strftime("%H%M%S")])

    @staticmethod
    def _create_group_recursive(hf, group_name, overwrite):

        group = hf
        group_names = group_name.split("/")
        for ii, group_name in enumerate(group_names):
            if group_name == "":
                continue
            if group_name in group:
                if ii == (len(group_names) - 1):
                    if overwrite:
                        del group[group_name]["data"]
                        del group[group_name]
                    else:
                        raise IOError("Block %s has already been imported into %s. To overwrite add overwrite=True" %
                                      (group_name, hf.filename))
                else:
                    group = group[group_name]
                    continue

            group = group.create_group(group_name)

        return group



def plot_interruption_rates(blocks):

    df = pd.DataFrame()
    for blk in blocks:
        if len(blk.data) > 0:
            df[blk.date] = blk.data.groupby("Class")["Response"].mean()
    df = df.T.sort_index()

    df.plot()
    plt.title(blocks[0].name)
    plt.ylim((0.0, 1.0))


def get_blocks(filename, date=None, start_date=None, end_date=None, birds=None):
    """
    Get all blocks from the hdf5 file filename that match certain criteria
    :param filename: An hdf5 file
    :param date: A specific date (format: "yyyy-mm-dd"). Overrides start_date and end_date.
    :param start_date: Beginning date (format: "yyyy-mm-dd")
    :param end_date: End date (format: "yyyy-mm-dd")
    :param birds: a list of bird names to select
    :return: a list of Block objects
    """

    df = pd.read_hdf(filename, "/values")
    df = filter_block_metadata(df, date=date, start_date=start_date,
                               end_date=end_date, birds=birds)
    df = df.sort_index().sort("Name")
    paths = df["Path"].values

    return [Block.load(filename, path) for path in paths]


def filter_blocks(blocks, **kwargs):
    """ Filter the list of blocks using the key-value pairs in kwargs
    :param blocks: a list of block objects
    :param kwargs: key-value pairs to be matched with block attributes/annotations
    """
    results = list()
    for blk in blocks:
        match = True
        for key, value in kwargs.items():
            if hasattr(blk, key):
                if getattr(blk, key) != value:
                    match = False
                    break
            elif key in blk.annotations:
                if blk.annotations[key] == value:
                    match = False
                    break
            else:
                match = False

        if match:
            results.append(blk)

    return results


def filter_block_metadata(df, date=None, start_date=None, end_date=None, birds=None, **kwargs):
    """
    Get all blocks from a loaded dataframe that match certain criteria
    :param df: Dataframe read from a HDF5Store.
    :param date: A specific date (format: "yyyy-mm-dd"). Overrides start_date and end_date.
    :param start_date: Beginning date (format: "yyyy-mm-dd")
    :param end_date: End date (format: "yyyy-mm-dd")
    :param birds: a list of bird names to select
    :return: a filtered Dataframe
    """

    if date is not None:
        df = df.ix[date]
    else:
        if start_date is not None:
            df = df.ix[start_date:]
        if end_date is not None:
            df = df.ix[:end_date]

    if birds is not None:
        if isinstance(birds, list):
            df = df[df["Name"].isin(birds)]
        else:
            df = df[df["Name"] == birds]

    return df


def summarize_file(filename, date=None, start_date=None, end_date=None, birds=None):
    """
    Summarize the data stored in the filename that match certain criteria
    :param filename: An hdf5 file
    :param date: A specific date (format: "yyyy-mm-dd"). Overrides start_date and end_date.
    :param start_date: Beginning date (format: "yyyy-mm-dd")
    :param end_date: End date (format: "yyyy-mm-dd")
    :param birds: a list of bird names to select
    """

    df = pd.read_hdf(filename, "/values")
    df = filter_block_metadata(df, date=date, start_date=start_date,
                               end_date=end_date, birds=birds)
    df = df.rename(columns={"Path": "File count"})
    return df.groupby("Name").count().sort("File count", ascending=False)


def export_csvs(args):
    from pecking_analysis.utils import get_csv, convert_date
    from pecking_analysis.importer import PythonCSV

    if (args.date is None) and (args.bird is None):
        args.date = "today"

    date = convert_date(args.date)
    csv_files = get_csv(data_dir, date=date, bird=args.bird)

    blocks = PythonCSV.parse(csv_files)
    for blk in blocks:
        blk.save(args.filename, args.overwrite)


if __name__ == "__main__":
    import sys
    import argparse

    h5_file = os.path.abspath(os.path.expanduser("~/data/flicker_fusion.h5"))
    parser = argparse.ArgumentParser(description="Export CSV files to h5 file")
    parser.add_argument("-d", "--date", dest="date", help="Date in the format of DD-MM-YY (e.g. 14-12-15) or one of \"today\" or \"yesterday\"")
    parser.add_argument("-b", "--bird", dest="bird", help="Name of bird to check. If not specified, checks all birds for the specified date")
    parser.add_argument("-f", "--filename", dest="filename", help="Path to h5 file", default=h5_file)
    parser.add_argument("--overwrite", help="Overwrite block in h5 file if it already exists", action="store_true")
    parser.set_defaults(func=export_csvs)

    if len(sys.argv) == 1:
        parser.print_usage()
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)
