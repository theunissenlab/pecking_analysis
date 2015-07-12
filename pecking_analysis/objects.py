from __future__ import division, print_function
import h5py
import os
import pandas as pd

class Block(object):
    '''
    This class organizes data from a single block of trials.
    '''

    first_peck = property(fget=lambda self: self.data.index[0], doc="The timestamp of the first peck")
    last_peck = property(fget=lambda self: self.data.index[-1], doc="The timestamp of the last peck")

    def __init__(self,
                 name=None,
                 date=None,
                 start=None,
                 filename=None,
                 data=None,
                 **kwargs):
        """
        Creates a Block object that stores data about a single chunk of trials for the pecking test
        :param name: The bird's name
        :param date: The date of the block - A datetime.date
        :param start: A start time of the block - A datetime.time
        :param filename: The CSV filename where the data came from
        :param data: The imported pandas DataFrame
        :param kwargs: Any additional keyword arguments will be added as annotations
        """

        self.name = name
        self.date = date
        self.filename = filename
        self.start = start
        self.data = data

        self.annotations = dict()
        self.annotate(**kwargs)

    def __str__(self):

        output = ["%s Date: %s" % (self.name, self.date.isoformat())]
        output.append("Time: %s" % (self.start.isoformat()))
        output.append("Filename: %s" % (self.filename))

        g = self.data[["Response", "Class"]].groupby("Class")
        c = g["Response"].count().to_frame().transpose().rename({"Response": "Played"})
        m = g["Response"].mean().to_frame().transpose().rename({"Response": "Fraction Interrupt"})
        output.extend(("%s" % c.append(m)).splitlines())

        if len(self.annotations):
            output.append("Annotations:")
            for key, val in self.annotations.iteritems():
                output.append("\t%s = %s" % (str(key), str(val)))

        return "\n".ljust(13).join(output)

    def annotate(self, **annotations):
        """
        Add an annotation to the block
        :param annotations:
        :return:
        """

        self.annotations.update(annotations)

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

            filenames.append(blk.filename)
            data = pd.concat([data, blk.data])

        return cls(name=name,
                   date=earliest.date(),
                   start=earliest.time(),
                   data=data,
                   filenames=filenames)

    def save(self, filename, overwrite=False):
        """
        Store the block in the hdf5 file named filename
        :param filename: hdf5 file
        :param overwrite: Whether or not to overwrite if the data already exists (default False)
        :return: True if saving was successful
        """

        if filename.endswith((".h5", ".hdf5", ".hdf")):
            return HDF5Store.save_block(filename, self, overwrite=overwrite)
        else:
            print("Only .h5 files are currently supported")

    @classmethod
    def load(cls, filename, path):
        """
        Loads a block object from the hdf5 file at the specified path
        :param filename: the path to the hdf5 file
        :param path: the path to the group within the hdf5 file where the block is stored
        :return:
        """

        if filename.endswith((".h5", ".hdf5", ".hdf")):
            return HDF5Store.load_block(filename, path)
        else:
            print("Only .h5 files are currently supported")

    def plot(self):
        pass


class HDF5Store(object):

    @classmethod
    def save_block(cls, filename, blk, overwrite=True):

        if blk.name is None:
            ValueError("Cannot save to hdf5 file when block.name is None")
        if blk.date is None:
            ValueError("Cannot save to hdf5 file when block.date is None")
        if blk.start is None:
            ValueError("Cannot save to hdf5 file when block.start is None")

        or_none = lambda val: val if val is not None else "none"

        with h5py.File(filename, mode="a") as hf:
            # File is structured /bird_name
            g = cls.create_group_recursive(hf, blk, overwrite)
            group_name = g.name

            g.attrs["name"] = blk.name
            g.attrs["date"] = blk.date.strftime("%d%m%Y")
            g.attrs["start"] = blk.start.strftime("%H%M%S")
            g.attrs["filename"] = or_none(blk.filename)
            for key, val in blk.annotations.iteritems():
                g.attrs[key] = or_none(val)

        blk.data.to_hdf(filename, group_name + "/data")
        df = pd.DataFrame({"Name": blk.name,
                           "Timestamp": pd.Timestamp(pd.datetime.combine(blk.date, blk.start)),
                           "Path": str(group_name)},
                          index=[0])
        df = df.set_index("Timestamp")
        df.to_hdf(filename, "/values", format="table", append=True)

        return True

    @classmethod
    def load_block(cls, filename, path):

        or_none = lambda val: val if (not isinstance(val, str) or (val != "none")) else None
        data = pd.read_hdf(filename, path + "/data")
        with h5py.File(filename, "r") as hf:
            g = hf.get(path)
            annotations = dict(g.attrs.items())
            name = annotations.pop("name")
            date = pd.datetime.strptime(annotations.pop("date"), "%d%m%Y").date()
            start = pd.datetime.strptime(annotations.pop("start"), "%H%M%S").time()
            filename = or_none(annotations.pop("filename"))

            for key, val in annotations.iteritems():
                annotations[key] = or_none(val)

        return Block(name=name,
                     date=date,
                     start=start,
                     filename=filename,
                     data=data,
                     **annotations)

    @staticmethod
    def create_group_recursive(hf, blk, overwrite):

        group = hf
        group_names = [blk.name, blk.date.strftime("%d%m%Y"), blk.start.strftime("%H%M%S")]
        for ii, group_name in enumerate(group_names):
            if group_name in group:
                if ii == (len(group_names) - 1):
                    if overwrite:
                        values = pd.read_hdf(hf, "/values")
                        values = values.loc[values["Path"] != group[group_name].name]

                        del group[group_name]
                        del hf["values"]
                        
                        values.to_hdf(hf, "/values", format="table", append=True)
                    else:
                        IOError("Block %s has already been imported into %s. To overwrite add overwrite=True" % (blk,
                                                                                                                 hf.filename))
                else:
                    group = group[group_name]
                    continue
            else:
                group = group.create_group(group_name)

            # try:
            #     g = group[group_name]
            #     if ii == (len(group_names) - 1):
            #         if overwrite:
            #             del g
            #             KeyError("")
            #         else:
            #             IOError("Block %s has already been imported into %s. To overwrite add overwrite=True" % (blk,
            #                                                                                                      hf.filename))
            # except KeyError:
            #     group = group.create_group(group_name)
            # else:
            #     group = g

        return group

def get_blocks(filename, start_date=None, end_date=None, birds=None):
    """
    Get all blocks from the hdf5 file filename that match certain criteria
    :param filename: An hdf5 file
    :param start_date: Beginning date (format: "yyyy-mm-dd")
    :param end_date: End date (format: "yyyy-mm-dd")
    :param birds: a list of bird names to select
    :return: a list of Block objects
    """

    df = pd.read_hdf(filename, "/values")
    if start_date is not None:
        df = df.ix[start_date:]
    if end_date is not None:
        df = df.ix[:end_date]
    if birds is not None:
        if isinstance(birds, list):
            df = df[df["Name"].isin(birds)]
        else:
            df = df[df["Name"] == birds]

    df = df.sort("Name")
    paths = df["Path"].values

    return [Block.load(filename, path) for path in paths]




