import os
import datetime as dt


def convert_date(date):
    """
    Converts the date given to %d%m%y. Can be of the form %d-%m-%y or one of 'today' or 'yesterday'    """

    if date is not None:
        if date.lower() == "today":
            date = dt.date.today()
        elif date.lower() == "yesterday":
            date = dt.date.today() - dt.timedelta(days=1)
        else:
            try:
                date = dt.datetime.strptime(date, "%d-%m-%y").date()
            except ValueError:
                raise ValueError("date must be specified as \"dd-mm-yy\" (e.g. 14-12-15 for December 14, 2015)")
        date = date.strftime("%d%m%y")

        return date


def get_csv(directory, date=None, bird=None):
    """
    Returns all of the csv files from the data directory hierarchy.
    If date is None, then any date works
    If bird is None, then any bird works
    Returns a list of csv files that match date and bird
    """
    csv_files = list()
    if bird is not None:
        dirnames = [os.path.join(directory, bird)]
        if not os.path.isdir(dirnames[0]):
            raise IOError("Bird %s does not exist!" % bird)
    else:
        dirnames = list()
        for dirname in os.listdir(directory):
            dirname = os.path.join(directory, dirname)
            if os.path.isdir(dirname):
                dirnames.append(dirname)

    datedirs = list()
    for dirname in dirnames:
        if date is not None:
            datedir = os.path.join(dirname, date)
            if not os.path.isdir(datedir):
                continue
            else:
                datedirs.append(datedir)
        else:
            datedirs = list()
            for datedir in os.listdir(dirname):
                datedir = os.path.join(dirname, datedir)
                if os.path.isdir(datedir):
                    datedirs.append(datedir)

    for dirname in datedirs:
        filenames = [fname for fname in os.listdir(dirname) if fname.lower().endswith(".csv")]
        csv_files.extend([os.path.join(dirname, fname) for fname in filenames])

    return csv_files
