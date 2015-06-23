import os
from matplotlib import pyplot as plt

def get_filename_frequency(filename):

    return int(os.path.basename(filename).split("_")[1])

def get_response_by_frequency(blk):

    freq_df = blk.data["Stimulus"].apply(get_filename_frequency).to_frame()
    freq_df["Response"] = blk.data["Response"]
    grouped = freq_df.groupby("Stimulus")
    c = grouped.count()
    c["Index"] = range(len(c))
    m = grouped.mean()
    m["Index"] = range(len(m))
    freqs = m.index
    ax = m.plot(x="Index", y="Response")
    ax.set_ylim([0, 1])
    ax.xaxis.set_ticks(m["Index"])
    ax.xaxis.set_ticklabels(freqs)
    for ii in ax.xaxis.get_ticklocs():
        plt.text(ii, 1.0, "%d" % c["Response"].iat[ii])

if __name__ == "__main__":

    import sys
    import os
    from pecking_analysis import importer

    csv_file = os.path.abspath(os.path.expanduser(sys.argv[1]))
    print("Attempting to parse file %s" % csv_file)
    if not os.path.exists(csv_file):
        raise IOError("File %s does not exist!" % csv_file)

    csv_importer = importer.PythonCSV()
    blocks = csv_importer.parse([csv_file])
    get_response_by_frequency(blocks[0])
