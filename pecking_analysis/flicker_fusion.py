import os


def get_filename_frequency(filename):

    return int(os.path.basename(filename).split("_")[1])

def get_response_by_frequency(blk):

    freq_df = blk.data["Stimulus"].apply(get_filename_frequency).to_frame()
    freq_df["Response"] = blk.data["Response"]
    grouped = freq_df.groupby("Stimulus")


