import os
import re
from collections import defaultdict

import pandas as pd


def parse_filename(filename):
    """Funky function that searches a filename for stimulus metadata

    Searches for call type ("Te", "So", "Song", ...]
    Searches for a bird name (Col00 or ColCol0000)
    Searches for a rendition (thirds section after splitting by _)
    """
    sections = filename.split("_")
    if sections[-1] == "norm":
        sections.pop()

    for i, section in enumerate(sections):
        if section.upper() in ["SO", "SONG"]:
            call_type = "SO"
            break
        elif section.upper() in ["TE", "BE", "NE", "DC", "AG", "LT", "TH"]:
            call_type = section.upper()
            break
    else:
        call_type = "unknown"

    for i, section in enumerate(sections):
        if re.search("[a-zA-Z]{6}[a-zA-Z0-9]{4}", section):
            bird_name = section
            rendition = "_".join([section, sections[i-1]])
            break
        if re.search("[a-zA-Z]{3}[a-zA-Z0-9]{2}", section):
            bird_name = section
            rendition = "_".join([section, sections[i-1]])
            break

    else:
        bird_name = filename
        if len(sections) < 3:
            rendition = "0"
        else:
            rendition = sections[2]
        
    return {
        "call_type": call_type,
        "bird_name": bird_name,
        "rendition": rendition,
    }


def preprocess_stimuli(block):
    """Preprocessing of stimuli information in block
    
    Adds new columns to block's dataframe, and creates a new separate dataframe
    with stimulus information. Each dataframe is given a new column called "Stim Key"
    that is sufficient for joining these dataframes later.
    
    Columns added to block:
        Stimulus Name - the stimulus wav file with the rest of the path and .wav stripped
        Stim Key - a key (used to join stimuli to the corresponding rows in trials df)
        Bird Name - read from filename
        Call Type - read from filename
        Rendition - read from filename
        
    Columns in stim df:
        Stim Key - key for joining with block df
        Bird Name - Name of vocalizer
        Call Type - Call type of vocalization
        Class - Rewarded or Unrewarded (if the stim was rewarded within this block)
        Trials - Number of trials that this stimulus file was played
    """
    block.data["Stimulus Name"] = block.data["Stimulus"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )

    call_type = []
    bird_name = []
    rendition = []
    for i, row in block.data.iterrows():
        filename_info = parse_filename(row["Stimulus Name"])
        call_type.append(filename_info["call_type"])
        bird_name.append(filename_info["bird_name"])
        rendition.append(filename_info["rendition"])
    
    block.data["Call Type"] = pd.Series(call_type, index=block.data.index)
    block.data["Bird Name"] = pd.Series(bird_name, index=block.data.index)
    block.data["Rendition"] = pd.Series(rendition, index=block.data.index)
    block.data["Stim Key"] = block.data[["Bird Name", "Call Type", "Class"]].apply(
        lambda x: " ".join(x)
    )

    split_on_columns = ["Bird Name", "Call Type", "Class"]
    grouped = block.data.groupby(split_on_columns)

    stim_data = []
    for (bird_name, call_type, rewarded), t in sorted(grouped.groups.items(), key=lambda x: (x[0][1], len(x[0][0]), x[0][0])):
        stim_data.append([" ".join((bird_name, call_type, rewarded)), bird_name, call_type, rewarded, len(t)])
    
    stims = pd.DataFrame(stim_data, columns=["Stim Key", "Bird Name", "Call Type", "Class", "Trials"])
    stims.set_index("Stim Key")

    block.stimuli = stims
    return block, stims


def insert_stimulus_history(stimulus_blocks):
    """Insert columns describing the stimulus history into a series of stimulus blocks

    A stimulus is described by its Stim Key; thus, all stim files with the same call type,
    bird name, and reward class, are considered the same "stimulus"

    Inserts into each dataframe three columns:
        Overall Seen Before: how many sessions this stimulus has been seen
        Consecutive: how many sessions consecutively this stimulus has been seen
        New: boolean flag if the stimulus is new or seen previously

    Parameters
    ----------
    stimulus_blocks : list
        a list of stimulus dataframes (produced by preprocess_stimuli()), that are
        in chronological order and ideally are grouped into one block per day

    Returns
    -------
    A list of stimulus dataframes that have been updated with the columns of
    Overall Seen Before, Consecutive, and New
    """
    overall_stims = defaultdict(int)
    consec_stims = defaultdict(int)

    all_stims = []
    stims = None
    for stims in stimulus_blocks:
        # Keep track of which stims we are seeing consecutively
        # from the day before. If we didn't see it for one day, clear
        # that out
        if stims is not None:
            for stim in stims["Stim Key"]:
                consec_stims[stim] += 1
                overall_stims[stim] += 1

            for k in list(consec_stims.keys()):
                if k not in stims["Stim Key"].unique():
                    del consec_stims[k]

        times_seen = []
        consecutive_seen = []
        for i, row in stims.iterrows():
            times_seen.append(overall_stims[row["Stim Key"]])
            consecutive_seen.append(consec_stims[row["Stim Key"]])
            filename_info = parse_filename(row["Stim Key"])

        stims["Overall Seen Before"] = pd.Series(times_seen, index=stims.index)
        stims["Consecutive"] = pd.Series(consecutive_seen, index=stims.index)
        stims["New"] = stims["Consecutive"].apply(lambda x: x == 0)

        stims = stims.sort_values(["Class", "New", "Bird Name"], ascending=[True, False, True])
        all_stims.append(stims)
        
    return all_stims
