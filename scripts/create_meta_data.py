import os
import glob
import re
from mne.io import read_raw_eeglab

import pandas as pd
from tqdm import tqdm

import gc


def extract_metadata(
    raw,
    stage: str = "pre",
    stage_no: int = -1,
    patient: str = "000",
    condition_number: int = -1,
    filepath: str = None,
):

    # channel_locations = extract_channel_base_on_location(raw)
    channels = raw.ch_names.copy()
    channels.sort()

    metadata = {
        "stage": stage,
        "stage_no": stage_no,
        "patient": patient,
        "condition_number": condition_number,
        "filename": os.path.basename(raw.filenames[0]),
        "channels": channels,
        "n_channels": len(raw.info["ch_names"]),
        "duration": raw.times[-1],  # in seconds
        "sample_rate": raw.info["sfreq"],
        "filepath": filepath,
    }

    # for key, value in channel_locations.items():
    #     metadata[key] = value
    #     metadata[f"n_{key}"] = len(value)

    return metadata


def extract_channel_base_on_location(raw):
    # Group electrodes by region

    def get_channels(electrodes, location: str = "frontal"):
        channels_list = []
        if location == "frontal":

            for e in electrodes:
                if e.startswith("F"):
                    channels_list.append(e)
                    # remove e from electrodes
                    electrodes.remove(e)

        elif location == "central":
            for e in electrodes:
                if e.startswith("C"):
                    channels_list.append(e)
                    electrodes.remove(e)

        elif location == "parietal":
            for e in electrodes:
                if e.startswith("P"):
                    channels_list.append(e)
                    electrodes.remove(e)

        elif location == "occipital":
            for e in electrodes:
                if e.startswith("O"):
                    channels_list.append(e)
                    electrodes.remove(e)

        elif location == "temporal":
            for e in electrodes:
                if e.startswith("T"):
                    channels_list.append(e)
                    electrodes.remove(e)

        elif location == "anterior":
            for e in electrodes:
                if e.startswith("A"):
                    channels_list.append(e)
                    electrodes.remove(e)

        channels_list.sort()

        return channels_list

    electrodes = raw.ch_names.copy()
    # print(electrodes)
    output = {
        key: get_channels(electrodes, key)
        for key in [
            "frontal",
            "central",
            "parietal",
            "occipital",
            "temporal",
            "anterior",
        ]
    }

    return output


def get_file_paths(directory):
    return glob.glob(os.path.join(directory, "*.set"))


def extract_condition_number(fname):

    match = re.search(r"Cond(\d+)", fname)

    if match:
        return match.group(1)
    else:
        return -1  # if no condition number is found

def extract_patient_number(fname):
    match = re.search(r'_(\d+)_', fname)
    if match:
        return match.group(1)

    return "000"


def generate_metadata(input_fnames, eog=(), verbose=True):
    data = {"data": [], "metadata": []}
    for input_fname in input_fnames:
        basename_fname = os.path.basename(input_fname)
        if "PRE" in basename_fname or "pre" in basename_fname:
            stage, stage_no = "pre", 1
        elif "POST" in basename_fname or "post" in basename_fname:
            stage, stage_no = "post", 3
        elif "DURING" in basename_fname or "during" in basename_fname:
            stage, stage_no = "during", 2
        else:
            stage, stage_no = "unknown", -1

        condition_number = extract_condition_number(input_fname)
        patient_number = extract_patient_number(input_fname)

        try:
            raw = read_raw_eeglab(
                input_fname=input_fname,
                eog=eog,
                preload=True,
                montage_units="mm",
                verbose=verbose,
            )

            # print(raw.ch_names)

            # break
            meta = extract_metadata(
                raw,
                stage=stage,
                stage_no=stage_no,
                patient=patient_number,
                condition_number=condition_number,
                filepath=input_fname,
            )
            
            # Delete the raw object to free up memory
            del raw
            
            # Force garbage collection
            gc.collect()

        except Exception as e:
            print(f"Error in reading {input_fname}: {e}")
            continue
            # break

        data["metadata"].append(meta)
        # data["data"].append(raw)

    return data


def main():

    verbose = 0
    directory = "C:\Data"
    input_fnames = get_file_paths(directory=directory)
    data = generate_metadata(input_fnames, verbose=verbose)

    columns = list(data["metadata"][0].keys())
    df = pd.DataFrame(data["metadata"], columns=columns)
    # df.sort_values(by=["condition_number", "stage_no"], inplace=True)
    df.sort_values(by=["condition_number", "patient", "stage_no"], inplace=True)
    
    df.to_csv("metadata.csv", index=False)


if __name__ == "__main__":
    main()
