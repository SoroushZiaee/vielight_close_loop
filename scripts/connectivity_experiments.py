from typing import List
import pandas as pd
from mne.io import read_raw_eeglab
from mne_connectivity import spectral_connectivity_epochs
import mne
import matplotlib.pyplot as plt
import numpy as np
import pickle


def check_channel_availability(metadata: pd.DataFrame, channel_name: str):
    trials_channels = metadata["channels"].values

    for tr_ch in trials_channels:
        if not channel_name in tr_ch:
            return False

    return True


def channel_availability(metadata: pd.DataFrame, total_channels: list):
    """
    Check if the base_channels are available in the metadata
    :param metadata: pd.DataFrame
    :param base_channels: list
    :return: list
    """
    available_channels = []
    for ch in total_channels:
        available_channels.append((ch, check_channel_availability(metadata, ch)))

    return available_channels


def extract_base_channels(metadata: pd.DataFrame):
    base_channels = metadata.loc[metadata["n_channels"].idxmax()]["channels"]
    base_channels = channel_availability(metadata, base_channels)
    base_channels = list(filter(lambda x: x[1], base_channels))
    base_channels = [ch[0] for ch in base_channels]

    # channel_mapping = {ch: ch.capitalize() for ch, _ in base_channels}

    return base_channels


def load_eeg_data(
    file_path, eog: tuple = (), verbose: bool = True, picks: List[str] = None
):
    raw = read_raw_eeglab(
        input_fname=file_path,
        eog=eog,
        preload=True,
        montage_units="mm",
        verbose=verbose,
    )

    print(f"{raw.get_data().shape = }")
    raw = raw.pick(picks)
    print(f"{raw.get_data().shape = }")
    print(f"{raw.ch_names = }")
    raw.rename_channels(
        mapping={
            "FPZ": "Fpz",
            "OZ": "Oz",
            "FP1": "Fp1",
            "FP2": "Fp2",
            # "FZ": "Fz",
            "FCZ": "FCz",
            # "CPZ": "CPz",
            "PZ": "Pz",
            "POZ": "POz",
        }
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    raw = raw.crop(tmin=10, tmax=raw.times[-1] - 10)

    return raw


def preprocess_raw_data(raw: mne.io.Raw, smallers_duration: float):
    if raw.times[-1] <= smallers_duration:
        return raw

    total_duration = raw.times[-1]  # in seconds
    middle_duration = int(total_duration / 2)
    start = middle_duration - int(smallers_duration / 2)
    end = middle_duration + int(smallers_duration / 2)

    # print(f"{start = }")
    # print(f"{end = }")

    raw = raw.crop(tmin=start, tmax=end)

    return raw


def save_plot(plot, file_name):
    plt.savefig(file_name, dpi=300)
    plt.close()


def calculate_connectivity(epochs, sfreq, fmin, fmax, tmin, method: str = "wpli"):

    return spectral_connectivity_epochs(
        epochs,
        method=method,
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,  # Average connectivity scores for each frequency band. If True, the output freqs will be a list with arrays of the frequencies that were averaged.
        tmin=tmin,
        mt_adaptive=False,  # Use adaptive weights for multitaper
        n_jobs=-1,
    )


def main():
    metadata_path = "/Users/soroush/Documents/Code/freelance-project/vielight/vielight_close_loop/metadata.csv"
    metadata = pd.read_csv(metadata_path)
    metadata["channels"] = metadata["channels"].apply(lambda x: eval(x))

    metadata["raw"] = None

    # Extract the base channels in all trials
    base_channels = extract_base_channels(metadata)

    raw_list = {}
    for i, file_path in enumerate(metadata["filepath"]):
        raw = load_eeg_data(file_path, picks=base_channels)
        metadata.loc[i, "raw"] = raw
        metadata.loc[i, "duration"] = metadata.loc[i, "duration"] - 20

    print(f"{metadata['raw'][0] = }")
    print(f"{metadata['duration'][0] = }")

    smallest_duration = metadata["duration"].min()
    print(f"{smallest_duration = }")

    metadata["raw"] = metadata["raw"].apply(
        lambda x: preprocess_raw_data(x, smallest_duration)
    )

    metadata["duration"] = metadata["raw"].apply(lambda x: x.times[-1])
    metadata["n_channels"] = metadata["raw"].apply(lambda x: len(x.ch_names))
    metadata["channels"] = metadata["raw"].apply(lambda x: x.ch_names)

    new_metadata = metadata[["filepath", "condition_number", "duration"]]
    # Drop column 'B'
    new_metadata = metadata.drop(columns=["raw"])
    new_metadata.to_csv("new_metadata.csv", index=False)

    epochs = {
        "pre": [],
        "during": [],
        "post": [],
    }

    for i, row in metadata.iterrows():
        raw = row["raw"]
        epochs[row["stage"]].append(
            np.expand_dims(
                raw.crop(tmin=0, tmax=int(smallest_duration) - 1).get_data(), axis=0
            )
        )

    for key, value in epochs.items():
        epochs[key] = np.concatenate(value, axis=0)

    epochs["combine"] = np.concatenate(
        [epochs["pre"], epochs["during"], epochs["post"]], axis=0
    )

    print(f"{epochs['pre'].shape = }")
    print(f"{epochs['during'].shape = }")
    print(f"{epochs['post'].shape = }")

    with open(f"processed_epochs.pkl", "wb") as f:
        pickle.dump(epochs, f)

    sfreq = metadata["raw"][0].info["sfreq"]
    fmin = 8
    fmax = 50
    tmin = 0
    method = "wpli"

    connectivity = {
        key: calculate_connectivity(
            value, sfreq=sfreq, fmin=fmin, fmax=fmax, tmin=tmin, method=method
        )
        for key, value in epochs.items()
    }

    for key, value in connectivity.items():
        print(f"{key = }")
        print(f"{value.shape = }")

    # plot_connectivity(connectivity)

    # save connectivity as pickle

    with open(f"{method}_connectivity.pkl", "wb") as f:
        pickle.dump(connectivity, f)


if __name__ == "__main__":
    main()
