from typing import List
import pandas as pd
from mne.io import read_raw_eeglab
from mne_connectivity import spectral_connectivity_epochs
import mne
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mne_features.univariate import compute_hjorth_complexity, compute_hjorth_mobility


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


def generate_timepoints(total_recording_time, epoch_length, overlap):
    overlap_length = epoch_length * overlap

    start = 0
    timepoints = []
    while start + epoch_length < total_recording_time:
        timepoints.append((start, start + epoch_length))
        start += epoch_length - overlap_length

    return timepoints


def create_epochs(
    raw: mne.io.Raw, epoch_length=1, overlap=0.2, pick: List[str] = None
) -> mne.Epochs:

    total_recording_time = raw.times[-1]  # in seconds
    overlap_length = epoch_length * overlap

    timepoints = generate_timepoints(total_recording_time, epoch_length, overlap)
    # print(f"{timepoints = }")

    epochs = []
    for start, end in timepoints:
        if pick is not None:
            epochs.append(raw.copy().pick(pick).crop(tmin=start, tmax=end))
        else:
            epochs.append(
                np.expand_dims(
                    raw.copy().crop(tmin=start, tmax=end).get_data(),
                    axis=0,
                )
            )

    return np.concatenate(epochs, axis=0)


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


def calculate_hjorth_complexity(epochs: np.ndarray):

    complexities = []
    for epoch in tqdm(epochs, desc="Calculating Hjorth complexity for each epoch"):
        hjorth_complexity = compute_hjorth_complexity(epoch)
        complexities.append(np.expand_dims(hjorth_complexity, axis=0))

    return np.concatenate(complexities, axis=0)


def calculate_hjorth_mobility(epochs: np.ndarray):

    complexities = []
    for epoch in tqdm(epochs, desc="Calculating Hjorth mobility for each epoch"):
        hjorth_mobility = compute_hjorth_mobility(epoch)
        complexities.append(np.expand_dims(hjorth_mobility, axis=0))

    return np.concatenate(complexities, axis=0)


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
        "pre": None,
        "during": None,
        "post": None,
    }

    epoch_length = 9
    overlap = 0.0

    for i, row in tqdm(metadata.iterrows(), desc="Creating epochs"):
        raw = row["raw"]
        epochs[row["stage"]] = create_epochs(
            raw.crop(tmin=0, tmax=int(smallest_duration) - 1),
            epoch_length=epoch_length,
            overlap=overlap,
        )

    print(f"{epochs['pre'].shape = }")
    print(f"{epochs['during'].shape = }")
    print(f"{epochs['post'].shape = }")

    complexities = {}
    for key, value in epochs.items():
        complexities[key] = calculate_hjorth_complexity(value)

    mobilities = {}
    for key, value in epochs.items():
        mobilities[key] = calculate_hjorth_mobility(value)

    # save the features
    import pickle

    with open("hjorth_complexity.pkl", "wb") as f:
        pickle.dump(complexities, f)

    with open("hjorth_mobility.pkl", "wb") as f:
        pickle.dump(mobilities, f)

    # we need to calculate the hjorth parameters for each epoch
    # use apply function to calculate hjorth parameters for each epoch


if __name__ == "__main__":
    main()
