import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

import mne
from mne.io import read_raw_eeglab, read_raw_fif
from mne.viz import plot_raw, plot_raw_psd


def argparser():

    parser = argparse.ArgumentParser(description="Interactive plot for single data.")
    parser.add_argument(
        "--input-fname",
        type=str,
        required=True,
        help="Path to the input file.",
    )
    parser.add_argument(
        "--eog",
        type=tuple,
        default=(),
        help="Tuple of strings that are channel names to be considered EOG channels.",
    )

    # type plot - psd or time
    parser.add_argument(
        "--plot-type",
        type=str,
        default="time",
        help="Type of plot to be displayed. Options: 'time' or 'psd'.",
    )

    return parser


def select_plot_type(plot_type: str, raw):

    if plot_type == "time":
        events = None

        duration = 1 * 60 + 42  # 1 minute and 42 seconds
        n_channels = 20
        scalings = None

        plot = plot_raw(
            raw=raw,
            events=events,
            duration=duration,
            n_channels=n_channels,
            scalings=scalings,
            show=False,
        )
    elif plot_type == "psd":
        fmin = 0
        fmax = np.inf
        tmin = None
        tmax = None
        proj = False  # whether to apply SSP projection vectors
        n_fft = None  # number of points to use in Welch FFT calculations
        n_overlap = 0  # number of points of overlap between blocks
        xscale = "linear"  # scaling of the x-axis (‘linear’ or ‘log’)
        area_mode = "std"  # mode for plotting area (‘std’ or ‘range’)
        dB = True
        estimate = "power"
        n_jobs = -1
        average = True
        sphere = "auto"

        plot = plot_raw_psd(
            raw=raw,
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            proj=proj,
            n_fft=n_fft,
            n_overlap=n_overlap,
            xscale=xscale,
            area_mode=area_mode,
            dB=dB,
            estimate=estimate,
            n_jobs=n_jobs,
            average=average,
            sphere=sphere,
            show=True,
        )
    else:
        raise ValueError("Invalid plot type. Please use 'time' or 'psd'.")

    return plot


def get_raw(input_fname: str, *args, **kwargs):
    base_filename = os.path.basename(input_fname)
    filename, extenstion = os.path.splitext(base_filename)

    if extenstion == ".fif":
        raw = mne.io.Raw(fname=input_fname)
    elif extenstion == ".set":
        raw = read_raw_eeglab(
            input_fname=input_fname, montage_units="mm", *args, **kwargs
        )
    else:
        raise ValueError("Invalid file format. Please use .fif or .set.")

    return raw


def main(input_fname: str, eog: tuple = (), plot_type: str = "time"):

    raw = get_raw(input_fname=input_fname, eog=eog, preload=True, verbose=True)

    plot = select_plot_type(plot_type=plot_type, raw=raw)

    plt.show()


if __name__ == "__main__":
    argparser = argparser()
    args = argparser.parse_args()

    input_fname = args.input_fname
    eog = args.eog
    plot_type = args.plot_type

    main(input_fname=input_fname, eog=eog, plot_type=plot_type)


# Run the script
# python scripts/interactive_plot_single_data.py --input-fname /Users/soroush/Documents/Code/freelance-project/vielight/vielight_close_loop/output_data/io_experiments/pre_whole_raws.fif --plot-type psd
