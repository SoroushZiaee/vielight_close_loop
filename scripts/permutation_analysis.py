from mne_connectivity import spectral_connectivity_epochs
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Kolmogorov-Smirnov Test
# The Kolmogorov-Smirnov test compares the data to a normal distribution.
from scipy.stats import kstest
from itertools import combinations
from scipy.stats import wilcoxon

from mne_connectivity import spectral_connectivity_epochs
import os

from functools import partial
from memory_profiler import profile
import gc


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
        n_jobs=8,
    )


def extract_lower_triangular(matrix):
    return matrix[np.tril_indices(matrix.shape[0], k=-1)]


def extract_stage_lower_triangular(connectivity):
    return extract_lower_triangular(connectivity.get_data("dense").squeeze())


def perform_paired_ttest(sample1, sample2, alpha=0.05):
    t_statistic, p_value = stats.ttest_rel(sample1, sample2)

    return t_statistic, p_value


def single_permutation(combined_data, *args, **kwargs):
    indices = np.arange(len(combined_data))
    np.random.shuffle(indices)
    shuffled_data = combined_data[indices]

    shuffled_pre = shuffled_data[:7]
    shuffled_during = shuffled_data[7:12]
    shuffled_post = shuffled_data[12:]

    perm_wPLI_pre = extract_stage_lower_triangular(
        calculate_connectivity(shuffled_pre, *args, **kwargs)
    )
    perm_wPLI_during = extract_stage_lower_triangular(
        calculate_connectivity(shuffled_during, *args, **kwargs)
    )
    perm_wPLI_post = extract_stage_lower_triangular(
        calculate_connectivity(shuffled_post, *args, **kwargs)
    )

    return {
        "pre_post": perform_paired_ttest(
            perm_wPLI_post.flatten(), perm_wPLI_pre.flatten()
        ),
        "pre_during": perform_paired_ttest(
            perm_wPLI_during.flatten(), perm_wPLI_pre.flatten()
        ),
        "during_post": perform_paired_ttest(
            perm_wPLI_post.flatten(), perm_wPLI_during.flatten()
        ),
    }


def permutation_test(epochs, n_permutations=1000, *args, **kwargs):
    combined_data = epochs["combine"]

    # Create a partial function with fixed arguments
    perm_func = partial(single_permutation, combined_data, *args, **kwargs)

    # This will be replaced with parallel execution
    results = [perm_func() for _ in range(n_permutations)]

    # Aggregate results
    aggregated_results = {
        "pre_post": np.array([r["pre_post"] for r in results]),
        "pre_during": np.array([r["pre_during"] for r in results]),
        "during_post": np.array([r["during_post"] for r in results]),
    }

    return aggregated_results


def run_permutation_batch(
    start_index, end_index, epochs, sfreq, fmin, fmax, tmin, method
):
    results = []
    for _ in range(start_index, end_index):
        result = single_permutation(epochs["combine"], sfreq, fmin, fmax, tmin, method)
        results.append(result)

        gc.collect()

    return results


@profile
def main():
    process_epoch_path = "/home/soroush1/projects/def-kohitij/soroush1/vielight_close_loop/data/processed_epochs.pkl"
    with open(process_epoch_path, "rb") as f:
        epochs = pickle.load(f)

    for key, value in epochs.items():
        print(key, value.shape)

    # Get parameters from environment variables or use defaults
    sfreq = int(os.environ.get("SFREQ", 500))
    fmin = int(os.environ.get("FMIN", 8))
    fmax = int(os.environ.get("FMAX", 50))
    tmin = float(os.environ.get("TMIN", 0))
    method = os.environ.get("METHOD", "wpli")

    # Get the task ID and total number of tasks from SLURM environment variables
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    # Calculate the range of permutations for this task
    n_permutations = int(os.environ.get("N_PERMUTATIONS", 1))

    permutations_per_task = n_permutations // n_tasks
    start_index = task_id * permutations_per_task
    end_index = start_index + permutations_per_task

    print(f"{task_id = }")
    print(f"{n_tasks = }")
    print(f"{n_permutations = }")
    print(f"{permutations_per_task = }")
    print(f"{start_index = }")
    print(f"{end_index = }")

    # Run the permutations for this batch
    results = run_permutation_batch(
        start_index, end_index, epochs, sfreq, fmin, fmax, tmin, method
    )

    # Save the results
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        "/home/soroush1/projects/def-kohitij/soroush1/vielight_close_loop/resutls/permutation_conditions",
    )

    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/permutation_results_{task_id}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
