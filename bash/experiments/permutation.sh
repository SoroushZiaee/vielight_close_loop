#!/bin/bash
#SBATCH --job-name=permutation_analysis
#SBATCH --output=permutation_analysis_%A_%a.out
#SBATCH --error=permutation_analysis_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --array=0-50
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
    ## SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
    ## SBATCH --mail-user=soroush1@yorku.ca

module load python/3.11.5 openmpi/4.1.5 mpi4py/3.1.6
module list

pip freeze

export EPOCH_PATH="/home/soroush1/projects/def-kohitij/soroush1/vielight_close_loop/data/processed_epochs.pkl"
export OUTPUT_DIR="/home/soroush1/projects/def-kohitij/soroush1/vielight_close_loop/resutls/permutation_conditions"
export N_PERMUTATIONS=1000
export SFREQ=500
export FMIN=8
export FMAX=50
export TMIN=0
export METHOD="wpli"

source /home/soroush1/projects/def-kohitij/soroush1/vielight_close_loop/venv/bin/activate

pip freeze

python /home/soroush1/projects/def-kohitij/soroush1/vielight_close_loop/scripts/permutation_analysis.py