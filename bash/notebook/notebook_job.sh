#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter.out
#SBATCH --error=jupyter.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Start Installing and setup env"

module load python/3.11.5 openmpi/4.1.5 mpi4py/3.1.6
module list

pip freeze

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

echo "Installing requirements"
pip install --no-index numpy pandas matplotlib seaborn scikit-learn scipy mne jupyterlab jupyter

echo "Env has been set up"

pip freeze

/home/soroush1/projects/def-kohitij/soroush1/vielight_close_loop/bash/notebook/lab.sh
