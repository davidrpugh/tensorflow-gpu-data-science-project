#!/bin/bash --login

set -e

PROJECT_ROOT=.

# build the conda environment
CONDA_PREFIX=$PROJECT_ROOT/env
conda env create --prefix $CONDA_PREFIX --file $PROJECT_ROOT/environment.yml --force

# activate the environment and then install pip dependencies
conda activate $CONDA_PREFIX
HOROVOD_CUDA_HOME=$CONDA_PREFIX
HOROVOD_NCCL_HOME=$CONDA_PREFIX
HOROVOD_GPU_ALLREDUCE=NCCL
pip install --no-cache-dir -r $PROJECT_ROOT/requirements.txt

# confirm that Horovod has been built with support for TensorFlow, MPI, Gloo, and NCCL
horovodrun --check-build

# run the postBuild script to build any JupyterLab extensions
. $PROJECT_ROOT/postBuild
conda activate base
