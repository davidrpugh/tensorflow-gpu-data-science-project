#!/bin/bash --login

# define environment variables for building Horovod
export CUDA_HOME=/usr/local/cuda-10.0 # /usr/local/cuda-10.1 for TF 2.1
export ENV_PREFIX=$PWD/env
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL

# create the conda environment
conda env create --prefix $ENV_PREFIX --file environment.yml --force
conda activate $ENV_PREFIX
. postBuild
