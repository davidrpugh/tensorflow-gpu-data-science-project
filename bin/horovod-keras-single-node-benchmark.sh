#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100:8
#SBATCH --constraint=cpu_intel_platinum_8260
#SBATCH --partition=batch
#SBATCH --output=../results/%x/slurm-%j.out
#SBATCH --error=../results/%x/slurm-%j.err

# Some logs will be on persistent storage
PERSISTENT_LOGGING_DIR=../results/$SLURM_JOB_NAME/$SLURM_JOB_ID/logs
mkdir -p $PERSISTENT_LOGGING_DIR

# Some logs will be on local storage
LOCAL_LOGGING_DIR=/tmp/$SLURM_JOB_NAME/$SLURM_JOB_ID/logs
mkdir -p $LOCAL_LOGGING_DIR

# Load software stack
module load cuda/10.0.130
conda activate ../env

# start the nvidia-smi process in the background
nvidia-smi dmon --delay 60 --options DT >> $PERSISTENT_LOGGING_DIR/nvidia-smi.log &
NVIDIA_SMI_PID=$!

# start the training process in the background
SRC_DIR=../src/horovod-keras-example
DATA_DIR=/local/reference/CV/ILSVR/classification-localization/data/jpeg
horovodrun -np $SLURM_NTASKS python $SRC_DIR/train.py \
    --data-dir $DATA_DIR \
    --checkpoints-logging-dir $LOCAL_LOGGING_DIR/checkpoints \
    --tensorboard-logging-dir $LOCAL_LOGGING_DIR/tensorboard &
HOROVODRUN_PID=$!

# asynchronous rsync of training logs between local and persistent storage
RSYNC_DELAY_SECONDS=600
HOROVODRUN_STATE=$(ps -h --pid $HOROVODRUN_PID -o state | head -n 1)
while [ "${HOROVODRUN_STATE}" != "" ]
    do
        HOROVODRUN_STATE=$(ps -h --pid $HOROVODRUN_PID -o state | head -n 1)
        rsync -a $LOCAL_LOGGING_DIR/ $PERSISTENT_LOGGING_DIR
        sleep $RSYNC_DELAY_SECONDS
done

# kill off the nvidia-smi process
kill $NVIDIA_SMI_PID

# make sure to get any new files written since last rsync 
rsync -a $LOCAL_LOGGING_DIR/ $PERSISTENT_LOGGING_DIR

