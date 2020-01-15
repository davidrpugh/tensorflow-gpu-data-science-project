## Slurm

### Understanding the job script

The job script can be broken down into a number of sections.

#### Slurm directives

```bash
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
```

#### Checkpointing

```bash
...
# Some logs will be on persistent storage
PERSISTENT_LOGGING_DIR=../results/$SLURM_JOB_NAME/$SLURM_JOB_ID/logs
mkdir -p $PERSISTENT_LOGGING_DIR

# Some logs will be on local storage
LOCAL_LOGGING_DIR=/tmp/$SLURM_JOB_NAME/$SLURM_JOB_ID/logs
mkdir -p $LOCAL_LOGGING_DIR
...
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
...
# make sure to get any new files written since last rsync
rsync -a $LOCAL_LOGGING_DIR/ $PERSISTENT_LOGGING_DIR
```

#### Loading the software application stack

```bash
...
# Load software stack
module load cuda/10.0.130
conda activate ../env
...
```

#### GPU Resource Monitoring

```bash
...
# start the nvidia-smi process in the background
NVIDIA_SMI_DELAY_SECONDS=60
nvidia-smi dmon --delay $NVIDIA_SMI_DELAY_SECONDS --options DT >> $PERSISTENT_LOGGING_DIR/nvidia-smi.log &
NVIDIA_SMI_PID=$!
...
# kill off the nvidia-smi process
kill $NVIDIA_SMI_PID
...
```

#### Running the training job

```bash
...
# start the training process in the background
horovodrun -np $SLURM_NTASKS python $TRAINING_SCRIPT \
    --data-dir $DATA_DIR \
    --checkpoints-logging-dir $LOCAL_LOGGING_DIR/checkpoints \
    --tensorboard-logging-dir $LOCAL_LOGGING_DIR/tensorboard &
...
```


### Submitting jobs

```bash
$ JOB_NAME=horovod-keras-single-node-benchmark
$ mkdir ../results/$JOB_NAME
$ TRAINING_SCRIPT=../src/horovod-keras-example/train.py
$ DATA_DIR=/local/reference/CV/ILSVR/classification-localization/data/jpeg
$ sbatch --job-name $JOB_NAME --mail-user USER_EMAIL --mail-type=ALL --export SRC_DIR=$SRC_DIR,DATA_DIR=$DATA_DIR horovod-keras-single-node-benchmark.sh
```
