## Slurm

### Understanding the job script

The job script can be broken down into a number of sections.

#### Slurm directives

```bash
#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=0
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

Efficient checkpointing can be a bit tricky. The main concern is that the worker responsible for 
writing the checkpoint files will be blocked from training while the checkpoint files are being 
written. Thus we should avoid writing checkpoints directly to `/ibex/(f)scratch` which is a 
shared file system whose performance can vary depending on overall IO load. Instead should write 
checkpoints files to local, on-node storage. However local, on-node storage is not persistent and  
will be wiped after the job terminates. So if we are going to write checkpoint files to local 
storage we need to periodically sync our checkpoint files with persistent storage (in a manner 
which will not block our training progress).
 
```bash
...
# Need to define persistent storage for logging... 
PERSISTENT_LOGGING_DIR=../results/$SLURM_JOB_NAME/logs
PERSISTENT_CHECKPOINTS_DIR=$PERSISTENT_LOGGING_DIR/checkpoints
PERSISTENT_TENSORBOARD_DIR=$PERSISTENT_LOGGING_DIR/tensorboard

# N.B. mkdir does not overwrite if these directories already exist
mkdir -p $PERSISTENT_CHECKPOINTS_DIR
mkdir -p $PERSISTENT_TENSORBOARD_DIR

# ...but for best performance write checkpoints and tensorboard logs to local storage
LOCAL_LOGGING_DIR=/tmp/$SLURM_JOB_NAME/$SLURM_JOB_ID/logs
LOCAL_CHECKPOINTS_DIR=$LOCAL_LOGGING_DIR/checkpoints
LOCAL_TENSORBOARD_DIR=$LOCAL_LOGGING_DIR/tensorboard
mkdir -p $LOCAL_CHECKPOINTS_DIR
mkdir -p $LOCAL_TENSORBOARD_DIR
...
HOROVODRUN_PID=$!

# asynchronous rsync of training logs between local and persistent storage
RSYNC_DELAY_SECONDS=600
HOROVODRUN_STATE=$(ps -h --pid $HOROVODRUN_PID -o state | head -n 1)
while [ "${HOROVODRUN_STATE}" != "" ]
    do
        rsync -a $LOCAL_CHECKPOINTS_DIR/ $PERSISTENT_CHECKPOINTS_DIR
        rsync -a $LOCAL_TENSORBOARD_DIR/ $PERSISTENT_TENSORBOARD_DIR
        sleep $RSYNC_DELAY_SECONDS
        HOROVODRUN_STATE=$(ps -h --pid $HOROVODRUN_PID -o state | head -n 1)
done
...
# make sure to get any new files written since last rsync 
rsync -a $LOCAL_CHECKPOINTS_DIR/ $PERSISTENT_CHECKPOINTS_DIR
rsync -a $LOCAL_TENSORBOARD_DIR/ $PERSISTENT_TENSORBOARD_DIR
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
    --read-checkpoints-from $PERSISTENT_CHECKPOINTS_DIR \
    --write-checkpoints-to  $LOCAL_CHECKPOINTS_DIR \
    --tensorboard-logging-dir $LOCAL_TENSORBOARD_DIR &
...
```

### Submitting jobs

```bash
$ JOB_NAME=horovod-keras-single-node-benchmark
$ mkdir ../results/$JOB_NAME
$ TRAINING_SCRIPT=../src/horovod-keras-example/train.py
$ DATA_DIR=/local/reference/CV/ILSVR/classification-localization/data/jpeg
$ sbatch --job-name $JOB_NAME --mail-user USER_EMAIL --mail-type=ALL --export SRC_DIR=$SRC_DIR,DATA_DIR=$DATA_DIR horovod-single-node-job.sh
```
