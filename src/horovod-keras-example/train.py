import argparse
import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import horovod.tensorflow.keras as hvd


parser = argparse.ArgumentParser(description="Horovod + Keras distributed training benchmark")
parser.add_argument("--data-dir",
                    type=str,
                    help="Path to ILSVR data")
parser.add_argument("--shuffle-buffer-size",
                    type=int,
                    default=12811,
                    help="Size of the shuffle buffer (default buffer size is 1% of all training images)")
parser.add_argument("--prefetch-buffer-size",
                    type=int,
                    help="Size of the prefetch buffer (if not provided, Tensorflow will tune the prefetch buffer size based on runtime conditions")
parser.add_argument("--read-checkpoints-from",
                    type=str,
                    help="Path to a directory containing existing checkpoints")
parser.add_argument("--write-checkpoints-to",
                    type=str,
                    help="Path to the directory where checkpoints should be written")
parser.add_argument("--tensorboard-logging-dir",
                    type=str,
                    help="Path to the tensorboard logging directory")

# Most default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument("--batch-size",
                    type=int,
                    default=256,
                    help="input batch size for training")
parser.add_argument("--base-batch-size",
                    type=int,
                    default=32,
                    help="batch size used to determine number of effective GPUs")
parser.add_argument("--val-batch-size",
                    type=int,
                    default=32,
                    help="input batch size for validation")
parser.add_argument("--warmup-epochs",
                    type=float,
                    default=5,
                    help="number of warmup epochs")
parser.add_argument("--epochs",
                    type=int,
                    default=90,
                    help="number of epochs to train")
parser.add_argument("--base-lr",
                    type=float,
                    default=1.25e-2,
                    help="learning rate for a single GPU")
parser.add_argument("--momentum",
                    type=float,
                    default=0.9,
                    help="SGD momentum")
parser.add_argument("--weight-decay",
                    type=float,
                    default=5e-5,
                    help="weight decay")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="random seed")
args = parser.parse_args()

hvd.init()
tf.random.set_seed(args.seed)

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

# define the data and logging directories
data_dir = pathlib.Path(args.data_dir)
training_data_dir = data_dir / "train"
validation_data_dir = data_dir / "val"
testing_data_dir = data_dir / "test"

# only log from first worker to avoid logging data corruption
verbose = 2 if hvd.rank() == 0 else 0

checkpoints_logging_dir = pathlib.Path(args.write_checkpoints_to)
if not os.path.isdir(checkpoints_logging_dir) and hvd.rank() == 0:
    os.mkdir(checkpoints_logging_dir)

tensorboard_logging_dir = pathlib.Path(args.tensorboard_logging_dir)
if not os.path.isdir(tensorboard_logging_dir) and hvd.rank() == 0:
    os.mkdir(tensorboard_logging_dir)

# define constants used in data preprocessing
resized_img_width, resized_img_height = 256, 256
target_img_width, target_img_height = 224, 224
n_training_images = 1281167
n_validation_images = 50000
n_testing_images = 100000
class_names = tf.constant([item.name for item in training_data_dir.glob('*')])

@tf.function
def _get_label(file_path) -> tf.Tensor:
    # convert the path to a list of path components
    split_file_path = (tf.strings
                         .split(file_path, '/'))
    # The second to last is the class-directory
    label = tf.equal(split_file_path[-2], class_names)
    return label

@tf.function
def preprocess_image(image):
    label = _get_label(image)
    # read the file and decode the image
    str_tensor = (tf.io
                    .read_file(image))
    int_tensor = (tf.image
                    .decode_jpeg(str_tensor, channels=3))
    # standardize the image    
    resized_image = (tf.image
                       .resize(int_tensor, size=[resized_img_height, resized_img_width]))
    standardized_image = (tf.image
                            .per_image_standardization(resized_image))
    return standardized_image, label

@tf.function
def transform_image(preprocessed_image, label):
    _augmented_image = (tf.image
                          .random_crop(preprocessed_image, size=[target_img_height, target_img_width, 3]))
    _augmented_image = (tf.image
                          .random_flip_left_right(_augmented_image))
    _augmented_image = (tf.image
                          .random_contrast(_augmented_image, lower=0.8, upper=1.2))
    return _augmented_image, label

# allow Tensorflow to choose the amount of parallelism used in data pipelines
AUTOTUNE = (tf.data
              .experimental
              .AUTOTUNE)

# allow Tensorflow to tune the optimal prefetch buffer size based on runtime conditions
_prefetch_buffer_size = AUTOTUNE if args.prefetch_buffer_size is None else args.prefetch_buffer_size

# make sure that each GPU uses a different seed so that each GPU trains on different random sample of training data
training_dataset = (tf.data
                      .Dataset
                      .list_files(f"{training_data_dir}/*/*", shuffle=True, seed=hvd.rank())
                      .map(preprocess_image, num_parallel_calls=AUTOTUNE) # good place to cache in memory!
                      .shuffle(args.shuffle_buffer_size, reshuffle_each_iteration=True, seed=hvd.rank())
                      .map(transform_image, num_parallel_calls=AUTOTUNE)
                      .repeat()
                      .batch(args.batch_size)
                      .prefetch(_prefetch_buffer_size))

validation_dataset = (tf.data
                        .Dataset
                        .list_files(f"{validation_data_dir}/*/*", shuffle=False)
                        .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                        .batch(args.val_batch_size))
    
# Look for a pre-existing checkpoint from which to resume training
existing_checkpoints_dir = pathlib.Path(args.read_checkpoints_from)
checkpoint_filepath = None
initial_epoch = 0
for _most_recent_epoch in range(args.epochs, 0, -1):
    _checkpoint_filepath = f"{existing_checkpoints_dir}/checkpoint-epoch-{_most_recent_epoch:02d}.h5"
    if os.path.exists(_checkpoint_filepath):
        checkpoint_filepath = _checkpoint_filepath
        initial_epoch = _most_recent_epoch
        break
        
# make sure that all workers agree to resume training from the same epoch
intial_epoch = hvd.broadcast(initial_epoch, root_rank=0, name='initial_epoch')

_loss_fn = (keras.losses
                 .CategoricalCrossentropy())
    
# adjust initial learning rate based on number of "effective GPUs".
_global_batch_size = args.batch_size * hvd.size()
_n_effective_gpus = _global_batch_size // args.base_batch_size 
_initial_lr = args.base_lr * _n_effective_gpus 
_optimizer = (keras.optimizers
                   .SGD(lr=_initial_lr, momentum=args.momentum))
_distributed_optimizer = hvd.DistributedOptimizer(_optimizer)

_metrics = [
    keras.metrics.CategoricalAccuracy(),
    keras.metrics.TopKCategoricalAccuracy(k=5)
]

model_fn = (keras.applications
                 .ResNet50(weights=None))

# restore checkpoint on rank 0 worker (weights will be broadcast to all other workers)
if checkpoint_filepath is not None and hvd.rank() == 0:
    model_fn.load_weights(checkpoint_filepath)

model_fn.compile(loss=_loss_fn,
                 optimizer=_distributed_optimizer,
                 metrics=_metrics,
                 experimental_run_tf_function=False, # required for Horovod to work with TF 2.0
                 )

callbacks = [
    # Broadcast initial variable states from rank 0 worker to all other processes.
    #
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),

    # Average metrics among workers at the end of every epoch.
    #
    # This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
    
    # Using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),

    # After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=30, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),
]

# Logging callbacks only on the rank 0 worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    _checkpoints_logging = (keras.callbacks
                                 .ModelCheckpoint(f"{checkpoints_logging_dir}/checkpoint-epoch-{{epoch:02d}}.h5",
                                                  save_best_only=False,
                                                  save_freq="epoch"))
    _tensorboard_logging = (keras.callbacks
                                 .TensorBoard(tensorboard_logging_dir))
    callbacks.extend([_checkpoints_logging, _tensorboard_logging])

model_fn.fit(training_dataset,
             epochs=args.epochs,
             initial_epoch=initial_epoch,
             steps_per_epoch=n_training_images // (args.batch_size * hvd.size()),
             validation_data=validation_dataset,
             validation_steps=n_validation_images // (args.val_batch_size * hvd.size()),
             verbose=verbose,
             callbacks=callbacks)
