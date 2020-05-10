import argparse
import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


parser = argparse.ArgumentParser(description="Distributed training benchmark")
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

tf.random.set_seed(args.seed)
_physical_devices = (tf.config
                       .experimental
                       .list_physical_devices("GPU"))
NUMBER_GPUS = len(_physical_devices)

# define the data and logging directories
data_dir = pathlib.Path(args.data_dir)
training_data_dir = data_dir / "train"
validation_data_dir = data_dir / "val"
testing_data_dir = data_dir / "test"

# only log from first worker to avoid logging data corruption
verbose = 2

checkpoints_logging_dir = pathlib.Path(args.write_checkpoints_to)
if not os.path.isdir(checkpoints_logging_dir):
    os.mkdir(checkpoints_logging_dir)

tensorboard_logging_dir = pathlib.Path(args.tensorboard_logging_dir)
if not os.path.isdir(tensorboard_logging_dir):
    os.mkdir(tensorboard_logging_dir)

# define constants used in data preprocessing
img_width, img_height = 224, 224
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
def _decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = (tf.image
             .decode_jpeg(img, channels=3))
    # convert to floats in the [0,1] range.
    img = (tf.image
             .convert_image_dtype(img, tf.float32))
    # resize the image to the desired size.
    img = (tf.image
             .resize(img, [img_width, img_height]))
    return img

@tf.function
def preprocess(image):
    label = _get_label(image)
    # load the raw data from the file as a string
    img = tf.io.read_file(image)
    img = _decode_img(img)
    return img, label

# allow Tensorflow to choose the amount of parallelism used in preprocessing based on number of available CPUs
AUTOTUNE = (tf.data
              .experimental
              .AUTOTUNE)

# allow Tensorflow to tune the optimal prefetch buffer size based on runtime conditions
_prefetch_buffer_size = AUTOTUNE if args.prefetch_buffer_size is None else args.prefetch_buffer_size

# make sure that each GPU uses a different seed so that each GPU trains on different random sample of training data
training_dataset = (tf.data
                      .Dataset
                      .list_files(f"{training_data_dir}/*/*", shuffle=True, seed=None
                      .map(preprocess, num_parallel_calls=AUTOTUNE)
                      .shuffle(args.shuffle_buffer_size, reshuffle_each_iteration=True, seed=None)
                      .repeat()
                      .batch(args.batch_size)
                      .prefetch(_prefetch_buffer_size))

validation_dataset = (tf.data
                        .Dataset
                        .list_files(f"{validation_data_dir}/*/*", shuffle=False)
                        .map(preprocess, num_parallel_calls=AUTOTUNE)
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


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    _loss_fn = (keras.losses
                     .CategoricalCrossentropy())
    
    # adjust initial learning rate based on number of "effective GPUs".
    _global_batch_size = args.batch_size * NUMBER_GPUS
    _n_effective_gpus = _global_batch_size // args.base_batch_size 
    _initial_lr = args.base_lr * _n_effective_gpus 
    _optimizer = (keras.optimizers
                       .SGD(lr=_initial_lr, momentum=args.momentum))

    _metrics = [
        keras.metrics.CategoricalAccuracy(),
        keras.metrics.TopKCategoricalAccuracy(k=5)
    ]

    model_fn = (keras.applications
                     .ResNet50(weights=None))

    if checkpoint_filepath is not None:
        model_fn.load_weights(checkpoint_filepath)

    model_fn.compile(loss=_loss_fn,
                     optimizer=_distributed_optimizer,
                     metrics=_metrics,
                     )

callbacks = [
    
    # Average metrics among workers at the end of every epoch.
    #
    # This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    keras.callbacks.MetricAverageCallback(),
    
    # Using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    keras.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),

    # After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    keras.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=30, multiplier=1.),
    keras.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
    keras.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
    keras.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),
]

# Logging callbacks only on the rank 0 worker to prevent other workers from corrupting them.
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
             steps_per_epoch=n_training_images // (args.batch_size * NUMBER_GPUS),
             validation_data=validation_dataset,
             validation_steps=n_validation_images // (args.val_batch_size * NUMBER_GPUS),
             verbose=verbose,
             callbacks=callbacks)
