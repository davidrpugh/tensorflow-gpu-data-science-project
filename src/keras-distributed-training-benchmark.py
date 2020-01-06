import argparse
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import horovod.tensorflow.keras as hvd


parser = argparse.ArgumentParser(description="TensorFlow + Horovod distributed training benchmark")
parser.add_argument("--data-dir", type=str, help="path to ILSVR data")
parser.add_argument("--results-dir", type=str, help="Path to the results directory")

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument("--batch-size", type=int, default=32, help="input batch size for training")
parser.add_argument("--val-batch-size", type=int, default=32, help="input batch size for validation")
parser.add_argument("--warmup-epochs", type=float, default=5, help="number of warmup epochs")
parser.add_argument("--epochs", type=int, default=90, help="number of epochs to train")
parser.add_argument("--base-lr", type=float, default=1.25e-2, help="learning rate for a single GPU")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--weight-decay", type=float, default=5e-5, help="weight decay")
parser.add_argument("--seed", type=int, default=42, help="random seed")
args = parser.parse_args()

# initialize Horovod.
hvd.init()

# pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

# define the data directories
data_dir = pathlib.Path(args.data_dir)
TRAINING_DATA_DIR = data_dir / "train"
VALIDATION_DATA_DIR = data_dir / "val"
TESTING_DATA_DIR = data_dir / "test"

# define the logging directories
VERBOSE = 2 if hvd.rank() == 0 else 0
RESULTS_DIR = pathlib.Path(args.results_dir)
LOGGING_DIR = RESULTS_DIR / "logs"

# create the training and validation datasets
IMG_WIDTH, IMG_HEIGHT = 224, 224
N_TRAINING_IMAGES = 1281167
N_VALIDATION_IMAGES = 50000
N_TESTING_IMAGES = 100000
CLASS_NAMES = tf.constant([item.name for item in TRAINING_DATA_DIR.glob('*')])

@tf.function
def _get_label(file_path) -> tf.Tensor:
    # convert the path to a list of path components
    split_file_path = (tf.strings
                         .split(file_path, '/'))
    # The second to last is the class-directory
    label = tf.equal(split_file_path[-2], CLASS_NAMES)
    return label


@tf.function
def _decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = (tf.image
             .decode_jpeg(img, channels=3))
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = (tf.image
             .convert_image_dtype(img, tf.float32))
    # resize the image to the desired size.
    img = (tf.image
             .resize(img, [IMG_WIDTH, IMG_HEIGHT]))
    return img


@tf.function
def preprocess(image):
    label = _get_label(image)
    # load the raw data from the file as a string
    img = tf.io.read_file(image)
    img = _decode_img(img)
    return img, label


AUTOTUNE = (tf.data
              .experimental
              .AUTOTUNE)

training_dataset = (tf.data
                      .Dataset
                      .list_files(f"{TRAINING_DATA_DIR}/*/*", shuffle=True, seed=args.seed)
                      .map(preprocess, num_parallel_calls=AUTOTUNE)
                      .cache()
                      .shuffle(N_TRAINING_IMAGES, reshuffle_each_iteration=True, seed=args.seed)
                      .repeat()
                      .batch(args.batch_size)
                      .prefetch(buffer_size=1))

validation_dataset = (tf.data
                       .Dataset
                       .list_files(f"{VALIDATION_DATA_DIR}/*/*", shuffle=False)
                       .map(preprocess, num_parallel_calls=AUTOTUNE)
                       .cache()
                       .batch(args.val_batch_size))

# Set up the model
model_fn = (keras.applications
                 .ResNet50(weights=None, include_top=True))
_loss_fn = (keras.losses
                 .CategoricalCrossentropy())
_initial_lr = args.base_lr * hvd.size() # adjust initial learning rate based on number of GPUs.
_optimizer = (keras.optimizers
                   .SGD(lr=_initial_lr, momentum=args.momentum))
_metrics = [
    "accuracy",
    "top_k_categorical_accuracy"
]

model_fn.compile(loss=_loss_fn,
                 optimizer=_optimizer,
                 metrics=_metrics)      

# define the callbacks
_callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
    
    # using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=VERBOSE),

    # after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=30, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),
]

# save checkpoints only on the first worker to prevent other workers from corrupting them.
_checkpoints_logging = (keras.callbacks
                             .ModelCheckpoint(f"{LOGGING_DIR}/checkpoints",
                                              save_best_only=False,
                                              save_freq="epoch"))
_tensorboard_logging = (keras.callbacks
                             .TensorBoard(f"{LOGGING_DIR}/tensorboard"))

if hvd.rank() == 0:
    _callbacks.extend([_checkpoints_logging, _tensorboard_logging])
    

# model training loop
model_fn.fit(training_dataset,
             epochs=args.epochs,
             steps_per_epoch=N_TRAINING_IMAGES // args.batch_size,
             validation_data=validation_dataset,
             verbose=VERBOSE,
             callbacks=_callbacks)
