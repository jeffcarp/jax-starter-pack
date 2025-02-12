"""Fiddle config for training."""

import dataclasses
import datetime
import os

import fiddle as fdl
from flax import nnx
import optax

import data
import grain.python as grain
import model as starter_model


@dataclasses.dataclass
class TrainConfig:
  model: nnx.Module
  optimizer: nnx.Optimizer
  train_batch_size: int
  train_loader: grain.RandomAccessDataSource
  eval_loader: grain.RandomAccessDataSource
  log_dir: str
  checkpoint_dir: str
  train_total_steps: int = 2000
  eval_interval_steps: int = 500
  summary_interval_steps: int = 200
  checkpoint_interval_steps: int = 250
  run_profile: bool = True
  profile_start_step: int = 50
  profile_end_step: int = 60

  def __post_init__(self):
    assert self.profile_end_step > self.profile_start_step
    assert self.profile_start_step >= 0


def default_config(
  train_batch_size: int = 128,
  eval_batch_size: int = 128,
  base_log_dir: str = "/tmp/tensorboard",
) -> fdl.Config:
  model = fdl.Config(starter_model.ConvClassifier, rngs=nnx.Rngs(0))

  learning_rate = 0.0005
  momentum = 0.9
  optax_optimizer = fdl.Config(
    optax.adam,
    learning_rate,
    momentum,
  )
  optimizer = fdl.Config(
    nnx.Optimizer,
    model,
    optax_optimizer,
  )

  # Load the Dataset.
  # TODO(jeffcarp): Make this the base configuration without loading this
  # dataset automatically.
  (x_train, y_train), (x_test, y_test) = data.prepare_ucr_dataset()
  print("Loaded dataset")
  print("Train size:", x_train.size)
  print("Test size:", x_test.size)
  train_loader, eval_loader = data.get_grain_datasets(
    x_train,
    y_train,
    x_test,
    y_test,
    train_batch_size,
    eval_batch_size,
  )

  # Make full log dir.
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
  log_dir = os.path.join(base_log_dir, f"run_{timestamp}")
  checkpoint_dir = os.path.join(log_dir, "checkpoints")
  os.makedirs(checkpoint_dir, exist_ok=True)

  return fdl.Config(
    TrainConfig,
    model=model,
    optimizer=optimizer,
    log_dir=log_dir,
    checkpoint_dir=checkpoint_dir,
    train_batch_size=train_batch_size,
    train_loader=train_loader,
    eval_loader=eval_loader,
  )


def text_classification(
  train_batch_size: int = 128,
  eval_batch_size: int = 128,
) -> fdl.Config:
  config = default_config()
  config.model = fdl.Config(starter_model.Transformer, rngs=nnx.Rngs(0))
  config.optimizer = fdl.Config(
    nnx.Optimizer, config.model, fdl.Config(optax.adam, 0.0005, 0.9)
  )

  index_from = 3  # make sure that 0 encodes pad token
  vocab_size = 20000  # Only consider the top 20k words
  maxlen = 200  # Only consider the first 200 words of each movie review
  (x_train, y_train), (x_test, y_test) = data.prepare_imdb_dataset(
    num_words=vocab_size,
    index_from=index_from,
  )
  print(len(x_train), "Training sequences")
  print(len(x_test), "Validation sequences")
  x_train = data.pad_sequences(x_train, max_len=maxlen)
  x_test = data.pad_sequences(x_test, max_len=maxlen)

  print("Loaded dataset")
  print("Train size:", x_train.size)
  print("Test size:", x_test.size)

  config.train_loader, config.eval_loader = data.get_grain_datasets(
    x_train,
    y_train,
    x_test,
    y_test,
    train_batch_size,
    eval_batch_size,
  )

  return config
