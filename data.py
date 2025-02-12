import io
import os
import typing

import numpy as np
import grain.python as grain
from urllib import request
import requests


def prepare_ucr_dataset() -> tuple:
  root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
  save_dir = "/tmp/forda"

  # Create the directory if it doesn't exist
  os.makedirs(save_dir, exist_ok=True)

  train_path = os.path.join(save_dir, "FordA_TRAIN.tsv")
  test_path = os.path.join(save_dir, "FordA_TEST.tsv")

  # Download the dataset if it doesn't exist
  if not os.path.exists(train_path):
    request.urlretrieve(root_url + "FordA_TRAIN.tsv", train_path)
  if not os.path.exists(test_path):
    request.urlretrieve(root_url + "FordA_TEST.tsv", test_path)

  train_data = np.loadtxt(train_path, delimiter="\t")
  x_train, y_train = train_data[:, 1:], train_data[:, 0].astype(int)

  test_data = np.loadtxt(test_path, delimiter="\t")
  x_test, y_test = test_data[:, 1:], test_data[:, 0].astype(int)

  x_train = x_train.reshape((*x_train.shape, 1))
  x_test = x_test.reshape((*x_test.shape, 1))

  rng = np.random.RandomState(113)
  indices = rng.permutation(len(x_train))
  x_train = x_train[indices]
  y_train = y_train[indices]

  y_train[y_train == -1] = 0
  y_test[y_test == -1] = 0

  return (x_train, y_train), (x_test, y_test)


class DataSource(grain.RandomAccessDataSource):
  def __init__(self, x, y):
    self._x = x
    self._y = y

  def __getitem__(self, idx):
    return {"measurement": self._x[idx], "label": self._y[idx]}

  def __len__(self):
    return len(self._x)


def get_grain_datasets(
  x_train: np.array,
  y_train: np.array,
  x_test: np.array,
  y_test: np.array,
  train_batch_size: int,
  eval_batch_size: int,
  seed: int = 42,
) -> tuple[grain.DataLoader, grain.DataLoader]:
  train_source = DataSource(x_train, y_train)
  test_source = DataSource(x_test, y_test)

  train_sampler = grain.IndexSampler(
    len(train_source),
    shuffle=True,
    seed=seed,
    shard_options=grain.NoSharding(),
  )

  test_sampler = grain.IndexSampler(
    len(test_source),
    shuffle=False,
    seed=seed,
    shard_options=grain.NoSharding(),
    num_epochs=1,
  )

  train_loader = grain.DataLoader(
    data_source=train_source,
    sampler=train_sampler,
    worker_count=2,
    worker_buffer_size=2,
    operations=[
      grain.Batch(train_batch_size, drop_remainder=True),
    ],
  )

  test_loader = grain.DataLoader(
    data_source=test_source,
    sampler=test_sampler,
    worker_count=2,
    worker_buffer_size=2,
    operations=[
      grain.Batch(eval_batch_size),
    ],
  )

  return train_loader, test_loader


def prepare_imdb_dataset(num_words: int, index_from: int, oov_char: int = 2) -> tuple:
  response = requests.get(
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"
  )
  response.raise_for_status()
  with np.load(io.BytesIO(response.content), allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]
  print("DEBUG ------------ IMBD DATASET LOADED")

  rng = np.random.RandomState(113)
  indices = np.arange(len(x_train))
  rng.shuffle(indices)
  x_train = x_train[indices]
  y_train = y_train[indices]

  indices = np.arange(len(x_test))
  rng.shuffle(indices)
  x_test = x_test[indices]
  y_test = y_test[indices]

  x_train = [[w + index_from for w in x] for x in x_train]
  x_test = [[w + index_from for w in x] for x in x_test]

  xs = x_train + x_test
  labels = np.concatenate([y_train, y_test])
  xs = [[w if w < num_words else oov_char for w in x] for x in xs]

  idx = len(x_train)
  x_train, y_train = np.array(xs[:idx], dtype="object"), labels[:idx]
  x_test, y_test = np.array(xs[idx:], dtype="object"), labels[idx:]

  return (x_train, y_train), (x_test, y_test)


def pad_sequences(arrs: typing.Iterable, max_len: int) -> np.ndarray:
  # Ensure that each sample is the same length
  result = []
  for arr in arrs:
    arr_len = len(arr)
    if arr_len < max_len:
      padded_arr = np.pad(arr, (max_len - arr_len, 0), "constant", constant_values=0)
    else:
      padded_arr = np.array(arr[arr_len - max_len :])
    result.append(padded_arr)

  return np.asarray(result)
