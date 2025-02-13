"""JAX Starter Pack

Via:
https://docs.jaxstack.ai/en/latest/JAX_time_series_classification.html
"""

import os
from typing import Callable

from absl import app
from flax import nnx
from fiddle import absl_flags as fdl_flags
import fiddle as fdl
import jax
import jax.numpy as jnp
import grain.python as grain
import numpy as np
import optax
import orbax.checkpoint as ocp
from torch.utils import tensorboard
import tqdm

import config as starter_config

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.sdk.resources import Resource

# Configure resource attributes (service name, etc.)
resource = Resource(attributes={"service.name": "jax-ml-training"})

_COLLECTOR_ADDRESS = '0.0.0.0:4317' # test

metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=_COLLECTOR_ADDRESS)
)
metrics_provider = MeterProvider(
    resource=resource,
    metric_readers=[metric_reader],
)
metrics.set_meter_provider(metrics_provider)
meter = metrics.get_meter(__name__)

_CONFIG_FLAG = fdl_flags.DEFINE_fiddle_config(
  "config",
  help_string="The name of the Fiddle config",
  # default_module=sys.modules[__name__],
  default_module=starter_config,
)


def compute_losses_and_logits(
  model: nnx.Module,
  batch_tokens: jax.Array,
  labels: jax.Array,
):
  logits = model(batch_tokens)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=labels
  ).mean()
  return loss, logits


@nnx.jit
def train_step(
  model: nnx.Module,
  optimizer: nnx.Optimizer,
  batch: dict[str, jax.Array],
):
  batch_tokens = jnp.array(batch["measurement"])
  labels = jnp.array(batch["label"], dtype=jnp.int32)

  grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch_tokens, labels)
  optimizer.update(grads)  # In-place update.
  return loss


@nnx.jit
def eval_step(
  model: nnx.Module,
  batch: dict[str, jax.Array],
  eval_metrics: nnx.MultiMetric,
):
  batch_tokens = jnp.array(batch["measurement"])
  labels = jnp.array(batch["label"], dtype=jnp.int32)
  loss, logits = compute_losses_and_logits(model, batch_tokens, labels)

  eval_metrics.update(
    loss=loss,
    logits=logits,
    labels=labels,
  )


def evaluate_model(
  model: nnx.Module,
  step: int,
  eval_metrics,
  eval_loader: grain.RandomAccessDataSource,
  summary_writer: tensorboard.SummaryWriter,
):
  print(f"Evaluating model at step {step}...")
  model.eval()

  eval_metrics.reset()
  for test_batch in tqdm.tqdm(eval_loader):
    eval_step(model, test_batch, eval_metrics)

  for metric, value in eval_metrics.compute().items():
    key = f"eval/{metric}"
    summary_writer.add_scalar(key, np.array(value), step)

  summary_writer.flush()


# TODO(jeffcarp): Turn this into an officially provided hook.
def tb_duration_event(
  get_step_fn: Callable[[], int],
  summary_writer: tensorboard.SummaryWriter,
  event: str,
  duration: float,
):
  """Creates a TB event listener for jax.monitoring.

  Requires partial application of the first two params.
  """
  # Clip first directory so the TB group names are more meaningful.
  if event.startswith("/jax/"):
    event = event[5:]
  step = get_step_fn()
  summary_writer.add_scalar(event.lstrip("/jax/"), duration, step)


def train(config: starter_config.TrainConfig):
  print("Training with config:", config)
  step = 0
  model = config.model
  optimizer = config.optimizer
  summary_writer = tensorboard.SummaryWriter(config.log_dir)
  print(f"Logging TensorBoard metrics to: {config.log_dir}")
  print(nnx.display(model))

  # Set up jax.monitoring event listener.

  eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
    accuracy=nnx.metrics.Accuracy(),
  )
  checkpointer = ocp.StandardCheckpointer()

  progress_bar = tqdm.tqdm(
    enumerate(config.train_loader),
    total=config.train_total_steps,
  )
  for step, batch in progress_bar:
    model.train()

    if config.run_profile and step == config.profile_start_step:
      print("STARTING TRACE...")
      jax.profiler.start_trace(config.log_dir)
    elif config.run_profile and step == config.profile_end_step:
      print("ENDING TRACE...")
      jax.profiler.stop_trace()

    loss = train_step(model, optimizer, batch)
    progress_bar.set_postfix({"loss": loss.item()})
    # Record metrics
    meter.create_counter("training_loss").add(loss)

    if step % config.summary_interval_steps == 0 and step > 0:
      print(f"Writing summaries to {config.log_dir}...")
      summary_writer.add_scalar("train/loss", np.array(loss.item()), step)
      summary_writer.flush()

    if step % config.eval_interval_steps == 0 and step > 0:
      evaluate_model(
        model=model,
        step=step,
        eval_metrics=eval_metrics,
        eval_loader=config.eval_loader,
        summary_writer=summary_writer,
      )

    if step % config.checkpoint_interval_steps == 0 and step > 0:
      print(f"Writing checkpoint to {config.log_dir}...")
      _, state = nnx.split(model)
      pure_dict_state = state.to_pure_dict()
      checkpointer.save(
        os.path.join(config.checkpoint_dir, str(step)),
        pure_dict_state,
      )

    if step >= config.train_total_steps:
      break


def main(argv):

  # DEBUGGING
  meter.create_counter("training_loss").add(123)
  print('DEBUG --- CONFIG SENT')

  #buildable = _CONFIG_FLAG.value or starter_config.default_config()
  #config = fdl.build(buildable)
  #train(config)


if __name__ == "__main__":
  app.run(main)
