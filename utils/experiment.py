# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Experiment utils."""

from collections.abc import Mapping
from typing import Any

from absl import logging
import chex
import haiku as hk
import optax


Array = chex.Array


def log_params_info(params: hk.Params) -> None:
  """Log information about parameters."""
  num_params = hk.data_structures.tree_size(params)
  byte_size = hk.data_structures.tree_bytes(params)
  logging.info('%d params, size: %.2f MB', num_params, byte_size / 1e6)
  # print each parameter and its shape
  logging.info('Parameter shapes')
  for mod, name, value in hk.data_structures.traverse(params):
    logging.info('%s/%s: %s', mod, name, value.shape)


def partition_params_by_name(
    params: hk.Params, *, key: str
) -> tuple[hk.Params, hk.Params]:
  """Partition `params` along the `name` predicate checking for `key`.

  Note: Uses `in` as comparator; i.e., it checks whether `key in name`.

  Args:
    params: `hk.Params` to be partitioned.
    key: Key to check for in the `name`.

  Returns:
    Partitioned parameters; first params without key, then those with key.
  """
  predicate = lambda module_name, name, value: key in name
  with_key, without_key = hk.data_structures.partition(predicate, params)
  return without_key, with_key


def partition_params_by_module_name(
    params: hk.Params, *, key: str
) -> tuple[hk.Params, hk.Params]:
  """Partition `params` along the `module_name` predicate checking for `key`.

    Note: Uses `in` as comparator; i.e., it checks whether `key in module_name`.

  Args:
    params: `hk.Params` to be partitioned.
    key: Key to check for in the `module_name`.

  Returns:
    Partitioned parameters; first params without key, then those with key.
  """
  predicate = lambda module_name, name, value: key in module_name
  with_key, without_key = hk.data_structures.partition(predicate, params)
  return without_key, with_key


def merge_params(params_1: hk.Params, params_2: hk.Params) -> hk.Params:
  """Merges two sets of parameters into a single set of parameters."""
  # Ordering to mimic the old function structure.
  return hk.data_structures.merge(params_2, params_1)


def make_opt(
    transform_name: str,
    transform_kwargs: Mapping[str, Any],
    global_max_norm: float | None = None,
    cosine_decay_schedule: bool = False,
    cosine_decay_schedule_kwargs: Mapping[str, Any] | None = None,
    learning_rate: float | None = None,
) -> optax.GradientTransformation:
  """Creates optax optimizer that either uses a cosine schedule or fixed lr."""

  optax_list = []

  if global_max_norm is not None:
    # Optionally add clipping by global norm
    optax_list = [optax.clip_by_global_norm(max_norm=global_max_norm)]

  # The actual optimizer
  transform = getattr(optax, transform_name)
  optax_list.append(transform(**transform_kwargs))

  # Either use cosine schedule or fixed learning rate.
  if cosine_decay_schedule:
    assert cosine_decay_schedule_kwargs is not None
    assert learning_rate is None
    if 'warmup_steps' in cosine_decay_schedule_kwargs.keys():
      schedule = optax.warmup_cosine_decay_schedule
    else:
      schedule = optax.cosine_decay_schedule
    lr_schedule = schedule(**cosine_decay_schedule_kwargs)
    optax_list.append(optax.scale_by_schedule(lr_schedule))
  else:
    assert cosine_decay_schedule_kwargs is None
    assert learning_rate is not None
    optax_list.append(optax.scale(learning_rate))

  optax_list.append(optax.scale(-1))  # minimize the loss.
  return optax.chain(*optax_list)
