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

"""Quantization and entropy coding for synthesis and entropy models."""

import abc
import collections
from collections.abc import Hashable, Mapping, Sequence
import functools
import math

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from c3_neural_compression.model import laplace

Array = chex.Array


def _unnested_to_nested_dict(d: Mapping[str, jax.Array]) -> hk.Params:
  """Restructure an unnested (flat) mapping into a nested (2-level) dictionary.

  This function is used to convert the unnested (flat or 1-level) mapping of
  parameters as returned by `hk.Module.params_dict()` into a regular nested
  (2-level) mapping of `hk.Params`. For example, it maps:
  ```
    {'linear/w': ..., 'linear/b': ...} -> {'linear': {'w': ..., 'b': ...}}
  ```
  Also see the corresponding test for usage.

  Args:
    d: Dictionary of arrays as returned by `hk.Module().params_dict()`

  Returns:
    A two-level mapping of type `hk.Params`.
  """
  out = collections.defaultdict(dict)
  for name, value in d.items():
    # Everything before the last `'/'` is the module_name in `haiku`.
    module_name, name = name.rsplit('/', 1)
    out[module_name][name] = value
  return dict(out)


def _unflatten_and_unmask(
    flat_masked_array: Array, unflat_mask: Array
) -> Array:
  """Returns unmasked `flat_arr` reshaped into `unflat_mask.shape`.

  This function undoes the operation `arr[unflat_mask].flatten()` on unmasked
  entries and fills masked entries with zeros. See the corresponding test for a
  usage example.

  Args:
    flat_masked_array: The flat (1D) array to unflatten and unmask.
    unflat_mask: Binary mask used to select entries in the original array before
      flattening. `1` corresponds to "keep" whereas `0` correspond to "discard".
      Masked out entries are set to `0.` in the unmasked array. There should be
      as many `1`s in `unflat_mask` as there are entries in `flat_masked_arr`.
  """
  chex.assert_rank(flat_masked_array, 1)
  if np.all(unflat_mask == np.ones_like(unflat_mask)):
    out = flat_masked_array
  else:
    if np.sum(unflat_mask) != len(flat_masked_array):
      raise ValueError(
          '`unflat_mask` should have as many `1`s as `flat_masked_array` has'
          ' entries.'
      )
    out = []
    array_idx = 0
    for mask in unflat_mask.flatten():
      if mask == 1:
        out.append(flat_masked_array[array_idx])
        array_idx += 1
      else:
        out.append(0.0)  # Masked entries are filled in with `0`.
    out = np.array(out)
    # After adding back masked entries, `out` should have as many entries as the
    # mask.
    assert len(out) == len(unflat_mask.flatten())
  return np.reshape(out, unflat_mask.shape)


def _mask_and_flatten(arr: Array, mask: Array) -> Array:
  """Returns masked and flattened copy of `arr`."""
  if mask.dtype != bool:
    raise TypeError('`mask` needs to be boolean.')
  return arr[mask].flatten()


class QuantizableMixin(abc.ABC):
  """Mixin to add quantization and rate computation methods.

  This class is used as a Mixin to `hk.Module` to add quantization and rate
  computation methods directly in the components of the overall C3 model.
  """

  def _treat_as_weight(self, name: Hashable, shape: Sequence[int]) -> bool:
    """Whether a module parameter should be treated as weight."""
    del name
    return len(shape) > 1

  def _treat_as_bias(self, name: Hashable, shape: Sequence[int]) -> bool:
    """Whether a module parameter should be treated as bias."""
    return not self._treat_as_weight(name, shape)

  def _get_mask(
      self,
      dictkey: tuple[jax.tree_util.DictKey, ...],  # From `tree_map_with_path`.
      parameter_array: Array,
  ) -> Array | None:
    """Return mask for a particular module parameter `arr`."""
    del dictkey  # Do not mask out anything by default.
    return np.ones(parameter_array.shape, dtype=bool)

  @abc.abstractmethod
  def params_dict(self) -> Mapping[str, Array]:
    """Returns the parameter dictionary; implemented in `hk.Module`."""
    raise NotImplementedError()

  def _quantize_array_to_int(
      self,
      dictkey: tuple[jax.tree_util.DictKey, ...],  # From `tree_map_with_path`.
      arr: Array,
      q_step_weight: float,
      q_step_bias: float,
  ) -> Array:
    """Quantize `arr` into integers according to `q_step`."""
    if self._treat_as_weight(dictkey[0].key, arr.shape):
      q_step = q_step_weight
    elif self._treat_as_bias(dictkey[0].key, arr.shape):
      q_step = q_step_bias
    else:
      raise ValueError(f'{dictkey[0].key} is neither weight nor bias.')
    q = quantize_at_step(arr, q_step=q_step)
    return q.astype(jnp.float32)

  def _scale_quantized_array_by_q_step(
      self,
      dictkey: tuple[jax.tree_util.DictKey, ...],  # From `tree_map_with_path`.
      arr: Array,
      q_step_weight: float,
      q_step_bias: float,
  ) -> Array:
    """Scale quantized integer array `arr` by the corresponding `q_step`."""
    if self._treat_as_weight(dictkey[0].key, arr.shape):
      q_step = q_step_weight
    elif self._treat_as_bias(dictkey[0].key, arr.shape):
      q_step = q_step_bias
    else:
      raise ValueError(f'{dictkey[0].key} is neither weight nor bias.')
    return arr * q_step

  def get_quantized_nested_params(
      self, q_step_weight: float, q_step_bias: float
  ) -> hk.Params:
    """Returnes quantized but rescaled float parameters (nested `hk.Params`)."""
    quantized_params_int = jax.tree_util.tree_map_with_path(
        functools.partial(
            self._quantize_array_to_int,
            q_step_weight=q_step_weight,
            q_step_bias=q_step_bias,
        ),
        self.params_dict(),
    )
    quantized_params = jax.tree_util.tree_map_with_path(
        functools.partial(
            self._scale_quantized_array_by_q_step,
            q_step_weight=q_step_weight,
            q_step_bias=q_step_bias,
        ),
        quantized_params_int,
    )

    quantized_params = _unnested_to_nested_dict(quantized_params)
    return quantized_params

  def get_quantized_masked_flattened_params(
      self, q_step_weight: float, q_step_bias: float
  ) -> tuple[Mapping[str, Array], Mapping[str, Array]]:
    """Quantize, mask, and flatten the parameters of the module.

    Args:
      q_step_weight: Quantization step used to quantize the weights of the
        module. Weights are all the parameters for which `_treat_as_weight`
        returns True.
      q_step_bias: Quantization step used to quantize the biases of the module.

    Returns:
      Tuple of mappings of keys (strings) to masked and flattened arrays. The
      first mapping are the quantized integer values; the second mapping are the
      quantized but rescaled float values.
    """

    quantized_params_int = jax.tree_util.tree_map_with_path(
        functools.partial(
            self._quantize_array_to_int,
            q_step_weight=q_step_weight,
            q_step_bias=q_step_bias,
        ),
        self.params_dict(),
    )
    quantized_params = jax.tree_util.tree_map_with_path(
        functools.partial(
            self._scale_quantized_array_by_q_step,
            q_step_weight=q_step_weight,
            q_step_bias=q_step_bias,
        ),
        quantized_params_int,
    )

    mask = jax.tree_util.tree_map_with_path(self._get_mask, quantized_params)
    quantized_params_int = jax.tree_map(
        _mask_and_flatten, quantized_params_int, mask
    )
    quantized_params = jax.tree_map(_mask_and_flatten, quantized_params, mask)
    return (quantized_params_int, quantized_params)

  def unmask_and_unflatten_params(
      self,
      quantized_params_int: Mapping[str, Array],
      q_step_weight: float,
      q_step_bias: float,
  ) -> tuple[Mapping[str, Array], Mapping[str, Array]]:
    """Unmask and reshape the quantized parameters of the module.

    The masks and shapes are inferred from `self.params_dict()`.

    Note, the keys of `quantized_params_int` have to match the keys of
    `self.params_dict()`.

    Args:
      quantized_params_int: The quantized integer values (masked and flattened)
        to be reshaped and unmasked.
      q_step_weight: Quantization step that was used to quantize the weights.
      q_step_bias: Quantization step that was used to quantize the biases.

    Returns:
      Tuple of mappings of keys (strings) to arrays. The
      first mapping are the quantized integer values; the second mapping are the
      quantized but rescaled float values.

    Raises:
      KeyError: If the keys of `quantized_params_int` do not agree with keys of
      `self.quantized_params()`.
    """
    mask = jax.tree_util.tree_map_with_path(self._get_mask, self.params_dict())
    if set(mask.keys()) != set(quantized_params_int.keys()):
      raise KeyError(
          'Keys of `quantized_params_int` and `self.quantized_params()` should'
          ' be identical.'
      )
    quantized_params_int = jax.tree_map(
        _unflatten_and_unmask, quantized_params_int, mask
    )
    quantized_params = jax.tree_util.tree_map_with_path(
        functools.partial(
            self._scale_quantized_array_by_q_step,
            q_step_weight=q_step_weight,
            q_step_bias=q_step_bias,
        ),
        quantized_params_int,
    )
    return quantized_params_int, quantized_params

  def compute_rate(self, q_step_weight: float, q_step_bias: float) -> Array:
    """Compute the total rate of the module parameters for given `q_step`s.

    Args:
      q_step_weight: Quantization step for the weights.
      q_step_bias: Quantization step for the biases.

    Returns:
      Sum of all rates.
    """
    quantized_params_int, _ = self.get_quantized_masked_flattened_params(
        q_step_weight, q_step_bias
    )
    rates = jax.tree_map(laplace_rate, quantized_params_int)
    return sum(rates.values())


def quantize_at_step(x: Array, q_step: float) -> Array:
  """Returns quantized version of `x` at quantizaton step `q_step`.

  Args:
    x: Unquantized array of any shape.
    q_step: Quantization step.
  """
  return jnp.round(x / q_step)


def laplace_scale(x: Array) -> Array:
  """Estimate scale parameter of a zero-mean Laplace distribution.

  Args:
    x: Samples of a presumed zero-mean Laplace distribution.

  Returns:
    Estimate of the scale parameter of a zero-mean Laplace distribution.
  """
  return jnp.std(x) / math.sqrt(2)


def laplace_rate(
    x: Array, eps: float = 1e-3, mask: Array | None = None
) -> float:
  """Compute rate of array under Laplace distribution.

  Args:
    x: Quantized array of any shape.
    eps: Epsilon used to ensure the scale of Laplace distribution is not too
      close to zero (for numerical stability).
    mask: Optional mask for masking out entries of `x` for the computation of
      the rate. Also excludes these entries of `x` in the computation of the
      scale of the Laplace.

  Returns:
    Rate of quantized array under Laplace distribution.
  """
  # Compute the discrete probabilities using the Laplace CDF. The mean is set to
  # 0 and the scale is set to std / sqrt(2) following the COOL-CHIC code. See:
  # https://github.com/Orange-OpenSource/Cool-Chic/blob/16c41c033d6fd03e9f038d4f37d1ca330d5f7e35/src/models/mlp_coding.py#L61
  loc = jnp.zeros_like(x)
  # Ensure scale is not too close to zero for numerical stability
  # Optionally only use not masked out values for std computation.
  scale = max(laplace_scale(x[mask]), eps)
  scale = jnp.ones_like(x) * scale

  dist = laplace.Laplace(loc, scale)
  log_probs = dist.integrated_log_prob(x)
  # Change base of logarithm
  rate = -log_probs / jnp.log(2.0)

  # No value can cost more than 32 bits
  rate = jnp.clip(rate, a_max=32)

  return jnp.sum(rate, where=mask)  # pytype: disable=bad-return-type  # jnp-type
