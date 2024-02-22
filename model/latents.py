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

"""Latent grids for C3."""

from collections.abc import Sequence
import functools
import math

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


Array = chex.Array


def soft_round(x, temperature):
  """Differentiable approximation to `jnp.round`.

  Lower temperatures correspond to closer approximations of the round function.
  For temperatures approaching infinity, this function resembles the identity.

  This function is described in Sec. 4.1 of the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  The temperature argument is the reciprocal of `alpha` in the paper.

  For convenience, we support `temperature = None`, which is the same as
  `temperature = inf`, which is the same as identity.

  Args:
    x: Array. Inputs to the function.
    temperature: Float >= 0. Controls smoothness of the approximation.

  Returns:
    Array of same shape as `x`.
  """
  if temperature is None:
    temperature = jnp.inf

  m = jnp.floor(x) + 0.5
  z = 2 * jnp.tanh(0.5 / temperature)
  r = jnp.tanh((x - m) / temperature) / z
  return m + r


def soft_round_inverse(x, temperature):
  """Inverse of `soft_round`.

  This function is described in Sec. 4.1 of the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  The temperature argument is the reciprocal of `alpha` in the paper.

  For convenience, we support `temperature = None`, which is the same as
  `temperature = inf`, which is the same as identity.

  Args:
    x: Array. Inputs to the function.
    temperature: Float >= 0. Controls smoothness of the approximation.

  Returns:
    Array of same shape as `x`.
  """
  if temperature is None:
    temperature = jnp.inf

  m = jnp.floor(x) + 0.5
  z = 2 * jnp.tanh(0.5 / temperature)
  r = jnp.arctanh((x - m) * z) * temperature
  return m + r


def soft_round_conditional_mean(x, temperature):
  """Conditional mean of inputs given noisy soft rounded values.

  Computes `g(z) = E[X | Q(X) + U = z]` where `Q` is the soft-rounding function,
  `U` is uniform between -0.5 and 0.5 and `X` is considered uniform when
  truncated to the interval `[z - 0.5, z + 0.5]`.

  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  Args:
    x: The input tensor.
    temperature: Float >= 0. Controls smoothness of the approximation.

  Returns:
    Array of same shape as `x`.
  """
  return soft_round_inverse(x - 0.5, temperature) + 0.5


class Latent(hk.Module):
  """Hierarchical latent representation of C3.

  Notes:
    Based on https://github.com/Orange-OpenSource/Cool-Chic.
  """

  def __init__(
      self, *,
      input_res: tuple[int, ...],
      num_grids: int,
      downsampling_exponents: tuple[float, ...] | None,
      add_gains: bool = True,
      learnable_gains: bool = False,
      gain_values: Sequence[float] | None = None,
      gain_factor: float | None = None,
      q_step: float = 1.,
      init_fn: hk.initializers.Initializer = jnp.zeros,
      downsampling_factor: float | tuple[float, ...] = 2.,
  ):
    """Constructor.

    The size of the i-th dimension of the k-th latent grid will be
    `input_res[i] / (downsampling_factor[i] ** downsampling_exponents[k])`.

    Args:
      input_res: Size of image as (H, W) or video as (T, H, W).
      num_grids: Number of latent grids, each of different resolution. For
        example if `input_res = (512, 512)` and `num_grids = 3`, then by default
        the latent grids will have sizes (512, 512), (256, 256), (128, 128).
      downsampling_exponents: Determines how often each grid is downsampled. If
        provided, should be of length `num_grids`. By default the first grid
        has the same resolution as the input and the last grid is downsampled
        by a factor of `downsampling_factor ** (num_grids - 1)`.
      add_gains: Whether to add gains used in COOL-CHIC paper.
      learnable_gains: Whether gains should be learnable.
      gain_values: Optional. If provided, use these values to initialize the
        gains.
      gain_factor: Optional. If provided, use this value as a gain factor to
        initialize the gains.
      q_step: Step size used for quantization. Defaults to 1.
      init_fn: Init function for grids. Defaults to `jnp.zeros`.
      downsampling_factor: Downsampling factor for each grid of latents. This
        can be a float or a tuple of length equal to the length of `input_size`.
    """
    super().__init__()
    self.input_res = input_res
    self.num_grids = num_grids
    self.add_gains = add_gains
    self.learnable_gains = learnable_gains
    self.q_step = q_step
    if gain_values is not None or gain_factor is not None:
      assert add_gains, '`add_gains` must be True to use `gain_values`.'
      assert gain_values is None or gain_factor is None, (
          'Can only use one out of `gain_values` or `gain_factors` but both'
          ' were provided.'
      )

    if downsampling_exponents is None:
      downsampling_exponents = range(num_grids)
    else:
      assert len(downsampling_exponents) == num_grids

    num_dims = len(input_res)

    # Convert downsampling_factor to a tuple if not already a tuple.
    if isinstance(downsampling_factor, (int, float)):
      df = (downsampling_factor,) * num_dims
    else:
      assert len(downsampling_factor) == num_dims
      df = downsampling_factor

    if learnable_gains:
      assert add_gains, '`add_gains` must be True to use `learnable_gains`.'

    # Initialize latent grids
    self._latent_grids = []
    for i, exponent in enumerate(downsampling_exponents):
      # Latent grid sizes of ({T / df[-3]^j}, H / df[-2]^j, W / df[-1]^j)
      latent_grid = [
          int(math.ceil(x / (df[dim] ** exponent)))
          for dim, x in enumerate(input_res)
      ]
      self._latent_grids.append(
          hk.get_parameter(f'latent_grid_{i}', latent_grid, init=init_fn)
      )
    self._latent_grids = tuple(self._latent_grids)

    # Optionally initialise gains
    if self.add_gains:
      if gain_values is not None:
        assert len(gain_values) == self.num_grids
        gains = jnp.array(gain_values)
      elif gain_factor is not None:
        gains = jnp.array([gain_factor ** j for j in downsampling_exponents])
      else:
        # Use geometric mean of downsampling factors to compute gains_factor.
        gain_factor = np.prod(df) ** (1/num_dims)
        gains = jnp.array([gain_factor ** j for j in downsampling_exponents])

      if self.learnable_gains:
        self._gains = hk.get_parameter(
            'gains',
            shape=(self.num_grids,),
            init=lambda *_: gains,
        )
      else:
        self._gains = gains

  @property
  def gains(self) -> Array:
    """Latents are multiplied by these values before quantization."""
    if self.add_gains:
      return self._gains
    return jnp.ones(self.num_grids)

  @property
  def latent_grids(self) -> tuple[Array, ...]:
    """Optionally add gains to latents (following COOL-CHIC paper)."""
    return tuple(grid * gain for grid, gain
                 in zip(self._latent_grids, self._gains))

  def __call__(
      self,
      quant_type: str = 'none',
      soft_round_temp: float | None = None,
      kumaraswamy_a: float | None = None,
  ) -> tuple[Array, ...]:
    """Upsamples each latent grid and concatenates them to a single array.

    Args:
      quant_type: Type of quantization to use. One of either: "none": No
        quantization is applied. "noise": Quantization is simulated by adding
        uniform noise. Used at training time. "round": Quantization is applied
        by rounding array entries to nearest integer. Used at test time. "ste":
        Straight through estimator. Quantization is applied by rounding and
        gradient is set to 1.
      soft_round_temp: The temperature to use for the soft-rounded dither for
        quantization. Optional. Has to be passed when using `quant_type =
        'soft_round'`.
      kumaraswamy_a: Optional `a` parameter of the Kumaraswamy distribution to
        determine the noise that is used for noise quantization. The `b`
        parameter of the Kumaraswamy distribution is computed such that the mode
        of the distribution is at 0.5. For `a = 1` the distribution is uniform.
        For `a > 1` the distribution is more peaked around `0.5` and increasing
        `a` decreased the variance of the distribution.

    Returns:
     Concatenated upsampled latents as array of shape (*input_size, num_grids)
     and quantized latent_grids as list of arrays.
    """
    # Optionally apply quantization (quantize just returns latent_grid if
    # quant_type is "none")
    latent_grids = jax.tree_map(
        functools.partial(
            quantize,
            quant_type=quant_type,
            q_step=self.q_step,
            soft_round_temp=soft_round_temp,
            kumaraswamy_a=kumaraswamy_a,
        ),
        self.latent_grids,
    )

    return latent_grids


def kumaraswamy_inv_cdf(x: Array, a: chex.Numeric, b: chex.Numeric) -> Array:
  """Inverse CDF of Kumaraswamy distribution."""
  return (1 - (1 - x) ** (1 / b)) ** (1 / a)


def kumaraswamy_b_fn(a: chex.Numeric) -> chex.Numeric:
  """Returns `b` of Kumaraswamy distribution such that mode is at 0.5."""
  return (2**a * (a - 1) + 1) / a


def quantize(
    arr: Array,
    quant_type: str = 'noise',
    q_step: float = 1.0,
    soft_round_temp: float | None = None,
    kumaraswamy_a: chex.Numeric | None = None,
) -> Array:
  """Quantize array.

  Args:
    arr: Float array to be quantized.
    quant_type: Type of quantization to use. One of either: "none": No
      quantization is applied. "noise": Quantization is simulated by adding
      uniform noise. Used at training time. "round": Quantization is applied by
      rounding array entries to nearest integer. Used at test time. "ste":
      Straight through estimator. Quantization is applied by rounding and
      gradient is set to 1. "soft_round": Soft-rounding is applied before and
      after adding noise. "ste_soft_round": Quantization is applied by rounding
      and gradient uses soft-rounding.
    q_step: Step size used for quantization. Defaults to 1.
    soft_round_temp: Smoothness of soft-rounding. Values close to 0 correspond
      to close approximations of hard quantization ("round"), large values
      correspond to smooth functions and identity in the limit ("noise").
    kumaraswamy_a: Optional `a` parameter of the Kumaraswamy distribution to
      determine the noise that is used for noise quantization. If `None`, use
      uniform noise. The `b` parameter of the Kumaraswamy distribution is
      computed such that the mode of the distribution is at 0.5. For `a = 1` the
      distribution is uniform. For `a > 1` the distribution is more peaked
      around `0.5` and increasing `a` decreased the variance of the
      distribution.

  Returns:
    Quantized array.

  Notes:
    Setting `quant_type` to "soft_round" simulates quantization by applying a
    point-wise nonlinearity (soft-rounding) before and after adding noise:

        y = r(s(x) + u)

    Here, `r` and `s` are differentiable approximations of rounding, with
    `r(z) = E[X | s(X) + U = z]` for some assumptions on `X`. Both depend
    on a temperature parameter, `soft_round_temp`. For detailed definitions,
    see Sec. 4.1 of Agustsson & Theis (2020; https://arxiv.org/abs/2006.09952).
  """

  # First map inputs to scaled space where each bin has width 1.
  arr = arr / q_step
  if quant_type == 'none':
    pass
  elif quant_type == 'noise':
    # Add uniform noise U(-0.5, 0.5) during training.
    arr = arr + jax.random.uniform(hk.next_rng_key(), shape=arr.shape) - 0.5
  elif quant_type == 'round':
    # Round at test time
    arr = jnp.round(arr)
  elif quant_type == 'ste':
    # Numerically correct straight through estimator. See
    # https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#straight-through-estimator-using-stop-gradient
    zero = arr - jax.lax.stop_gradient(arr)
    arr = zero + jax.lax.stop_gradient(jnp.round(arr))
  elif quant_type == 'soft_round':
    if soft_round_temp is None:
      raise ValueError(
          '`soft_round_temp` must be specified if `quant_type` is `soft_round`.'
      )
    noise = jax.random.uniform(hk.next_rng_key(), shape=arr.shape)
    if kumaraswamy_a is not None:
      kumaraswamy_b = kumaraswamy_b_fn(kumaraswamy_a)
      noise = kumaraswamy_inv_cdf(noise, kumaraswamy_a, kumaraswamy_b)
    noise = noise - 0.5
    arr = soft_round(arr, soft_round_temp)
    arr = arr + noise
    arr = soft_round_conditional_mean(arr, soft_round_temp)
  elif quant_type == 'ste_soft_round':
    if soft_round_temp is None:
      raise ValueError(
          '`ste_soft_round_temp` must be specified if `quant_type` is '
          '`ste_soft_round`.'
      )
    fwd = jnp.round(arr)
    bwd = soft_round(arr, soft_round_temp)
    arr = bwd + jax.lax.stop_gradient(fwd - bwd)
  else:
    raise ValueError(f'Unknown quant_type: {quant_type}')
  # Map inputs back to original range
  arr = arr * q_step
  return arr
