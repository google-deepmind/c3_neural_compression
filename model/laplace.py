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

"""Laplace distributions for entropy model."""

import chex
import distrax
import jax
import jax.numpy as jnp


Array = chex.Array
Numeric = chex.Numeric


def log_expbig_minus_expsmall(big: Numeric, small: Numeric) -> Array:
  """Stable implementation of `log(exp(big) - exp(small))`.

  Taken from `distrax.utils` as it is a private function there.

  Args:
    big: First input.
    small: Second input. It must be `small <= big`.

  Returns:
    The resulting `log(exp(big) - exp(small))`.
  """
  return big + jnp.log1p(-jnp.exp(small - big))


class Laplace(distrax.Laplace):
  """Laplace distribution with integrated log probability."""

  def __init__(
      self,
      loc: Numeric,
      scale: Numeric,
      eps: Numeric | None = None,
  ) -> None:
    super().__init__(loc=loc, scale=scale)
    self._eps = eps

  def integrated_log_prob(self, x: Numeric) -> Array:
    """Returns integrated log_prob in (x - 0.5, x + 0.5)."""
    # Numerically stable implementation taken from `distrax.Quantized.log_prob`.

    log_cdf_big = self.log_cdf(x + 0.5)
    log_cdf_small = self.log_cdf(x - 0.5)
    log_sf_small = self.log_survival_function(x + 0.5)
    log_sf_big = self.log_survival_function(x - 0.5)
    # Use the survival function instead of the CDF when its value is smaller,
    # which happens to the right of the median of the distribution.
    big = jnp.where(log_sf_small < log_cdf_big, log_sf_big, log_cdf_big)
    small = jnp.where(log_sf_small < log_cdf_big, log_sf_small, log_cdf_small)
    if self._eps is not None:
      # use stop_gradient to block updating in this case
      big = jnp.where(
          big - small > self._eps, big, jax.lax.stop_gradient(small) + self._eps
      )
    log_probs = log_expbig_minus_expsmall(big, small)

    # Return -inf and not NaN when `log_cdf` or `log_survival_function` are
    # infinite (i.e. probability = 0). This can happen for extreme outliers.
    is_outside = jnp.logical_or(jnp.isinf(log_cdf_big), jnp.isinf(log_sf_big))
    log_probs = jnp.where(is_outside, -jnp.inf, log_probs)
    return log_probs
