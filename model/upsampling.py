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

"""Upsampling layers for latent grids."""

import functools

import chex
import jax
import jax.numpy as jnp

Array = chex.Array


def jax_image_upsampling(latent_grids: tuple[Array, ...],
                         input_res: tuple[int, ...],
                         interpolation_method: str,
                         **unused_kwargs) -> Array:
  """Returns upsampled latents stacked along last dim: ({T}, H, W, num_grids).

  Uses `jax.image.resize` with `interpolation_method` for upsampling. Upsamples
  each latent grid separately to the size of the largest grid.

  Args:
    latent_grids: Tuple of latent grids of size ({T}, H, W), ({T/2}, H/2, W/2),
      etc.
    input_res: Resolution to which latent_grids are upsampled.
    interpolation_method: Interpolation method.
  """
  # First latent grid is assumed to be the largest and corresponds to the input
  # shape of the image or video ({T}, H, W).
  assert len(latent_grids[0].shape) == len(input_res)
  # input_size = latent_grids[0].shape
  upsampled_latents = jax.tree_map(
      functools.partial(
          jax.image.resize,
          shape=input_res,
          method=interpolation_method,
      ),
      latent_grids,
  )  # Tuple of latent grids of shape ({T}, H, W)
  return jnp.stack(upsampled_latents, axis=-1)
