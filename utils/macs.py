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

"""Utility functions for mutiply-accumulate (MAC) calculations for C3."""
from collections.abc import Mapping
import math
from typing import Any

from ml_collections import config_dict
import numpy as np


def mlp_macs_per_output_pixel(layer_sizes: tuple[int, ...]) -> int:
  """Number of MACs per output pixel for a forward pass of a MLP.

  Args:
    layer_sizes: Sizes of layers of MLP including input and output layers.

  Returns:
    MACs per output pixel to compute forward pass.
  """
  return sum(
      [f_in * f_out for f_in, f_out in zip(layer_sizes[:-1], layer_sizes[1:])]
  )


def conv_macs_per_output_pixel(
    kernel_shape: int | tuple[int, ...], f_in: int, f_out: int, ndim: int
) -> int:
  """Number of MACs per output pixel for a forward pass of a ConvND layer.

  Args:
    kernel_shape: Size of convolution kernel.
    f_in: Number of input channels.
    f_out: Number of output channels.
    ndim: number of dims of kernel. 2 for images and 3 for videos.

  Returns:
    MACs to compute forward pass.
  """
  if isinstance(kernel_shape, int):
    kernel_size = kernel_shape ** ndim
  else:
    assert len(kernel_shape) == ndim
    kernel_size = np.prod(kernel_shape)
  return int(f_in * f_out * kernel_size)


def macs_per_pixel_upsampling(
    num_grids: int,
    upsampling_type: str,
    interpolation_method: str,
    is_video: bool,
    **unused_kwargs,
) -> tuple[int, int]:
  """Computes the number of MACs per pixel for upsampling of latent grids.

  Assume that the largest grid has the size of the input image and is not
  changed.

  Args:
    num_grids: Number of latent grids.
    upsampling_type: Method to use for upsampling.
    interpolation_method: Method to use for interpolation.
    is_video: Whether latent grids are for image or video.
    unused_kwargs: Unused keyword arguments.

  Returns:
    macs_pp: MACs per pixel to compute forward pass
    upsampling_macs_pp: Macs per output pixel for upsampling.
  """
  if interpolation_method == 'bilinear':
    if is_video:
      upsampling_macs_pp = 16
    else:
      upsampling_macs_pp = 8
  else:
    raise ValueError(f'Unknown interpolation method: {interpolation_method}')

  if upsampling_type == 'image_resize':
    macs_pp = upsampling_macs_pp * (num_grids - 1)
  else:
    raise ValueError(f'Unknown upsampling type: {upsampling_type}')

  return macs_pp, upsampling_macs_pp


def get_macs_per_pixel(
    *,
    input_shape: tuple[int, ...],
    layers_synthesis: tuple[int, ...],
    layers_entropy: tuple[int, ...],
    context_size: int,
    num_grids: int,
    upsampling_type: str,
    upsampling_kwargs: Mapping[str, Any],
    downsampling_factor: float | tuple[float, ...],
    downsampling_exponents: None | tuple[int, ...],
    synthesis_num_residual_layers: int,
    synthesis_residual_kernel_shape: int,
    synthesis_per_frame_conv: bool,
    entropy_use_prev_grid: bool,
    entropy_prev_kernel_shape: None | tuple[int, ...],
    entropy_mask_config: config_dict.ConfigDict,
) -> dict[str, float]:
  """Compute MACs/pixel of C3-like model.

  Args:
    input_shape: Image or video size as ({T}, H, W, C).
    layers_synthesis: Hidden layers in synthesis model.
    layers_entropy: Hidden layers in entropy model.
    context_size: Size of context for entropy model.
    num_grids: Number of latent grids.
    upsampling_type: Method to use for upsampling.
    upsampling_kwargs: Keyword arguments for upsampling.
    downsampling_factor: Downsampling factor fo each grid of latents. This can
      be a float or a tuple of length equal to `len(input_shape) - 1`.
    downsampling_exponents: Determines how often each grid is downsampled. If
      provided, should be of length `num_grids`.
    synthesis_num_residual_layers: Number of residual conv layers in synthesis
      model.
    synthesis_residual_kernel_shape: Kernel shape for residual conv layers in
      synthesis model.
    synthesis_per_frame_conv: Whether to use per frame convolutions for video.
      Has no effect for images.
    entropy_use_prev_grid: Whether the previous grid is used as extra
      conditioning for the entropy model.
    entropy_prev_kernel_shape: Kernel shape used to determine how many latents
      of the previous grid are used to condition the entropy model. Only takes
      effect when entropy_use_prev_grid=True.
    entropy_mask_config: mask config for video. Has no effect for images.

  Returns:
    Dictionary with MACs/pixel of each part of model.
  """
  output_dict = {
      'interpolation': 0,
      'entropy_model': 0,
      'synthesis': 0,
      'total_no_interpolation': 0,
      'total': 0,
      'num_pixels': 0,
      'num_latents': 0,
  }

  is_video = len(input_shape) == 4

  # Compute image/video size statistics
  if is_video:
    num_frames, height, width, channels = input_shape
    num_pixels = num_frames * height * width
  else:
    num_frames = 1  # To satisfy linter
    height, width, channels = input_shape
    num_pixels = height * width
  output_dict['num_pixels'] = num_pixels

  # Compute total number of latents
  num_dims = len(input_shape) - 1
  # Convert downsampling_factor to a tuple if not already a tuple.
  if isinstance(downsampling_factor, (int, float)):
    df = (downsampling_factor,) * num_dims
  else:
    assert len(downsampling_factor) == num_dims
    df = downsampling_factor
  if downsampling_exponents is None:
    downsampling_exponents = range(num_grids)
  num_latents = 0
  for i in downsampling_exponents:
    if is_video:
      num_latents += (
          np.ceil(num_frames // (df[0]**i))
          * np.ceil(height // (df[1]**i))
          * np.ceil(width // (df[2]**i))
      )
    else:
      num_latents += (
          np.ceil(height // (df[0]**i))
          * np.ceil(width // (df[1]**i))
      )
  output_dict['num_latents'] = num_latents

  output_dict['interpolation'], upsampling_macs_pp = macs_per_pixel_upsampling(
      num_grids=num_grids,
      upsampling_type=upsampling_type,
      is_video=is_video,
      **upsampling_kwargs,
  )

  # Compute MACs for entropy model
  if is_video:
    mask_config = entropy_mask_config
    use_learned_mask = (
        mask_config.use_custom_masking and mask_config.learn_prev_frame_mask
    )
    if use_learned_mask:
      # The context size for the current latent frame is given by
      # the causal mask of height=width=current_frame_mask_size
      context_size_current = (mask_config.current_frame_mask_size**2 - 1) // 2
      # The context size for the previious latent frame is given by
      # lprev_frame_contiguous_mask_shape, with no masking (not causal).
      # Note this context is only used for latent grids with indices in
      # prev_frame_mask_grids.
      context_size_prev = np.prod(
          mask_config.prev_frame_contiguous_mask_shape
      )
    else:
      context_size_current = context_size_prev = 0  # dummy to satisfy linter.
    # Compute macs/pixel for each grid of entropy layer
    entropy_macs_pp = 0
    for grid_idx in range(num_grids):
      if use_learned_mask:
        if grid_idx in mask_config.prev_frame_mask_grids:
          context_size = context_size_current + context_size_prev
        else:
          context_size = context_size_current
      input_dims = (context_size,) + layers_entropy
      output_dims = layers_entropy + (2,)
      entropy_macs_pp_per_grid = sum(
          f_in * f_out for f_in, f_out in zip(input_dims, output_dims)
      )
      # The ratio of num_latents between grid 0 and grid {grid_idx} is approx
      # np.prod(df)**(-downsampling_exponents[grid_idx]), so weigh
      # entropy_macs_per_grid by this factor.
      factor = np.prod(df)**(-downsampling_exponents[grid_idx])
      entropy_macs_pp += int(entropy_macs_pp_per_grid * factor)
    output_dict['entropy_model'] = entropy_macs_pp
  else:
    # Update context size based on whether previous grid is used or not.
    if entropy_use_prev_grid:
      context_size += np.prod(entropy_prev_kernel_shape)
    # Input layer of entropy model corresponds to context size and output size
    # is 2 since we output both the location and scale of Laplace distribution.
    layers_entropy = (context_size, *layers_entropy, 2)
    macs_per_latent = mlp_macs_per_output_pixel(layers_entropy)
    # We perform a forward pass of entropy model for each latent, so total MACs
    # will be macs_per_latent * num_latents. Then divide to get MACs/pixel
    entropy_macs = macs_per_latent * num_latents
    if entropy_use_prev_grid:
      # Compute the cost of upsampling latent grids - almost every latent
      # entry has a value up/down-sampled to it from a neighbouring grid
      # (except for the first grid, although in practice we upsample zeros - see
      # `__call__` method of `AutoregressiveEntropyModelConvImage`. The below
      # would be an overestimate if we don't upsample for the first grid).
      entropy_macs += num_latents * upsampling_macs_pp
    output_dict['entropy_model'] = math.ceil(entropy_macs / num_pixels)

  # Compute MACs for synthesis model
  # Synthesis model takes as input concatenated latent grids and outputs a
  # single pixel
  layers_synthesis = (num_grids, *layers_synthesis, channels)
  synthesis_backbone_macs = mlp_macs_per_output_pixel(layers_synthesis)

  # Compute macs for residual conv layers in synthesis model. Note that we
  # only count multiplications for MACs, so ignore the addition operation for
  # the residual connection and the edge padding, though we include the extra
  # multiplications due to the edge padding.
  residual_layers = (channels,) * (synthesis_num_residual_layers + 1)
  synthesis_residual_macs = 0
  # We use Conv2D for the residual layers unless `is_video` and
  # `synthesis_per_frame_conv`.
  if is_video and not synthesis_per_frame_conv:
    ndim = 3
  else:
    ndim = 2
  for l, lp1 in zip(residual_layers[:-1], residual_layers[1:]):
    synthesis_residual_macs += conv_macs_per_output_pixel(
        synthesis_residual_kernel_shape, f_in=l, f_out=lp1, ndim=ndim,
    )
  output_dict['synthesis'] = synthesis_backbone_macs + synthesis_residual_macs

  # Compute total MACs/pixel without counting interpolation
  output_dict['total_no_interpolation'] = (
      output_dict['entropy_model'] + output_dict['synthesis']
  )
  # Compute total MACs/pixel
  output_dict['total'] = (
      output_dict['total_no_interpolation'] + output_dict['interpolation']
  )

  return output_dict
