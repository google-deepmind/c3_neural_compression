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

"""Synthesis networks mapping latents to images."""

from collections.abc import Callable
import functools
from typing import Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from c3_neural_compression.model import model_coding

Array = chex.Array


def b_init_custom_value(shape, dtype, value=None):
  """Initializer for biases that sets them to a particular value."""
  if value is None:
    return jnp.zeros(shape, dtype)
  else:
    chex.assert_shape(value, shape)
    assert jnp.dtype(value) == dtype
    return value


def edge_padding(
    x: Array,
    kernel_shape: int = 3,
    is_video: bool = False,
    per_frame_conv: bool = False,
) -> Array:
  """Replication/edge padding along the edges."""
  # Note that the correct padding is k/2 for even k and (k-1)/2 for odd k.
  # This can be achieved with k // 2.
  pad_len = kernel_shape // 2
  if is_video:
    assert x.ndim == 4
    if per_frame_conv:
      # When we apply convolution per frame, the time dimension is considered
      # as a batch dimension and no convolution is applied
      pad_width = (
          (0, 0),  # Time (no convolution, so no padding)
          (pad_len, pad_len),  # Height (convolution, so pad)
          (pad_len, pad_len),  # Width (convolution, so pad)
          (0, 0),  # Channels (no convolution, so no padding)
      )
    else:
      pad_width = (
          (pad_len, pad_len),  # Time (convolution, so pad)
          (pad_len, pad_len),  # Height (convolution, so pad)
          (pad_len, pad_len),  # Width (convolution, so pad)
          (0, 0),  # Channels (no convolution, so no padding)
      )
  else:
    assert x.ndim == 3
    pad_width = (
        (pad_len, pad_len),  # Height (convolution, so pad)
        (pad_len, pad_len),  # Width (convolution, so pad)
        (0, 0),  # Channels (no convolution, so no padding)
    )

  return jnp.pad(x, pad_width, mode='edge')


class ResidualWrapper(hk.Module):
  """Wrapper to make haiku modules residual, i.e., out = x + hk_module(x)."""

  def __init__(
      self,
      hk_module: Callable[..., Any],
      name: str | None = None,
      num_channels: int | None = None,
  ):
    super().__init__(name=name)
    self._hk_module = hk_module

  def __call__(self, x, *args, **kwargs):
    return x + self._hk_module(x, *args, **kwargs)


class Synthesis(hk.Module, model_coding.QuantizableMixin):
  """Synthesis network: an elementwise MLP implemented as a 1x1 conv."""

  def __init__(
      self,
      *,
      layers: tuple[int, ...] = (12, 12),
      out_channels: int = 3,
      kernel_shape: int = 1,
      num_residual_layers: int = 0,
      residual_kernel_shape: int = 3,
      activation_fn: str = 'gelu',
      add_activation_before_residual: bool = False,
      add_layer_norm: bool = False,
      clip_range: tuple[float, float] = (0.0, 1.0),
      is_video: bool = False,
      per_frame_conv: bool = False,
      b_last_init_value: Array | None = None,
      **unused_kwargs,
  ):
    """Constructor.

    Args:
      layers: Sequence of layer sizes. Length of tuple corresponds to depth of
        network.
      out_channels: Number of output channels.
      kernel_shape: Shape of convolutional kernel.
      num_residual_layers: Number of extra residual conv layers.
      residual_kernel_shape: Kernel shape of extra residual conv layers.
        If None, will default to out_channels. Only used when
        num_residual_layers > 0.
      activation_fn: Activation function.
      add_activation_before_residual: If True, adds a nonlinearity before the
        residual layers.
      add_layer_norm: Whether to add layer norm to the input.
      clip_range: Range at which outputs will be clipped. Defaults to [0, 1]
        which is useful for images and videos.
      is_video: If True, synthesizes a video, otherwise synthesizes an image.
      per_frame_conv: If True, applies 2D residual convolution layers *per*
        frame. If False, applies 3D residual convolutional layer directly to 3D
        volume of latents. Only used when is_video is True and
        num_residual_layers > 0.
      b_last_init_value: Optional. Array to be used as initial setting for the
        bias in the last layer (residual or non-residual) of the network. If
        `None`, it defaults to zero init (the default for all biases).
    """
    super().__init__()
    self._output_clip_range = clip_range
    activation_fn = getattr(jax.nn, activation_fn)

    b_last_init = lambda shape, dtype: b_init_custom_value(  # pylint: disable=g-long-lambda
        shape, dtype, b_last_init_value)

    # Initialize layers (equivalent to a pixelwise MLP if we use {1}x1x1 convs)
    net_layers = []
    if is_video:
      conv_cstr = hk.Conv3D
    else:
      conv_cstr = hk.Conv2D

    if add_layer_norm:
      net_layers += [
          hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ]
    for layer_size in layers:
      net_layers += [
          conv_cstr(layer_size, kernel_shape=kernel_shape),
          activation_fn,
      ]
    # If we are not using residual conv layers, the number of output channels
    # will be out_channels. Otherwise, the number of output channels will be
    # equal to the number of conv channels to be used in the subsequent residual
    # conv layers.
    net_layers += [
        conv_cstr(
            out_channels,
            kernel_shape=kernel_shape,
            b_init=None if num_residual_layers > 0 else b_last_init,
        )
    ]

    # Optionally add nonlinearity before residual layers
    if num_residual_layers > 0 and add_activation_before_residual:
      net_layers += [activation_fn]
    # Define core convolutional layer for each residual conv layer.
    if is_video and not per_frame_conv:
      conv_cstr = hk.Conv3D
    else:
      conv_cstr = hk.Conv2D
    for i in range(num_residual_layers):
      # We use padding='VALID' to be compatible with edge (replication) padding.
      # We also use zero init such that the residual conv is initially identity.
      # Use width=out_channels for every residual layer.
      is_last_layer = i == num_residual_layers - 1
      core_conv = conv_cstr(
          out_channels,
          kernel_shape=residual_kernel_shape,
          padding='VALID',
          w_init=jnp.zeros,
          b_init=b_last_init if is_last_layer else None,
          name='residual_conv',
      )
      net_layers += [
          ResidualWrapper(
              hk.Sequential([
                  # Add edge padding for well-behaved synthesis at boundaries.
                  functools.partial(
                      edge_padding,
                      kernel_shape=residual_kernel_shape,
                      is_video=is_video,
                      per_frame_conv=per_frame_conv,
                  ),
                  core_conv,
              ]),
          )
      ]
      # Add non-linearity for all but last layer
      if not is_last_layer:
        net_layers += [activation_fn]

    self._net = hk.Sequential(net_layers)

  def __call__(self, latents: Array) -> Array:
    """Maps latents to image or video.

    The input latents have shape ({T}, H, W, C) while the output image or video
    has shape ({T}, H, W, out_channels).

    Args:
      latents: Array of latents of shape ({T}, H, W, C).

    Returns:
      Predicted image or video of shape ({T}, H, W, out_channels).
    """
    return jnp.clip(
        self._net(latents),
        self._output_clip_range[0],
        self._output_clip_range[1],
    )
