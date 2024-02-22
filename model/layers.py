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

"""Custom layers used in different parts of the model."""

import functools
from typing import Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Array = chex.Array


def causal_mask(
    kernel_shape: tuple[int, int] | tuple[int, int, int], f_out: int
) -> Array:
  """Returns a causal mask in n dimensions w/ `kernel_shape` and out dim `f_out`.

  The mask will have output shape `({t}, h, w, 1, f_out)`,
  and the first `prod(kernel_shape) // 2` spatial components are `True` while
  all others are `False`. E.g. for `ndims = 2` and a 3x3 kernel the spatial
  components are given by
  ```
      [[1., 1., 1.],
       [1., 0., 0.],
       [0., 0., 0.]]
  ```
  for a 3x3 kernel.

  Args:
    kernel_shape: Size or shape of the kernel.
    f_out: Number of output features.

  Returns:
    Mask of shape ({t}, h, w, 1, f_out) where the spatio-temporal dimensions are
    given by `kernel_shape`.
  """

  for i, k in enumerate(kernel_shape):
    assert k % 2 == 1, (
        f'Kernel shape needs to be odd in all dimensions, not {k=} in'
        f' dimension {i}.'
    )
  num_kernel_entries = np.prod(kernel_shape)

  # Boolean array for spatial dimensions of the kernel.
  # All entries preceding the center are set to `True` (unmasked), while the
  # center and all subsequent entries are set to `False` (masked). See above for
  # an example.
  mask = jnp.arange(num_kernel_entries) < num_kernel_entries // 2
  # reshape according to `ndims` and add f_in and f_out dims.
  mask = jnp.reshape(mask, kernel_shape + (1, 1))
  # tile across the output dimension.
  mask = jnp.broadcast_to(mask, kernel_shape + (1, f_out))

  return mask


def central_mask(
    kernel_shape: tuple[int, int] | tuple[int, int, int],
    mask_shape: tuple[int, int] | tuple[int, int, int],
    f_out: int,
) -> Array:
  """Returns a mask with `kernel_shape` where the central `mask_shape` is 1."""

  for i, k in enumerate(kernel_shape):
    assert k % 2 == 1, (
        f'Kernel shape needs to be odd in all dimensions, not {k=} in'
        f' dimension {i}.'
    )
  for i, k in enumerate(mask_shape):
    assert k % 2 == 1, (
        f'Mask shape needs to be odd in all dimensions, not {k=} in'
        f' dimension {i}.'
    )
  assert len(kernel_shape) == len(mask_shape)
  for o, i in zip(kernel_shape, mask_shape):
    assert o >= i, f'Mask shape {i=} can be at most kernel shape {o=}.'

  # Compute border_sizes that will be `0` in each dimension
  border_sizes = tuple(
      int((o - i) / 2) for o, i in zip(kernel_shape, mask_shape)
  )

  mask = np.ones((mask_shape))  # Ones in the center.
  mask = np.pad(
      mask,
      ((border_sizes[0], border_sizes[0]), (border_sizes[1], border_sizes[1])),
  )  # Pad with zeros.
  mask = mask.astype(bool)

  # add `f_in = 1` and `f_out` dimensions
  mask = mask[..., None, None]
  mask = np.broadcast_to(mask, mask.shape[:-1] + (f_out,))

  return mask


def get_prev_current_mask(
    kernel_shape: tuple[int, int],
    prev_kernel_shape: tuple[int, int],
    f_out: int,
) -> Array:
  """Returns mask of size `kernel_shape + (2, f_out)`."""
  mask_current = causal_mask(kernel_shape=kernel_shape, f_out=f_out)
  mask_prev = central_mask(
      kernel_shape=kernel_shape, mask_shape=prev_kernel_shape, f_out=f_out
  )
  return jnp.concatenate([mask_prev, mask_current], axis=-2)


def init_like_linear(shape, dtype):
  """Initialize Conv kernel with single input dim like a Linear layer.

  We have to use this function instead of just calling the initializer with
  suitable scale directly as calling the initializer with a different number of
  elements leads to completely different values. I.e., the values in the smaller
  array are not a prefix of the values in the larger array.

  Args:
    shape: Shape of the weights.
    dtype: Data type of the weights.

  Returns:
    Weights of shape ({t}, h, w, 1, f_out)
  """
  *spatial_dims, f_in, f_out = shape
  spatial_dims = tuple(spatial_dims)
  assert f_in == 1, f'Input feature dimension needs to be 1 not {f_in}.'
  lin_f_in = np.prod(spatial_dims) // 2
  # Initialise weights using same initializer as `Linear` uses by default.
  weights = hk.initializers.TruncatedNormal(stddev=1 / jnp.sqrt(lin_f_in))(
      (lin_f_in, f_out), dtype=dtype
  )
  weights = jnp.concatenate(
      [weights, jnp.zeros((lin_f_in + 1, f_out), dtype=dtype)], axis=0
  )  # set masked weights to zero
  weights = weights.reshape(spatial_dims + (1, f_out))
  return weights


def init_like_conv3d(
    f_in: int,
    f_out: int,
    kernel_shape_3d: tuple[int, int, int],
    kernel_shape_current: tuple[int, int],
    kernel_shape_prev: tuple[int, int],
    prev_frame_mask_top_left: tuple[int, int],
    dtype: Any = jnp.float32,
) -> tuple[Array, Array, Array | None]:
  """Initialize Conv2D kernels in EfficientConv with same init as masked Conv3D.

  Args:
    f_in: Number of input channels.
    f_out: Number of output channels.
    kernel_shape_3d: Shape of masked Conv3D kernel whose initialization we would
      like to match for the EfficientConv.
    kernel_shape_current: Kernel shape for the Conv2D applied to current latent
      frame.
    kernel_shape_prev: Kernel shape for the Conv2D applied to previous latent
      frame.
    prev_frame_mask_top_left: The position of the top left entry of the
      rectangular mask applied to the previous latent frame context, relative to
      the top left entry of the previous latent frame context. e.g. a value of
      (1, 2) would mean that the mask starts 1 latent pixel below and 2 latent
      pixels to the right of the top left entry of the previous latent frame
      context. If this value is set to None, None is returned for `w_prev`.
      In EfficientConv, this corresponds to the case where the previous frame is
      masked out from the context used by the conv entropy model i.e. only the
      current latent frame is used by the entropy model.
    dtype: dtype for init values.

  Returns:
    w3d: weights of shape (*kernel_shape_3d, f_in, f_out)
    w_current: weights of shape (*kernel_shape_current, f_in, f_out)
    w_prev: weights of shape (*kernel_shape_prev, f_in, f_out)
  """
  t, *kernel_shape_3d_spatial = kernel_shape_3d
  kernel_shape_3d_spatial = tuple(kernel_shape_3d_spatial)
  assert t == 3, f'Time dimension has to have size 3 not {t}'
  assert kernel_shape_current[0] % 2 == 1
  assert kernel_shape_current[1] % 2 == 1
  assert kernel_shape_3d_spatial[0] % 2 == 1
  assert kernel_shape_3d_spatial[1] % 2 == 1
  # Initialize Conv3D weights
  kernel_shape_3d_full = kernel_shape_3d + (f_in, f_out)
  fan_in_shape = np.prod(kernel_shape_3d_full[:-1])
  stddev = 1.0 / np.sqrt(fan_in_shape)
  w3d = hk.initializers.TruncatedNormal(stddev=stddev)(
      kernel_shape_3d_full, dtype=dtype
  )
  # Both (kh, kw, f_in, f_out)
  w_prev, w_current, _ = w3d

  # Slice out weights for current frame (t=1)
  # ((kh - mask_h)//2, (kw - mask_w)//2). Note that all of kh, kw, mask_h,
  # mask_w are odd.
  slice_current = jax.tree_map(
      lambda x, y: (x - y) // 2, kernel_shape_3d_spatial, kernel_shape_current
  )  # symmetrically slice out from center
  w_current = w_current[
      slice_current[0] : -slice_current[0],
      slice_current[1] : -slice_current[1],
      ...,
  ]

  # Slice out weights for previous frame (t=0).
  # Start from indices defined by prev_frame_mask_top_left
  if prev_frame_mask_top_left is not None:
    mask_h, mask_w = kernel_shape_prev
    y_start, x_start = prev_frame_mask_top_left
    w_prev = w_prev[
        y_start : (y_start + mask_h), x_start : (x_start + mask_w), ...
    ]
  else:
    w_prev = None

  return w3d, w_current, w_prev


class EfficientConv(hk.Module):
  """Conv2D ops equivalent to a masked Conv3D for a particular setting.

  Notes:
    The equivalence holds when the Conv3D is a masked convolution of kernel
    shape (3, kh, kw) and f_in=1, where for the index 0 of the first axis
    (previous time step), a rectangular mask of shape `kernel_shape_prev` is
    used and for index 1 (current time step) a causal mask with
    `np.prod(kernel_shape_current)//2` dims are used.
  """

  def __init__(
      self,
      output_channels: int,
      kernel_shape_current: tuple[int, int],
      kernel_shape_prev: tuple[int, int],
      kernel_shape_conv3d: tuple[int, int, int],
      name: str | None = None,
  ):
    """Constructor.

    Args:
      output_channels: the number of output channels.
      kernel_shape_current: Shape of kernel for the current time step (index 1
        in first axis of mask).
      kernel_shape_prev: Shape of kernel for the prev time step (index 0 in
        first axis of mask).
      kernel_shape_conv3d: Shape of masked Conv3D to which EfficientConv is
        equivalent.
      name:
    """
    super().__init__(name=name)
    self.output_channels = output_channels
    self.kernel_shape_current = kernel_shape_current
    self.kernel_shape_prev = kernel_shape_prev
    self.kernel_shape_conv3d = kernel_shape_conv3d

  def __call__(
      self,
      x: Array,
      prev_frame_mask_top_left: tuple[int, int] | None,
      **unused_kwargs,
  ) -> Array:
    assert x.ndim == 4  # T, H, W, C
    inputs = jnp.concatenate([jnp.zeros_like(x[0:1]), x], axis=0)

    if hk.running_init():
      _, w_current, w_prev = init_like_conv3d(
          f_in=1,
          f_out=self.output_channels,
          kernel_shape_3d=self.kernel_shape_conv3d,
          kernel_shape_current=self.kernel_shape_current,
          kernel_shape_prev=self.kernel_shape_prev,
          prev_frame_mask_top_left=prev_frame_mask_top_left,
          dtype=jnp.float32,
      )
      w_init_current = lambda *_: w_current
      w_init_prev = lambda *_: w_prev
    else:
      # we do not pass rng at apply-time
      w_init_current, w_init_prev = None, None

    # conv_current needs to be causally masked, but conv_prev doesn't use casual
    # masking.
    mask = causal_mask(
        kernel_shape=self.kernel_shape_current, f_out=self.output_channels
    )
    self.conv_current = hk.Conv2D(
        self.output_channels,
        self.kernel_shape_current,
        mask=mask,
        w_init=w_init_current,
        name='conv_current_masked_layer',
    )
    if prev_frame_mask_top_left is not None:
      self.conv_prev = hk.Conv2D(
          self.output_channels,
          self.kernel_shape_prev,
          w_init=w_init_prev,
          name='conv_prev',
      )
    apply_fn = functools.partial(
        self._apply, prev_frame_mask_top_left=prev_frame_mask_top_left
    )
    # We vmap across time, looping over current and previous frames.
    return jax.vmap(apply_fn)(inputs[1:], inputs[:-1])

  def _pad_before_or_after(self, pad_size: int) -> tuple[int, int]:
    # Note that a positive `pad_size` pads to before, and negative `pad_size`
    # pads to after.
    if pad_size > 0:
      return (pad_size, 0)
    else:
      return (0, -pad_size)

  def _apply(
      self,
      x_current: Array,
      x_prev: Array,
      prev_frame_mask_top_left: tuple[int, int] | None,
  ) -> Array:
    # The masked Conv3D with custom masking is implemented here as two separate
    # Conv2D's `self.conv_current` and `self.conv_prev` (if
    # `prev_frame_mask_top_left` is not None) applied to the current frame and
    # previous frame respectively. The sum of their outputs are returned.
    # In the case where `self.conv_prev = None`, only the output of
    # `self.conv_current` is returned.
    assert x_current.ndim == 3
    assert x_prev.ndim == 3
    h, w, _ = x_current.shape

    # Apply convolution to current frame
    out_current = self.conv_current(x_current)
    if prev_frame_mask_top_left is not None:
      # Apply convolution to previous frame
      conv3d_h, conv3d_w = self.kernel_shape_conv3d[1:3]
      prev_kh, prev_kw = self.kernel_shape_prev
      pad_y = (conv3d_h - prev_kh + 1) // 2 - prev_frame_mask_top_left[0]
      pad_x = (conv3d_w - prev_kw + 1) // 2 - prev_frame_mask_top_left[1]
      x_prev = jnp.pad(
          x_prev,
          (
              self._pad_before_or_after(pad_y),
              self._pad_before_or_after(pad_x),
              (0, 0),
          ),
      )
      out_prev = self.conv_prev(x_prev)
      y_start = 0 if pad_y > 0 else out_prev.shape[0] - h
      x_start = 0 if pad_x > 0 else out_prev.shape[1] - w
      out_prev = out_prev[y_start : h + y_start, x_start : w + x_start, :]
      return out_current + out_prev
    else:
      return out_current
