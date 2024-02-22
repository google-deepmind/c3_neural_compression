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

"""Entropy models for quantized latents."""

from collections.abc import Callable
from typing import Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np

from c3_neural_compression.model import laplace
from c3_neural_compression.model import layers as layers_lib
from c3_neural_compression.model import model_coding


Array = chex.Array
init_like_linear = layers_lib.init_like_linear
causal_mask = layers_lib.causal_mask


def _clip_log_scale(
    log_scale: Array,
    scale_range: tuple[float, float],
    clip_like_cool_chic: bool = True,
) -> Array:
  """Clips log scale to lie in `scale_range`."""
  if clip_like_cool_chic:
    # This slightly odd clipping is based on the COOL-CHIC implementation
    # https://github.com/Orange-OpenSource/Cool-Chic/blob/16c41c033d6fd03e9f038d4f37d1ca330d5f7e35/src/models/arm.py#L158
    log_scale = -0.5 * jnp.clip(
        log_scale,
        -2 * jnp.log(scale_range[1]),
        -2 * jnp.log(scale_range[0]),
    )
  else:
    log_scale = jnp.clip(
        log_scale,
        jnp.log(scale_range[0]),
        jnp.log(scale_range[1]),
    )
  return log_scale


class AutoregressiveEntropyModelConvImage(
    hk.Module, model_coding.QuantizableMixin
):
  """Convolutional autoregressive entropy model for COOL-CHIC. Image only.

  This convolutional version is mathematically equivalent to its non-
  convolutional counterpart but also supports explicit batch dimensions.
  """

  def __init__(
      self,
      conditional_spec: config_dict.ConfigDict,
      layers: tuple[int, ...] = (12, 12),
      activation_fn: str = 'gelu',
      context_num_rows_cols: int | tuple[int, int] = 2,
      shift_log_scale: float = 0.0,
      scale_range: tuple[float, float] | None = None,
      clip_like_cool_chic: bool = True,
      use_linear_w_init: bool = True,
  ):
    """Constructor.

    Args:
      conditional_spec: Spec determining the type of conditioning to apply.
      layers: Sizes of layers in the conv-net. Length of tuple corresponds to
        depth of network.
      activation_fn: Activation function of conv net.
      context_num_rows_cols: Number of rows and columns to use as context for
        autoregressive prediction. Can be an integer, in which case the number
        of rows and columns is equal, or a tuple. The kernel size of the first
        convolution is given by `2*context_num_rows_cols + 1` (in each
        dimension).
      shift_log_scale: Shift the `log_scale` by this amount before it is clipped
        and exponentiated.
      scale_range: Allowed range for scale of Laplace distribution. For example,
        if scale_range = (1.0, 2.0), the scales are clipped to lie in [1.0,
        2.0]. If `None` no clipping is applied.
      clip_like_cool_chic: If True, clips scale in Laplace distribution in the
        same way as it's done in COOL-CHIC codebase. This involves clipping a
        transformed version of the log scale.
      use_linear_w_init: Whether to initialise the convolutions as if they were
        an MLP.
    """
    super().__init__()
    self._layers = layers
    self._activation_fn = getattr(jax.nn, activation_fn)
    self._scale_range = scale_range
    self._clip_like_cool_chic = clip_like_cool_chic
    self._conditional_spec = conditional_spec
    self._shift_log_scale = shift_log_scale
    self._use_linear_w_init = use_linear_w_init

    if isinstance(context_num_rows_cols, tuple):
      self.context_num_rows_cols = context_num_rows_cols
    else:
      self.context_num_rows_cols = (context_num_rows_cols,) * 2
    self.in_kernel_shape = tuple(2 * k + 1 for k in self.context_num_rows_cols)

    mask, w_init = self._get_first_layer_mask_and_init()

    net = []
    net += [hk.Conv2D(
        output_channels=layers[0],
        kernel_shape=self.in_kernel_shape,
        mask=mask,
        w_init=w_init,
        name='masked_layer_0',
    ),]
    for i, width in enumerate(layers[1:] + (2,)):
      net += [
          self._activation_fn,
          hk.Conv2D(
              output_channels=width,
              kernel_shape=1,
              name=f'layer_{i+1}',
          ),
      ]
    self.net = hk.Sequential(net)

  def _get_first_layer_mask_and_init(
      self,
  ) -> tuple[Array, Callable[[Any, Any], Array] | None]:
    """Returns the mask and weight initialization of the first layer."""
    if self._conditional_spec.use_conditioning:
      if self._conditional_spec.use_prev_grid:
        mask = layers_lib.get_prev_current_mask(
            kernel_shape=self.in_kernel_shape,
            prev_kernel_shape=self._conditional_spec.prev_kernel_shape,
            f_out=self._layers[0],
        )
        w_init = None
      else:
        raise ValueError('Only use_prev_grid conditioning supported.')
    else:
      mask = causal_mask(
          kernel_shape=self.in_kernel_shape, f_out=self._layers[0]
      )
      w_init = init_like_linear if self._use_linear_w_init else None
    return mask, w_init

  def _get_mask(
      self,
      dictkey: tuple[jax.tree_util.DictKey, ...],  # From `tree_map_with_path`.
      array: Array,
  ) -> Array:
    assert isinstance(dictkey[0].key, str)
    if 'masked_layer_0/w' in dictkey[0].key:
      mask, _ = self._get_first_layer_mask_and_init()
      return mask
    else:
      return np.ones(shape=array.shape, dtype=bool)

  def __call__(self, latent_grids: tuple[Array, ...]) -> tuple[Array, Array]:
    """Maps latent grids to parameters of Laplace distribution for every latent.

    Args:
      latent_grids: Tuple of all latent grids of shape (H, W), (H/2, W/2), etc.

    Returns:
      Tuple of parameters of Laplace distribution (loc and scale) each of shape
      (num_latents,).
    """

    assert len(latent_grids[0].shape) in (2, 3)

    if len(latent_grids[0].shape) == 3:
      bs = latent_grids[0].shape[0]
    else:
      bs = None

    if self._conditional_spec.use_conditioning:
      if self._conditional_spec.use_prev_grid:
        grids_cond = (jnp.zeros_like(latent_grids[0]),) + latent_grids[:-1]
        dist_params = []
        for prev_grid, grid in zip(grids_cond, latent_grids):
          # Resize `prev_grid` to have the same resolution as the current grid
          prev_grid_resized = jax.image.resize(
              prev_grid,
              shape=grid.shape,
              method=self._conditional_spec.interpolation
          )
          inputs = jnp.stack(
              [prev_grid_resized, grid], axis=-1
          )  # (h[k], w[k], 2)
          out = self.net(inputs)
          dist_params.append(out)
      else:
        raise ValueError('use_prev_grid is False')
    else:
      # If not using conditioning, just apply the same network to each latent
      # grid. Each element of dist_params has shape (h[k], w[k], 2).
      dist_params = [self.net(g[..., None]) for g in latent_grids]

    if bs is not None:
      dist_params = [p.reshape(bs, -1, 2) for p in dist_params]
      dist_params = jnp.concatenate(dist_params, axis=1)  # (bs, num_latents, 2)
    else:
      dist_params = [p.reshape(-1, 2) for p in dist_params]
      dist_params = jnp.concatenate(dist_params, axis=0)  # (num_latents, 2)

    assert dist_params.shape[-1] == 2
    loc, log_scale = dist_params[..., 0], dist_params[..., 1]

    log_scale = log_scale + self._shift_log_scale

    # Optionally clip log scale (we clip scale in log space to avoid overflow).
    if self._scale_range is not None:
      log_scale = _clip_log_scale(
          log_scale, self._scale_range, self._clip_like_cool_chic
      )
    # Convert log scale to scale (which ensures scale is positive)
    scale = jnp.exp(log_scale)
    return loc, scale


class AutoregressiveEntropyModelConvVideo(
    hk.Module, model_coding.QuantizableMixin
):
  """Convolutional autoregressive entropy model for COOL-CHIC on video.

  This convolutional version is mathematically equivalent to its non-
  convolutional counterpart but also supports explicit batch dimensions.
  """

  def __init__(
      self,
      num_grids: int,
      conditional_spec: config_dict.ConfigDict,
      mask_config: config_dict.ConfigDict,
      layers: tuple[int, ...] = (12, 12),
      activation_fn: str = 'gelu',
      context_num_rows_cols: int | tuple[int, ...] = 2,
      shift_log_scale: float = 0.0,
      scale_range: tuple[float, float] | None = None,
      clip_like_cool_chic: bool = True,
      use_linear_w_init: bool = True,
  ):
    """Constructor.

    Args:
      num_grids: Number of latent grids.
      conditional_spec: Spec determining the type of conditioning to apply.
      mask_config: mask_config used for entropy model.
      layers: Sizes of layers in the conv-net. Length of tuple corresponds to
        depth of network.
      activation_fn: Activation function of conv net.
      context_num_rows_cols: Number of rows and columns to use as context for
        autoregressive prediction. Can be an integer, in which case the number
        of rows and columns is equal, or a tuple. The kernel size of the first
        convolution is given by `2*context_num_rows_cols + 1` (in each
        dimension).
      shift_log_scale: Shift the `log_scale` by this amount before it is clipped
        and exponentiated.
      scale_range: Allowed range for scale of Laplace distribution. For example,
        if scale_range = (1.0, 2.0), the scales are clipped to lie in [1.0,
        2.0]. If `None` no clipping is applied.
      clip_like_cool_chic: If True, clips scale in Laplace distribution in the
        same way as it's done in COOL-CHIC codebase. This involves clipping a
        transformed version of the log scale.
      use_linear_w_init: Whether to initialise the convolutions as if they were
        an MLP.
    """
    super().__init__()
    # Need at least two layers so that we have at least one intermediate layer.
    # This is mainly to make the code simpler.
    assert len(layers) > 1, 'Need to have at least two layers.'
    self._activation_fn = getattr(jax.nn, activation_fn)
    self._scale_range = scale_range
    self._clip_like_cool_chic = clip_like_cool_chic
    self._num_grids = num_grids
    self._conditional_spec = conditional_spec
    self._mask_config = mask_config
    self._layers = layers
    self._shift_log_scale = shift_log_scale
    self._ndims = 3  # Video model.

    if isinstance(context_num_rows_cols, tuple):
      assert len(context_num_rows_cols) == self._ndims
      self.context_num_rows_cols = context_num_rows_cols
    else:
      self.context_num_rows_cols = (context_num_rows_cols,) * self._ndims
    self.in_kernel_shape = tuple(2 * k + 1 for k in self.context_num_rows_cols)

    mask = causal_mask(kernel_shape=self.in_kernel_shape, f_out=layers[0])

    def first_layer(prefix):
      # When using learned contiguous custom mask, use the more efficient
      # alternative of masked 3D conv, which sums up two 2D convs, one for
      # the current frame and one for the previous frame.
      if mask_config.use_custom_masking:
        assert self._conditional_spec.type == 'per_grid'
        current_frame_kw = mask_config.current_frame_mask_size
        assert current_frame_kw % 2 == 1
        current_frame_ks = (current_frame_kw, current_frame_kw)
        prev_frame_ks = mask_config.prev_frame_contiguous_mask_shape
        first_conv = layers_lib.EfficientConv(
            output_channels=layers[0],
            kernel_shape_current=current_frame_ks,
            kernel_shape_prev=prev_frame_ks,
            kernel_shape_conv3d=self.in_kernel_shape,
            name=f'{prefix}layer_0',
        )
      else:
        first_conv = hk.Conv3D(
            output_channels=layers[0],
            kernel_shape=self.in_kernel_shape,
            mask=mask,
            w_init=init_like_linear if use_linear_w_init else None,
            name=f'{prefix}masked_layer_0',
        )
      return first_conv

    def intermediate_layer(prefix, width, layer_idx):
      intermediate_conv = hk.Conv3D(
          output_channels=width,
          kernel_shape=1,
          name=f'{prefix}layer_{layer_idx+1}',
      )
      return intermediate_conv

    def final_layer(prefix):
      final_conv = hk.Conv3D(
          output_channels=2,
          kernel_shape=1,
          name=f'{prefix}layer_{len(layers)}',
      )
      return final_conv

    def return_net(
        grid_idx=None,
        frame_idx=None,
    ):
      prefix = ''
      if grid_idx is not None:
        prefix += f'grid_{grid_idx}_'
      if frame_idx is not None:
        prefix += f'frame_{frame_idx}_'
      net = [first_layer(prefix)]
      for layer_idx, width in enumerate(layers[1:]):
        net += [self._activation_fn,
                intermediate_layer(prefix, width, layer_idx)]
      net += [self._activation_fn, final_layer(prefix)]
      return hk.Sequential(net)

    if self._conditional_spec and self._conditional_spec.use_conditioning:
      assert self._conditional_spec.type == 'per_grid'
      self.nets = [return_net(grid_idx=i) for i in range(self._num_grids)]
    else:
      self.net = return_net()

  def _get_mask(
      self,
      dictkey: tuple[jax.tree_util.DictKey, ...],  # From `tree_map_with_path`.
      array: Array,
  ) -> Array:
    assert isinstance(dictkey[0].key, str)
    # For the case use_custom_masking=False, we have Conv3D kernels with causal
    # masking (for both use_conditioning = True/False).
    if 'masked_layer_0/w' in dictkey[0].key:
      return causal_mask(
          kernel_shape=self.in_kernel_shape, f_out=self._layers[0]
      )
    # For the case use_custom_masking=True, we have Conv2D kernels for the
    # current and previous latent frame. Only the kernels for the current latent
    # frame is causally masked.
    elif 'conv_current_masked_layer/w' in dictkey[0].key:
      kw = kh = self._mask_config.current_frame_mask_size
      mask = causal_mask(
          kernel_shape=(kh, kw), f_out=self._layers[0]
      )
      return mask
    else:
      return np.ones(shape=array.shape, dtype=bool)

  def __call__(
      self,
      latent_grids: tuple[Array, ...],
      prev_frame_mask_top_lefts: (
          tuple[tuple[int, int] | None, ...] | None
      ) = None,
  ) -> tuple[Array, Array]:
    """Maps latent grids to parameters of Laplace distribution for every latent.

    Args:
      latent_grids: Tuple of all latent grids of shape (T, H, W), (T/2, H/2,
        W/2), etc.
      prev_frame_mask_top_lefts: Tuple of prev_frame_mask_top_left values, to be
        used when there are layers using EfficientConv. Each element is either:
        1) an index (y_start, x_start) indicating the position of
        the rectangular mask for the previous latent frame context of each grid.
        2) None, indicating that the previous latent frame is masked out of the
        context for that particular grid.
        Note that if `prev_frame_mask_top_lefts` is not None, then it's a tuple
        of length `num_grids` (same length as `latent_grids`). This is only
        used when mask_config.use_custom_masking=True.

    Returns:
      Parameters of Laplace distribution (loc and scale) each of shape
      (num_latents,).
    """

    assert len(latent_grids) == self._num_grids

    if len(latent_grids[0].shape) == 4:
      bs = latent_grids[0].shape[0]
    else:
      bs = None

    if self._conditional_spec and self._conditional_spec.use_conditioning:
      # Note that for 'per_grid' conditioning, params have names:
      # f'autoregressive_entropy_model_conv/~/grid_{grid_idx}_frame_{frame_idx}'
      # f'_layer_{layer_idx}/w' or f'_layer_{layer_idx}/b'.
      # We need one call to self.net per grid.
      assert self._conditional_spec.type == 'per_grid'
      dist_params = []
      for k, g in enumerate(latent_grids):
        # g has shape (t[k], h[h], w[k])
        # dp computed below will have shape (t[k], h[h], w[k], 2)
        if self._mask_config.use_custom_masking:
          dp = self.nets[k](
              g[..., None],
              prev_frame_mask_top_left=prev_frame_mask_top_lefts[k],
          )
        else:
          dp = self.nets[k](g[..., None])
        dist_params.append(dp)
    else:
      # If not using conditioning, just apply the same network to each latent
      # grid. Each element of dist_params has shape ({t[k]}, h[k], w[k], 2).
      dist_params = [self.net(g[..., None]) for g in latent_grids]

    if bs is not None:
      dist_params = [p.reshape(bs, -1, 2) for p in dist_params]
      dist_params = jnp.concatenate(dist_params, axis=1)  # (bs, num_latents, 2)
    else:
      dist_params = [p.reshape(-1, 2) for p in dist_params]
      dist_params = jnp.concatenate(dist_params, axis=0)  # (num_latents, 2)

    assert dist_params.shape[-1] == 2
    loc, log_scale = dist_params[..., 0], dist_params[..., 1]

    log_scale = log_scale + self._shift_log_scale

    # Optionally clip log scale (we clip scale in log space to avoid overflow).
    if self._scale_range is not None:
      log_scale = _clip_log_scale(
          log_scale, self._scale_range, self._clip_like_cool_chic
      )
    # Convert log scale to scale (which ensures scale is positive)
    scale = jnp.exp(log_scale)
    return loc, scale


def compute_rate(
    x: Array, loc: Array, scale: Array, q_step: float = 1.0
) -> Array:
  """Compute entropy of x (in bits) under the Laplace(mu, scale) distribution.

  Args:
    x: Array of shape (batch_size,) containing points whose entropy will be
      evaluated.
    loc: Array of shape (batch_size,) containing the location (mu) parameter of
      Laplace distribution.
    scale: Array of shape (batch_size,) containing scale parameter of Laplace
      distribution.
    q_step: Step size that was used for quantizing the input x (see
      latents.Latents for details). This implies that, when x is quantized with
      `round` or `ste`, the values of x / q_step should be integer-valued
      floats.

  Returns:
    Rate (entropy) of x under model as array of shape (num_latents,).
  """
  # Ensure that the rate is computed in space where the bin width of the array x
  # is 1. Also ensure that the loc and scale of the Laplace distribution are
  # appropriately scaled.
  x /= q_step
  loc /= q_step
  scale /= q_step

  # Compute probability of x by integrating pdf from x - 0.5 to x + 0.5. Note
  # that as x is not necessarily an integer (when using noise quantization) we
  # cannot use distrax.quantized.Quantized, which requires the input to be an
  # integer.
  # However, when x is an integer, the behaviour of the below lines of code
  # are equivalent to distrax.quantized.Quantized.
  dist = laplace.Laplace(loc, scale)
  log_probs = dist.integrated_log_prob(x)

  # Change base of logarithm
  rate = - log_probs / jnp.log(2.)

  # No value can cost more than 16 bits (based on COOL-CHIC implementation)
  rate = jnp.clip(rate, a_max=16)

  return rate


def flatten_latent_grids(latent_grids: tuple[Array, ...]) -> Array:
  """Flattens list of latent grids into a single array.

  Args:
    latent_grids: List of all latent grids of shape ({T}, H, W), ({T/2}, H/2,
      W/2), ({T/4}, H/4, W/4), etc.

  Returns:
    Array of shape (num_latents,) containing all flattened latent grids
    stacked into a single array.
  """
  # Reshape each latent grid from ({t}, h, w) to ({t} * h * w,)
  all_latents = [latent_grid.reshape(-1) for latent_grid in latent_grids]
  # Stack into a single array of size (num_latents,)
  return jnp.concatenate(all_latents)


def unflatten_latent_grids(
    flattened_latents: Array,
    latent_grid_shapes: tuple[Array, ...]
) -> tuple[Array, ...]:
  """Unflattens a single flattened latent grid into a list of latent grids.

  Args:
    flattened_latents: Flattened latent grids (a 1D array).
    latent_grid_shapes: List of shapes of latent grids.

  Returns:
    List of all latent grids of shape ({T}, H, W), ({T/2}, H/2, W/2),
    ({T/4}, H/4, W/4), etc.
  """

  assert sum([np.prod(s) for s in latent_grid_shapes]) == len(flattened_latents)

  latent_grids = []

  for shape in latent_grid_shapes:
    size = np.prod(shape)
    latent_grids.append(flattened_latents[:size].reshape(shape))
    flattened_latents = flattened_latents[size:]

  return tuple(latent_grids)
