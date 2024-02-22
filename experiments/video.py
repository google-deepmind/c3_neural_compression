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

"""Jaxline experiment for C3 video experiments. Assume single-device."""

import collections
from collections.abc import Mapping
import copy
import functools
import textwrap
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import chex
import dm_pix
import haiku as hk
import immutabledict
import jax
import jax.numpy as jnp
from jaxline import platform
import numpy as np
import optax
import scipy

from c3_neural_compression.experiments import base
from c3_neural_compression.model import entropy_models
from c3_neural_compression.utils import experiment as experiment_utils
from c3_neural_compression.utils import psnr as psnr_utils


Array = chex.Array

FLAGS = flags.FLAGS


class Experiment(base.Experiment):
  """Per data-point compression experiment. Assume single-device."""

  def init_params(self, input_res, soft_round_temp=None, input_mean=None,
                  learn_mask=False, prev_frame_mask_top_lefts=None):
    forward_init = jax.jit(
        self.forward.init,
        static_argnames=('quant_type', 'input_res', 'learn_mask',
                         'prev_frame_mask_top_lefts')
    )
    params = forward_init(
        self.init_rng,
        quant_type=self.config.quant.noise_quant_type,
        input_res=tuple(input_res),
        soft_round_temp=soft_round_temp,
        input_mean=input_mean,
        learn_mask=learn_mask,
        prev_frame_mask_top_lefts=prev_frame_mask_top_lefts,
    )
    opt = self.get_opt(use_cosine_schedule=True)
    opt_state = jax.jit(opt.init)(params)
    experiment_utils.log_params_info(params)
    return params, opt_state

  def _get_upsampling_fn(self, input_res):
    num_dims = len(input_res)
    assert num_dims == 3
    return super()._get_upsampling_fn(input_res)

  def _get_entropy_model(self, learn_mask=False):
    """Returns entropy model."""
    if learn_mask:
      # Use 'per_grid' entropy models and masked Conv3D to learn the entropy
      # model masking for the first few iterations.
      entropy_config = copy.deepcopy(self.config.model.entropy)
      assert entropy_config.conditional_spec.use_conditioning
      assert entropy_config.conditional_spec.type == 'per_grid'
      assert entropy_config.mask_config.use_custom_masking
      entropy_config.mask_config.prev_frame_contiguous_mask_shape = (
          2 * entropy_config.context_num_rows_cols[1] + 1,
          2 * entropy_config.context_num_rows_cols[2] + 1,
      )
    else:
      entropy_config = self.config.model.entropy
    return entropy_models.AutoregressiveEntropyModelConvVideo(
        num_grids=self.config.model.latents.num_grids,
        **entropy_config,
    )

  def _get_entropy_params(self, latent_grids, learn_mask=False,
                          prev_frame_mask_top_lefts=None):
    entropy_model = self._get_entropy_model(learn_mask=learn_mask)
    return entropy_model(latent_grids, prev_frame_mask_top_lefts)

  def _forward_fn(
      self,
      quant_type,
      input_res,
      soft_round_temp=None,
      input_mean=None,
      kumaraswamy_a=None,
      learn_mask=False,
      prev_frame_mask_top_lefts=None,
  ):
    """Forward pass.

    Args:
      quant_type: Quantization type changes during training, so must be passed
        as argument to function.
      input_res: Shape of input without channels.
      soft_round_temp: Smoothness of soft-rounding.
      input_mean: Optionally the mean RGB values of the input is used to
        initialise the last bias of the synthesis network. The mean is across
        the spatial dimensions. I.e., the shape of the input mean is (3,).
      kumaraswamy_a: `a` parameter of Kumaraswamy noise distribution.
      learn_mask: Whether to learn mask or not. If True, uses the `per_grid`
        conditioning for the convolutional entropy model.
      prev_frame_mask_top_lefts: Tuple of prev_frame_mask_top_left values, to be
        used when there are layers using EfficientConv. Each element is either:
        1) an index (y_start, x_start) indicating the position of
        the rectangular mask for the previous latent frame context of each grid.
        2) None, indicating that the previous latent frame is masked out of the
        context for that particular grid.
        Note that if `prev_frame_mask_top_lefts` is not None, then it's a tuple
        of length num_grids (same length as `latent_grids`). This is only
        used when mask_config.use_custom_masking=True.

    Returns:
      input_rec: Predicted video of shape (T, H, W, out_channels).
      grids: Latent grids as tuple of length num_grids containing grids of shape
      (T, H, W), (T/2, H/2, W/2), etc.
      all_latents: Array of shape (num_latents,) containing all latent variables
        from each grid, flattened into a single vector.
      loc: Array of shape (num_latents,) containing location parameter of
        Laplace distribution for each latent variable.
      scale: Array of shape (num_latents,) containing scale parameter of
        Laplace distribution for each latent variable.
    """
    latent_grids = self._get_latents(
        quant_type, input_res, soft_round_temp, kumaraswamy_a
    )

    # Compute parameters of autoregressive Laplace distribution, both of shape
    # shape (num_latents,) where
    # num_latents = sum([np.prod(grid.shape) for grid in grids]
    loc, scale = self._get_entropy_params(
        latent_grids, learn_mask, prev_frame_mask_top_lefts
    )

    # Flatten latent grids into a single latent vector of shape (num_latents,)
    # containing all latents
    all_latents = entropy_models.flatten_latent_grids(latent_grids)

    upsampled_latents = self._upsample_latents(latent_grids, input_res)
    # Synthesize video of shape (T, H, W, out_channels) from latent volume
    input_rec = self._synthesize(
        upsampled_latents, b_last_init_input_mean=input_mean, is_video=True
    )

    return input_rec, latent_grids, all_latents, loc, scale

  def _loss_fn(
      self,
      params,
      target,
      rng,
      quant_type,
      soft_round_temp,
      rd_weight,
      kumaraswamy_a,
      learn_mask=False,
      prev_frame_mask_top_lefts=None,
  ):
    """Rate distortion loss: distortion + lambda * rate.

    Args:
      params: Haiku parameters.
      target: Target video. Array of shape (T, H, W, 3).
      rng:
      quant_type: Type of quantization to use.
      soft_round_temp: Smoothness of soft-rounding.
      rd_weight: rd_weight used to weigh the bpp term in the loss.
      kumaraswamy_a: `a` parameter of Kumaraswamy noise distribution.
      learn_mask: Whether to learn mask or not. If True, uses the `per_grid`
        conditioning for the convolutional entropy model.
      prev_frame_mask_top_lefts: Tuple of prev_frame_mask_top_left values, to be
        used when there are layers using EfficientConv. Each element is either:
        1) an index (y_start, x_start) indicating the position of
        the rectangular mask for the previous latent frame context of each grid.
        2) None, indicating that the previous latent frame is masked out of the
        context for that particular grid.
        Note that if `prev_frame_mask_top_lefts` is not None, then it's a tuple
        of length num_grids (same length as `latent_grids`). This is only
        used when mask_config.use_custom_masking=True.

    Returns:
      Loss as jax array and dictionary of metrics.
    """

    out = self.forward.apply(
        params=params,
        rng=rng,
        quant_type=quant_type,
        input_res=target.shape[:-1],
        soft_round_temp=soft_round_temp,
        kumaraswamy_a=kumaraswamy_a,
        learn_mask=learn_mask,
        prev_frame_mask_top_lefts=prev_frame_mask_top_lefts,
    )
    pred_img, _, all_latents, loc, scale = out
    distortion = psnr_utils.mse_fn(pred_img, target)
    # Also report distortion for rounded rec, only for logging purposes.
    # Note that `jnp.round` rounds 0.5 to 0. but n + 0.5 to n + 1 for n >= 1.
    # The difference with standard rounding is only for 0.5 (set of measure 0)
    # so shouldn't matter in practice.
    pred_img_rounded = jnp.round(pred_img * 255.) / 255.
    distortion_rounded = psnr_utils.mse_fn(pred_img_rounded, target)
    # Sum rate over all pixels. Ensure that the rate is computed in the space
    # where the bin width of the latents is 1 by passing q_step to the rate
    # function.
    rate = entropy_models.compute_rate(
        all_latents, loc, scale, q_step=self.config.model.latents.q_step
    ).sum()
    num_pixels = self._num_pixels(target.shape[:-1])  # without channel dim
    loss = distortion + rd_weight * rate / num_pixels
    metrics = {
        'loss': loss,
        'distortion': distortion,
        'distortion_rounded': distortion_rounded,
        'rate': rate,
        'psnr': psnr_utils.psnr_fn(distortion),
        'psnr_rounded': psnr_utils.psnr_fn(distortion_rounded),
    }
    return loss, metrics

  # We use jit instead of pmap for simplicity, and assume that the experiment
  # runs on single-device (although the same code will also run on multi-device,
  # using only one of the devices). Note that jit gives faster runtime than pmap
  # on GPU, whereas pmap is faster than jit for TPU.
  @functools.partial(jax.jit, static_argnums=(0, 5, 8, 11, 12))
  def single_train_step(
      self,  # static
      params,
      opt_state,
      inputs,
      rng,
      quant_type,  # static
      rd_weight,
      soft_round_temp=None,
      use_cosine_schedule=True,  # static
      learning_rate=None,
      kumaraswamy_a=None,
      learn_mask=False,  # static
      prev_frame_mask_top_lefts=None,  # static
  ):
    logging.info('`single_train_step` was recompiled.')
    grads, metrics = jax.grad(self._loss_fn, has_aux=True)(
        params,
        target=inputs,
        rng=rng,
        quant_type=quant_type,
        soft_round_temp=soft_round_temp,
        rd_weight=rd_weight,
        kumaraswamy_a=kumaraswamy_a,
        learn_mask=learn_mask,
        prev_frame_mask_top_lefts=prev_frame_mask_top_lefts,
    )
    # Compute updates and update parameters.
    opt = self.get_opt(
        use_cosine_schedule=use_cosine_schedule,
        learning_rate=learning_rate,
    )
    # Optionally record gradient norms for each set of params.
    if self.config.log_gradient_norms:
      rest_grads, latent_grads = (
          experiment_utils.partition_params_by_module_name(grads, key='latent')
      )
      # Note that grads for upsampling parameters will be included in
      # synthesis_grads if present.
      synthesis_grads, entropy_grads = (
          experiment_utils.partition_params_by_module_name(
              rest_grads, key='autoregressive_entropy_model'
          )
      )
      latent_grad_norm = optax.global_norm(latent_grads)
      synthesis_grad_norm = optax.global_norm(synthesis_grads)
      entropy_grad_norm = optax.global_norm(entropy_grads)
      metrics = metrics | {'latent_grad_norm': latent_grad_norm,
                           'synthesis_grad_norm': synthesis_grad_norm,
                           'entropy_grad_norm': entropy_grad_norm}
    # Pass `params` for weight decay.
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics

  @functools.partial(jax.jit, static_argnums=(0, 3))
  def eval(self, params, inputs, prev_frame_mask_top_lefts):
    # We always use rounding for quantization at test time. Note that we call
    # the function with "ste" as this is equivalent to "round" at test time.
    out = self.forward.apply(
        params=params, rng=None, quant_type='ste', input_res=inputs.shape[:-1],
        prev_frame_mask_top_lefts=prev_frame_mask_top_lefts
    )
    rec, _, all_latents, loc, scale = out
    # Round rec to integer pixel values for distortion computation.
    # Note that `jnp.round` rounds 0.5 to 0. but n + 0.5 to n + 1 for n >= 1.
    # The difference with standard rounding is only for 0.5 (set of measure 0)
    # so shouldn't matter in practice.
    rec = jnp.round(rec * 255.) / 255.
    # Compute MSE loss
    distortion = psnr_utils.mse_fn(rec, inputs)
    # Sum rate over all pixels. Ensure that the rate is computed in the space
    # where the bin width of the latents is 1 by passing q_step to the rate
    # function.
    rate = entropy_models.compute_rate(
        all_latents, loc, scale, q_step=self.config.model.latents.q_step
    ).sum()
    # Compute rate distortion loss
    num_pixels = self._num_pixels(inputs.shape[:-1])  # without channel dim
    loss = distortion + self.config.loss.rd_weight * rate / num_pixels
    metrics = {
        'loss': loss,
        'distortion': distortion,
        'rate': rate,
        'psnr': psnr_utils.psnr_fn(distortion),
        'ssim': dm_pix.ssim(rec, inputs),
    }
    # For video, also log per frame distortion to calculate per frame psnr in
    # later stages of the eval.
    # Shape (T,) for input shape (T, H, W, C).
    per_frame_distortion = jnp.mean(
        (rec - inputs)**2, axis=tuple(range(1, inputs.ndim))
    )
    metrics['per_frame_distortion'] = per_frame_distortion
    return metrics

  def _log_train_metrics(
      self,
      i,
      metrics,
      delta_time,
      num_pixels,
      learn_mask=False,
      **other_metrics,
  ):
    if learn_mask:
      num_steps_total_str = (
          self.config.model.entropy.mask_config.learn_prev_frame_mask_iter
      )
    else:
      num_steps_total_str = (
          f'{self.config.opt.num_noise_steps}+'
          f'{self.config.opt.max_num_ste_steps}'
      )
    logging_message = (
        f'{i}/{num_steps_total_str},'
        f'loss={metrics["loss"]:.6e},'
        f' psnr={metrics["psnr"]:.3f},'
        f' bpp={(metrics["rate"] / num_pixels):.4f},'
    )
    # The below is for rd_weight and soft_round_temp.
    for k, v in other_metrics.items():
      logging_message += f' {k}={v:.6e},'
    logging_message += f' time={(delta_time):.4f}.'
    logging_message = textwrap.fill(logging_message, 80)
    logging.info(logging_message)

  def create_custom_mask(
      self,
      inputs,
      input_mean,
      rng,
      soft_round_temp_fn,
      rd_weight_fn,
      kumaraswamy_a_fn,
      quant_type,
  ):
    start = time.time()
    # Compute optional mask for entropy model from
    # self.config.model.entropy.mask_config.
    mask_config = self.config.model.entropy.mask_config
    context_num_rows_cols = self.config.model.entropy.context_num_rows_cols
    if mask_config is not None and mask_config.use_custom_masking:
      if isinstance(context_num_rows_cols, tuple):
        assert len(context_num_rows_cols) == 3
      else:
        context_num_rows_cols = (context_num_rows_cols,) * 3
      assert (
          context_num_rows_cols[0] == 1
      ), 'Custom masking only implemented for single previous latent frame'
      # Can have different masks for each grid index
      prev_frame_mask_top_lefts = []
      contiguous_mask_shape = mask_config.prev_frame_contiguous_mask_shape
      for grid_idx in range(self.config.model.latents.num_grids):
        if grid_idx in mask_config.prev_frame_mask_grids:
          prev_frame_mask_top_left = mask_config.prev_frame_mask_top_lefts[
              grid_idx
          ]
          prev_frame_mask_bottom_right = (
              prev_frame_mask_top_left[0] + contiguous_mask_shape[0] - 1,
              prev_frame_mask_top_left[1] + contiguous_mask_shape[1] - 1,
          )
          logging.info(
              'prev_frame_mask_top_left for grid %d: (%d, %d)',
              grid_idx,
              prev_frame_mask_top_left[0],
              prev_frame_mask_top_left[1]
          )
        else:
          prev_frame_mask_top_left = None
          prev_frame_mask_bottom_right = None
          logging.info(
              'prev_frame_mask_top_left for grid %d is None', grid_idx
          )
        prev_frame_mask_top_lefts.append(prev_frame_mask_top_left)
        logging.info(
            'prev_frame_mask_top_left, bottom_right for grid %d: %s, %s',
            grid_idx, prev_frame_mask_top_left, prev_frame_mask_bottom_right
        )
    else:
      prev_frame_mask_top_lefts = None

    # If mask_config.use_custom_masking and mask_config.learn_prev_frame_mask
    # are both True, learn the entropy model's mask for the previous frame
    # context by:
    # 1. Train the model with 'per_grid' conditioning and no prev frame masking
    #    for mask_config.learn_prev_frame_mask_iter iterations. Note that we
    #    can use current frame masking for this phase, if we will train with
    #    current frame masking for the subsequent phase.
    # 2. For each grid idx, take the first layer's conv weights of shape
    #    (kt, kh, kw, 1, c), and among the dims in the previous latent frame,
    #    take the top mask_config.learn_prev_frame_mask_num_dims spatial dims
    #    in terms of mean magnitude.
    # 3. Use these dims (different dims chosen per grid) as the fixed grid masks
    #    to use for training from scratch.
    if (
        mask_config is not None
        and mask_config.use_custom_masking
        and mask_config.learn_prev_frame_mask
    ):
      logging.info(
          'Start training with per-grid conditioning of conv entropy model '
          'in order to learn mask for previous latent frame.'
      )
      # Use masking given by mask_config, except use the full context for
      # the previous frame for the grids in mask_config.prev_frame_mask_grids.
      learn_prev_frame_mask_top_lefts = []
      for grid_idx in range(self.config.model.latents.num_grids):
        prev_frame_mask_top_left = (
            (0, 0) if grid_idx in mask_config.prev_frame_mask_grids else None
        )
        learn_prev_frame_mask_top_lefts.append(prev_frame_mask_top_left)
      learn_prev_frame_mask_top_lefts = tuple(learn_prev_frame_mask_top_lefts)

      # Initialize params and opt_state separately for this phase.
      logging.info('Temporary parameters for learning masking:')
      params_mask, opt_state_mask = self.init_params(
          input_res=inputs.shape[:-1],
          soft_round_temp=self.config.quant.soft_round_temp_start,
          input_mean=input_mean,
          learn_mask=True,
          prev_frame_mask_top_lefts=learn_prev_frame_mask_top_lefts,
      )

      for i in range(mask_config.learn_prev_frame_mask_iter):
        # Split `rng` to ensure we use a different noise for noise quantization
        # at each step.
        # It is faster to split this on the CPU than outside of jit on GPU.
        with jax.default_device(jax.local_devices(backend='cpu')[0]):
          rng, rng_used = jax.random.split(rng)

        # It is faster to compute schedules on CPU than outside of jit on GPU.
        with jax.default_device(jax.local_devices(backend='cpu')[0]):
          # Set soft-rounding temperature.
          soft_round_temp = soft_round_temp_fn(i)
          # Set rd_weight.
          rd_weight = rd_weight_fn(i)
          # Get parameter for Kumaraswamy noise distribution.
          kumaraswamy_a = kumaraswamy_a_fn(i)
        # Run a single training step
        params_mask, opt_state_mask, train_metrics = (
            self.single_train_step(
                params=params_mask,
                opt_state=opt_state_mask,
                inputs=inputs,
                rng=rng_used,
                quant_type=quant_type,
                rd_weight=rd_weight,
                soft_round_temp=soft_round_temp,
                kumaraswamy_a=kumaraswamy_a,
                prev_frame_mask_top_lefts=learn_prev_frame_mask_top_lefts,
                learn_mask=True,
            )
        )
        if i % self.config.opt.learn_mask_log_every == 0:
          end = time.time()
          self._log_train_metrics(
              i,
              train_metrics,
              end - start,
              num_pixels=np.prod(inputs.shape[:-1]),
              rd_weight=rd_weight,
              soft_round_temp=soft_round_temp,
              learn_mask=True,
          )
          start = time.time()
      # After mask_config.learn_prev_frame_mask_iter many iterations, take the
      # top mask_config.learn_prev_frame_mask_num_dims spatial dims of the first
      # layer of the entropy model in terms of mean magnitude. Do this per grid.
      prev_frame_mask_top_lefts = []
      for grid_idx in range(self.config.model.latents.num_grids):
        if grid_idx in mask_config.prev_frame_mask_grids:
          # Shape (kh, kw, 1, model.entropy.layers[0])
          prev_frame = params_mask[
              'autoregressive_entropy_model_conv_video/~/'
              f'grid_{grid_idx}_layer_0/conv_prev'
          ]['w']
          kh, kw, _, _ = prev_frame.shape
          # Do below computation in numpy for np.argpartition.
          prev_frame = np.array(prev_frame)
          arr = np.mean(np.abs(prev_frame), axis=(-2, -1))  # (kh, kw)
          # Find the contiguous mask with the best alignment with the mask
          # learned above, and log this.
          mask_h, mask_w = mask_config.prev_frame_contiguous_mask_shape
          assert mask_h <= kh and mask_w <= kw, (
              f'Contiguous mask shape ({mask_h},{mask_w}) must be smaller'
              ' than the spatial kernel dims'
          )
          # Sweep over all possible x,y indices of the top right of contiguous
          # mask to compute the total magnitude between the contiguous
          # mask at that location and learned mask above. Note that the
          # possible values of x index ranges from 0 to kh - mask_h, and
          # simlar for y. This can be implemented efficiently with a 2D Conv of
          # kernel np.ones((mask_h, mask_w)).
          kernel = np.ones((mask_h, mask_w))
          total_magnitude = scipy.signal.convolve2d(arr, kernel, mode='valid')
          # pylint: disable=unbalanced-tuple-unpacking
          y_min_best, x_min_best = np.unravel_index(
              total_magnitude.argmax(), total_magnitude.shape
          )
          # Use this to update `prev_frame_mask` to be True inside the
          # contiguous mask and False otherwise.
          prev_frame_mask = np.zeros_like(arr, dtype=bool)
          prev_frame_mask[
              y_min_best : y_min_best + mask_h,
              x_min_best : x_min_best + mask_w,
          ] = True
          assert np.sum(prev_frame_mask) == mask_h * mask_w
          prev_frame_mask_top_left = (y_min_best, x_min_best)
          prev_frame_mask_top_lefts.append(prev_frame_mask_top_left)
          logging.info(
              'Learned contiguous masking of previous latent frame for'
              ' grid %d, with prev_frame_mask_top_left: (%d, %d)',
              grid_idx,
              prev_frame_mask_top_left[0],
              prev_frame_mask_top_left[1]
          )
          with np.printoptions(threshold=np.inf):
            logging.info(prev_frame_mask)
        else:
          prev_frame_mask_top_lefts.append(None)
          logging.info(
              'Masking of previous latent frame for grid not learned because '
              'grid %d not in %s',
              grid_idx, str(mask_config.prev_frame_mask_grids)
          )
          logging.info(
              'Hence masking of previous latent frame for grid %d is all False',
              grid_idx
          )
    if prev_frame_mask_top_lefts is not None:
      # Convert prev_frame_mask_top_lefts to tuple that is hashable
      prev_frame_mask_top_lefts = tuple(prev_frame_mask_top_lefts)

    return prev_frame_mask_top_lefts

  def fit_datum(self, inputs, rng):
    # Move input to the GPU (or other existing device). Otherwise it gets
    # transferred every microstep!
    inputs = jax.device_put(inputs, jax.devices()[0])

    if self.config.model.synthesis.b_last_init_input_mean:
      input_ndims = len(inputs.shape[:-1])  # 3 for video
      input_mean = jnp.mean(inputs, axis=tuple(range(input_ndims)))  # (3,)
      chex.assert_shape(input_mean, (3,))
    else:
      input_mean = None

    # Start training with (typically) noise quantization for `num_noise_steps`.
    # This uses adam with cosine decay schedule.
    quant_type = self.config.quant.noise_quant_type
    # Function for linear annealing of soft-rounding temperature.
    soft_round_temp_fn = optax.linear_schedule(
        init_value=self.config.quant.soft_round_temp_start,
        end_value=self.config.quant.soft_round_temp_end,
        transition_steps=self.config.opt.num_noise_steps - 1
    )
    # Function for optional linear warmup of rd_weight.
    if self.config.loss.rd_weight_warmup_steps > 0:
      rd_weight_fn = optax.linear_schedule(
          init_value=0,
          end_value=self.config.loss.rd_weight,
          transition_steps=self.config.loss.rd_weight_warmup_steps - 1
      )
    else:
      rd_weight_fn = lambda _: self.config.loss.rd_weight
    if self.config.quant.use_kumaraswamy_noise:
      kumaraswamy_a_fn = optax.linear_schedule(
          init_value=self.config.quant.kumaraswamy_init_value,
          end_value=self.config.quant.kumaraswamy_end_value,
          transition_steps=self.config.quant.kumaraswamy_decay_steps,
      )
    else:
      kumaraswamy_a_fn = lambda _: None

    # Optionally create / learn a custom mask for the entropy model.
    prev_frame_mask_top_lefts = self.create_custom_mask(
        inputs,
        input_mean,
        rng,
        soft_round_temp_fn,
        rd_weight_fn,
        kumaraswamy_a_fn,
        quant_type,
    )

    # Training phase where noise is added to latents
    logging.info(
        'Start training with noise added to latents.'
    )
    start = time.time()
    params, opt_state = self.init_params(
        input_res=inputs.shape[:-1],
        soft_round_temp=self.config.quant.soft_round_temp_start,
        input_mean=input_mean,
        prev_frame_mask_top_lefts=prev_frame_mask_top_lefts,
    )

    for i in range(self.config.opt.num_noise_steps):
      # Split `rng` to ensure we use a different noise for noise quantization
      # at each step.
      # It is faster to split this on the CPU than outside of jit on GPU.
      with jax.default_device(jax.local_devices(backend='cpu')[0]):
        rng, rng_used = jax.random.split(rng)

      # It is faster to compute schedules on CPU than outside of jit on GPU.
      with jax.default_device(jax.local_devices(backend='cpu')[0]):
        # Set soft-rounding temperature.
        soft_round_temp = soft_round_temp_fn(i)
        # Set rd_weight.
        rd_weight = rd_weight_fn(i)
        # Get parameter for Kumaraswamy noise distribution.
        kumaraswamy_a = kumaraswamy_a_fn(i)
      # Run a single training step
      params, opt_state, train_metrics = self.single_train_step(
          params=params,
          opt_state=opt_state,
          inputs=inputs,
          rng=rng_used,
          quant_type=quant_type,
          rd_weight=rd_weight,
          soft_round_temp=soft_round_temp,
          kumaraswamy_a=kumaraswamy_a,
          prev_frame_mask_top_lefts=prev_frame_mask_top_lefts,
      )

      if i % self.config.opt.noise_log_every == 0:
        end = time.time()
        self._log_train_metrics(i, train_metrics, end - start,
                                num_pixels=np.prod(inputs.shape[:-1]),
                                rd_weight=rd_weight,
                                soft_round_temp=soft_round_temp)
        start = time.time()

    # Switch from noise quantization to straight through estimator of proper
    # (rounding) quantization after we finished the `num_noise_steps` many
    # iterations. This either
    # Option 1: continues the cosine decay schedule for `max_num_ste_steps`
    # Option 2: uses adam with constant learning rate (starting value =
    #   specified in config). We monitor progress of the optimisation and return
    #   the best params out of (according to loss value with rounding
    #   quantization):
    #     * params at the end of training with "noise" quantization
    #     * best throughout training with STE quantization
    if self.config.opt.max_num_ste_steps > 0:
      quant_type = self.config.quant.ste_quant_type
      assert 'ste' in quant_type
      best_params = params
      best_opt_state = opt_state
      best_loss = self.eval(params, inputs, prev_frame_mask_top_lefts)['loss']
      logging.info('Switched to STE!')
      logging.info('Best loss before STE: %.10e', best_loss)
      steps_not_improved = 0
      learning_rate = self.config.opt.ste_init_lr
      rd_weight = self.config.loss.rd_weight  # To satisfy linter
      for i in range(self.config.opt.max_num_ste_steps):
        if self.config.opt.ste_uses_cosine_decay:
          # STE continues to use cosine decay lr schedule
          params, opt_state, train_metrics = self.single_train_step(
              params=params,
              opt_state=opt_state,
              inputs=inputs,
              rng=None,  # no randomness used
              quant_type=quant_type,
              rd_weight=rd_weight,
              soft_round_temp=self.config.quant.ste_soft_round_temp,
              prev_frame_mask_top_lefts=prev_frame_mask_top_lefts,
          )
        else:
          # STE uses automatically decaying lr schedule
          params, opt_state, train_metrics = self.single_train_step(
              params=params,
              opt_state=opt_state,
              inputs=inputs,
              rng=None,  # no randomness used
              quant_type=quant_type,
              rd_weight=rd_weight,
              use_cosine_schedule=False,
              learning_rate=learning_rate,
              soft_round_temp=self.config.quant.ste_soft_round_temp,
              prev_frame_mask_top_lefts=prev_frame_mask_top_lefts,
          )
          if train_metrics['loss'] < best_loss:
            best_params = params
            best_opt_state = opt_state
            best_loss = train_metrics['loss']
            steps_not_improved = 0
          else:
            steps_not_improved += 1
          if (
              steps_not_improved == self.config.opt.ste_num_steps_not_improved):
            learning_rate = self.config.opt.ste_lr_decay_factor * learning_rate
            logging.info('Step: %d, new learning rate: %.5e', i, learning_rate)
            steps_not_improved = 0
            # Optionally reset parameters to previous best after lowering the
            # learning rate
            if self.config.opt.ste_reset_params_at_lr_decay:
              params = best_params
              opt_state = best_opt_state
            if learning_rate < self.config.opt.ste_break_at_lr:
              logging.info(
                  'Learning rate %.5e below threshold %.5e',
                  learning_rate, self.config.opt.ste_break_at_lr)
              break
        if i % self.config.opt.ste_log_every == 0:
          end = time.time()
          self._log_train_metrics(
              i + self.config.opt.num_noise_steps,
              train_metrics,
              end - start,
              num_pixels=np.prod(inputs.shape[:-1]),
              rd_weight=rd_weight,
              soft_round_temp=soft_round_temp,
          )
          start = time.time()

      if not self.config.opt.ste_uses_cosine_decay:
        params = best_params
      logging.info(
          '%d: Best loss after STE: %.10e. Finished with learning_rate %.4e',
          i,  # pylint: disable=undefined-variable
          min(best_loss, train_metrics['loss']),  # pylint: disable=undefined-variable
          (
              learning_rate if not self.config.opt.ste_uses_cosine_decay
              else self.config.opt.cosine_decay_schedule_kwargs.alpha
              * self.config.opt.cosine_decay_schedule_kwargs.init_value
          ),
      )
    return params, prev_frame_mask_top_lefts

  def _quantize_network_params(
      self, q_step_weight: float, q_step_bias: float
  ) -> hk.Params:
    """Returns quantized network parameters."""
    synthesis_model = self._get_synthesis_model(
        b_last_init_input_mean=None, is_video=True
    )
    # learn_mask=True is only ever used for training.
    entropy_model = self._get_entropy_model(learn_mask=False)

    quantized_synth_params = synthesis_model.get_quantized_nested_params(
        q_step_weight=q_step_weight, q_step_bias=q_step_bias
    )

    quantized_entropy_params = entropy_model.get_quantized_nested_params(
        q_step_weight=q_step_weight, q_step_bias=q_step_bias
    )

    return hk.data_structures.merge(
        quantized_synth_params, quantized_entropy_params
    )

  def _get_network_params_bits(
      self, q_step_weight: float, q_step_bias: float
  ) -> Mapping[str, float]:
    """Returns a dictionary of numbers of bits for different model parts."""
    synthesis_model = self._get_synthesis_model(
        b_last_init_input_mean=None, is_video=True
    )
    # learn_mask=True is only ever used for training.
    entropy_model = self._get_entropy_model(learn_mask=False)

    synthesis_rate = synthesis_model.compute_rate(
        q_step_weight=q_step_weight, q_step_bias=q_step_bias
    )
    entropy_rate = entropy_model.compute_rate(
        q_step_weight=q_step_weight, q_step_bias=q_step_bias
    )

    return immutabledict.immutabledict({
        'synthesis': synthesis_rate,
        'entropy': entropy_rate,
        'total': synthesis_rate + entropy_rate,
    })

  def quantization_step_search(
      self,
      params: hk.Params,
      inputs: Array,
      prev_frame_mask_top_lefts: tuple[
          tuple[int, int] | None, ...
      ] | None = None,
  ) -> tuple[hk.Params, Mapping[str, Any]]:
    """Searches for best weight and bias quantization step sizes.

    Args:
      params:
      inputs:
      prev_frame_mask_top_lefts:

    Returns:
      The best quantized params and the rates corresponding to the cost of
      storing the quantized model in bits.
    """
    num_pixels = self._num_pixels(inputs.shape[:-1])  # without channels.
    best_loss = float('inf')
    best_quantized_params = None
    best_metrics = None

    for q_step_weight in self.config.model.quant.q_steps_weight:
      for q_step_bias in self.config.model.quant.q_steps_bias:
        model_rates = hk.transform(self._get_network_params_bits).apply(
            params, None, q_step_weight, q_step_bias  # no rng needed
        )
        quantized_params = hk.transform(self._quantize_network_params).apply(
            params, None, q_step_weight, q_step_bias  # no rng needed
        )
        # Take latent parameters from `params` and overwrite the synthesis and
        # entropy parameters with their quantized version.
        quantized_params = hk.data_structures.merge(params, quantized_params)
        metrics = self.eval(quantized_params, inputs, prev_frame_mask_top_lefts)
        # Total rate corresponds to rate of entropy coded latents (first term)
        # and rate of synthesis and entropy model MLPs (second term)
        total_rate = metrics['rate'] + model_rates['total']
        # Note that the loss here is slightly different from the one we use to
        # optimize the params. On top of the distortion and the rate of the
        # latents (which is what is included in the standard loss), we also
        # minimize the rate of the synthesis and entropy models. This could in
        # principle also be optimized directly (by adding noise to the
        # parameters of the networks during training). However, this is not how
        # it was implemented in COOL-CHIC, but might be an interesting direction
        # for future work.
        loss = (
            metrics['distortion']
            + self.config.loss.rd_weight * total_rate / num_pixels
        )
        if loss < best_loss:
          best_loss = loss
          best_quantized_params = quantized_params
          # Save metrics including the model rates and quantization steps
          best_metrics = metrics | model_rates
          best_metrics = best_metrics | {
              'q_step_weight': q_step_weight,
              'q_step_bias': q_step_bias,
          }
    if best_metrics is None:
      best_quantized_params = params
      best_metrics = {k: float('inf') for k in metrics | model_rates}
      best_metrics = best_metrics | {
          'q_step_weight': float('inf'),
          'q_step_bias': float('inf'),
      }
      logging.warn(
          'Optimization appears to be unstable. Check hyperparameters.'
      )
    return best_quantized_params, best_metrics

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, *, global_step, rng, writer):
    # rng has shape [num_devices, 2].
    # We only use single-device so just use the first device's rng.
    rng = rng[0]  # [2]

    # Fit each video and log metrics pre/post model quantization per video.
    metrics_per_datum = collections.defaultdict(list)

    for i, input_dict in enumerate(self._train_data_iterator):
      # Extract video as array of shape [T, H, W, C]
      inputs = input_dict['array'].numpy()
      input_shape = inputs.shape
      num_pixels = self._num_pixels(input_res=input_shape[:-1])
      logging.info('inputs shape: %s', input_shape)
      logging.info('num_pixels: %s', num_pixels)

      # Compute MACs per pixel.
      macs_per_pixel = self._count_macs_per_pixel(input_shape)

      # Fit inputs of shape [T, H, W, C].
      params, prev_frame_mask_top_lefts = self.fit_datum(inputs, rng)

      # Evaluate unquantized model after training. Note that this will *not*
      # include model parameters in rate calculations.
      metrics = self.eval(
          params, inputs, prev_frame_mask_top_lefts
      )

      # Perform search over quantization steps to find best quantized model
      # params (in terms of rate-distortion loss).
      logging.info('Started quantization step search.')
      _, quantized_metrics = self.quantization_step_search(
          params, inputs, prev_frame_mask_top_lefts
      )
      logging.info('Finished quantization step search.')

      # Save metrics
      # Reconstruction metrics
      keys = ['psnr', 'loss', 'distortion', 'ssim', 'per_frame_distortion']
      for key in keys:
        metrics_per_datum[key].append(metrics[key])
        metrics_per_datum[f'{key}_quantized'].append(quantized_metrics[key])
      # Rate metrics
      metrics_per_datum['rate_latents'].append(metrics['rate'])
      metrics_per_datum['bpp_latents'].append(metrics['rate'] / num_pixels)
      metrics_per_datum['rate_latents_quantized'].append(
          quantized_metrics['rate']
      )
      metrics_per_datum['bpp_latents_quantized'].append(
          quantized_metrics['rate'] / num_pixels
      )
      for key in ['synthesis', 'entropy']:
        metrics_per_datum[f'rate_{key}'].append(quantized_metrics[key])
        metrics_per_datum[f'bpp_{key}'].append(
            quantized_metrics[key] / num_pixels
        )
      metrics_per_datum['rate_total'].append(
          quantized_metrics['rate']
          + quantized_metrics['synthesis']
          + quantized_metrics['entropy']
      )
      metrics_per_datum['bpp_total'].append(
          metrics_per_datum['rate_total'][-1] / num_pixels
      )
      metrics_per_datum['q_step_weight'].append(
          quantized_metrics['q_step_weight']
      )
      metrics_per_datum['q_step_bias'].append(quantized_metrics['q_step_bias'])

      # Add macs per pixel metrics
      for key, val in macs_per_pixel.items():
        metrics_per_datum[f'macs_per_pixel_{key}'].append(val)

      # For video, also add timestep, video_id and patch_id.
      # This is needed for evaluating per frame psnr.
      # Each element of timestep is a tensor that needs to be converted back
      # to a scalar.
      timestep = np.array([t.item() for t in input_dict['timestep']])
      metrics_per_datum['timestep'].append(timestep)
      metrics_per_datum['video_id'].append(input_dict['video_id'][0])
      metrics_per_datum['patch_id'].append(input_dict['patch_id'])

      # Log metrics for latest video
      logging_message = f'Train datum {i:05}: ' + str(
          {metric: vals[-1] for metric, vals in metrics_per_datum.items()}
      )
      logging_message = textwrap.fill(logging_message, 80)
      logging.info(logging_message)

    # Compute mean metrics and log these. For video, exclude 'timestep',
    # 'video_id', 'patch_id', 'per_frame_distortion_quantized', which will be
    # used for per frame psnr evaluation.
    exclude_keys = ['video_id', 'timestep', 'patch_id',
                    'per_frame_distortion_quantized']
    return_metrics = {
        key: jnp.mean(jnp.array(value))
        for key, value in metrics_per_datum.items() if key not in exclude_keys
    }
    # Compute metrics needed for per frame psnr evaluation and log these
    # separately as they are not scalars:
    # per_frame_distortion_quantized (num_examples, t),
    # video_id (num_examples,), timestep (num_examples, t),
    # patch_id (num_examples,) per_patch_bpp (num_examples,)
    # for per frame psnr evaluation & per video bpp evaluation down the line.
    video_id = jnp.stack(metrics_per_datum['video_id'], axis=0)
    timestep = jnp.stack(metrics_per_datum['timestep'], axis=0)
    patch_id = jnp.stack(metrics_per_datum['patch_id'], axis=0)
    per_frame_distortion_quantized = jnp.stack(
        metrics_per_datum['per_frame_distortion_quantized'], axis=0)
    per_patch_bpp = jnp.stack(metrics_per_datum['bpp_total'], axis=0)
    extra_metrics = {
        'video_id': video_id,
        'timestep': timestep,
        'patch_id': patch_id,
        'per_frame_distortion_quantized': per_frame_distortion_quantized,
        'per_patch_bpp': per_patch_bpp,
    }
    all_metrics = dict(return_metrics, **extra_metrics)
    logging_message = 'All metrics: ' + str(all_metrics)
    logging_message = textwrap.fill(logging_message, 80)
    logging.info(logging_message)
    return return_metrics


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
