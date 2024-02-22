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

"""Jaxline experiment for C3 image experiments. Assume single-device."""

import collections
from collections.abc import Mapping
import functools
import textwrap
import time

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

from c3_neural_compression.experiments import base
from c3_neural_compression.model import entropy_models
from c3_neural_compression.utils import experiment as experiment_utils
from c3_neural_compression.utils import psnr as psnr_utils


Array = chex.Array

FLAGS = flags.FLAGS


class Experiment(base.Experiment):
  """Per data-point compression experiment for images. Assume single-device."""

  def init_params(self, input_res, soft_round_temp=None, input_mean=None):
    forward_init = jax.jit(
        self.forward.init, static_argnames=('quant_type', 'input_res')
    )
    params = forward_init(
        self.init_rng,
        quant_type=self.config.quant.noise_quant_type,
        input_res=tuple(input_res),
        soft_round_temp=soft_round_temp,
        input_mean=input_mean,
    )
    opt = self.get_opt(use_cosine_schedule=True)
    opt_state = jax.jit(opt.init)(params)
    experiment_utils.log_params_info(params)
    return params, opt_state

  def _get_upsampling_fn(self, input_res):
    num_dims = len(input_res)
    assert num_dims == 2
    return super()._get_upsampling_fn(input_res)

  def _get_entropy_model(self):
    """Returns entropy model."""
    return entropy_models.AutoregressiveEntropyModelConvImage(
        **self.config.model.entropy,
    )

  def _get_entropy_params(self, latent_grids):
    """Returns parameters of autoregressive Laplace distribution."""
    entropy_model = self._get_entropy_model()
    return entropy_model(latent_grids)

  def _forward_fn(
      self,
      quant_type,
      input_res,
      soft_round_temp=None,
      input_mean=None,
      kumaraswamy_a=None,
  ):
    """Forward pass of C3.

    Args:
      quant_type: Quantization type changes during training, so must be passed
        as argument to function.
      input_res: Shape of input without channels.
      soft_round_temp: Smoothness of soft-rounding.
      input_mean: Optionally the mean RGB values of the input is used to
        initialise the last bias of the synthesis network. The mean is across
        the spatial dimensions. I.e., the shape of the input mean is (3,).
      kumaraswamy_a: `a` parameter of Kumaraswamy noise distribution.

    Returns:
      input_rec: Predicted image of shape (H, W, out_channels).
      grids: Latent grids as tuple of length num_grids containing grids of shape
      (H, W), (H/2, W/2), etc.
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
    loc, scale = self._get_entropy_params(latent_grids)

    # Flatten latent grids into a single latent vector of shape (num_latents,)
    # containing all latents
    all_latents = entropy_models.flatten_latent_grids(latent_grids)

    upsampled_latents = self._upsample_latents(latent_grids, input_res)
    # Synthesize image of shape (H, W, out_channels) from latent volume
    input_rec = self._synthesize(
        upsampled_latents, b_last_init_input_mean=input_mean, is_video=False
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
  ):
    """Rate distortion loss: distortion + lambda * rate.

    Args:
      params: Haiku parameters.
      target: Target image. Array of shape (H, W, 3).
      rng:
      quant_type: Type of quantization to use.
      soft_round_temp: Smoothness of soft-rounding.
      rd_weight: rd_weight used to weigh the bpp term in the loss.
      kumaraswamy_a: `a` parameter of Kumaraswamy noise distribution.

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
  @functools.partial(jax.jit, static_argnums=(0, 5, 8))
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
      metrics = metrics | {
          'latent_grad_norm': latent_grad_norm,
          'synthesis_grad_norm': synthesis_grad_norm,
          'entropy_grad_norm': entropy_grad_norm,
      }
    # Pass `params` for weight decay.
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics

  @functools.partial(jax.jit, static_argnums=(0, 3))
  def eval(self, params, inputs, blocked_rates=False):
    # We always use rounding for quantization at test time. Note that we call
    # the function with "ste" as this is equivalent to "round" at test time.
    out = self.forward.apply(params=params, rng=None, quant_type='ste',
                             input_res=inputs.shape[:-1])
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
    return metrics

  def _log_train_metrics(self, i, metrics, delta_time, num_pixels,
                         **other_metrics):
    logging_message = (
        f'{i}/{self.config.opt.num_noise_steps}+{self.config.opt.max_num_ste_steps},'
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

  def fit_datum(self, inputs, rng):
    # Move input to the GPU (or other existing device). Otherwise it gets
    # transferred every microstep!
    inputs = jax.device_put(inputs, jax.devices()[0])

    if self.config.model.synthesis.b_last_init_input_mean:
      input_ndims = len(inputs.shape[:-1])  # 2 for images.
      input_mean = jnp.mean(inputs, axis=tuple(range(input_ndims)))  # (3,)
      chex.assert_shape(input_mean, (3,))
    else:
      input_mean = None
    params, opt_state = self.init_params(
        input_res=inputs.shape[:-1],
        soft_round_temp=self.config.quant.soft_round_temp_start,
        input_mean=input_mean,
    )

    start = time.time()

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
      best_loss = self.eval(params, inputs)['loss']  # no randomness used
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
    return params

  def _quantize_network_params(
      self, q_step_weight: float, q_step_bias: float
  ) -> hk.Params:
    """Returns quantized network parameters."""
    synthesis_model = self._get_synthesis_model(
        b_last_init_input_mean=None, is_video=False
    )
    entropy_model = self._get_entropy_model()

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
        b_last_init_input_mean=None, is_video=False
    )
    entropy_model = self._get_entropy_model()

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
      self, params: hk.Params, inputs: Array
  ) -> tuple[hk.Params, Mapping[str, float]]:
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
        metrics = self.eval(quantized_params, inputs, blocked_rates=False)
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

    # Fit each image and log metrics pre/post model quantization per image.
    metrics_per_datum = collections.defaultdict(list)

    for i, input_dict in enumerate(self._train_data_iterator):
      # Extract image as array of shape [H, W, C]
      inputs = input_dict['array'].numpy()
      input_shape = inputs.shape
      num_pixels = self._num_pixels(input_res=input_shape[:-1])
      logging.info('inputs shape: %s', input_shape)
      logging.info('num_pixels: %s', num_pixels)

      # Compute MACs per pixel.
      macs_per_pixel = self._count_macs_per_pixel(input_shape)

      # Fit inputs of shape [H, W, C].
      params = self.fit_datum(inputs, rng)

      # Evaluate unquantized model after training. Note that this will *not*
      # include model parameters in rate calculations.
      metrics = self.eval(params, inputs, blocked_rates=True)

      # Perform search over quantization steps to find best quantized model
      # params (in terms of rate-distortion loss).
      logging.info('Started quantization step search.')
      _, quantized_metrics = self.quantization_step_search(
          params, inputs
      )
      logging.info('Finished quantization step search.')

      # Save metrics
      # Reconstruction metrics
      keys = ['psnr', 'loss', 'distortion', 'ssim']
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

      # Log metrics for latest image
      logging_message = f'Train datum {i:05}: ' + str(
          {metric: vals[-1] for metric, vals in metrics_per_datum.items()}
      )
      logging_message = textwrap.fill(logging_message, 80)
      logging.info(logging_message)

    # Compute mean metrics and log these.
    all_metrics = {
        key: jnp.mean(jnp.array(value))
        for key, value in metrics_per_datum.items()
    }

    # Add per image metrics to dictionary of all metrics. Note that we store per
    # image metrics as scalars with an image index. Index i refers to index of
    # image.
    if self.config.log_per_datum_metrics:
      for i in range(len(metrics_per_datum['psnr'])):
        all_metrics = all_metrics | {
            f'{metric}_{i}': vals[i]
            for metric, vals in metrics_per_datum.items()
        }

    logging_message = 'All metrics: ' + str(all_metrics)
    logging_message = textwrap.fill(logging_message, 80)
    logging.info(logging_message)

    return all_metrics

  def _num_latents(self, input_res, params=None):
    """Count total number of latents."""
    if params is None:
      params = self.init_params(
          input_res=input_res,
          soft_round_temp=self.config.quant.soft_round_temp_start,
      )[0]
    _, latent_params = experiment_utils.partition_params_by_module_name(
        params, key='latent'
    )
    latent_sizes_per_grid = jax.tree_util.tree_map(
        lambda x: x.size, latent_params
    )
    total_latent_size = sum(jax.tree_util.tree_leaves(latent_sizes_per_grid))
    return total_latent_size

if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))

