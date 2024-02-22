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

"""C3 base experiment. Contains shared methods for images and video."""

import abc
from collections.abc import Mapping
import functools

import chex
import haiku as hk
from jaxline import experiment
from ml_collections import config_dict
import numpy as np

from c3_neural_compression.model import latents
from c3_neural_compression.model import synthesis
from c3_neural_compression.model import upsampling
from c3_neural_compression.utils import data_loading
from c3_neural_compression.utils import experiment as experiment_utils
from c3_neural_compression.utils import macs

Array = chex.Array


class Experiment(experiment.AbstractExperiment):
  """Per data-point compression experiment. Assume single-device."""

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super().__init__(mode=mode, init_rng=init_rng)
    self.mode = mode
    self.init_rng = init_rng
    # This config holds all the experiment specific keys defined in get_config
    self.config = config

    # Define model and forward function (note that since we use noise
    # quantization on the latents we cannot apply without rng)
    self.forward = hk.transform(self._forward_fn)

    assert (
        self.config.loss.rd_weight_warmup_steps
        <= self.config.opt.num_noise_steps
    )

    # Set up train/test data loader. A datum can either be a full video or a
    # spatio-temporal patch of video, depending on the values of `num_frames`
    # and `spatial_patch_size`.
    self._train_data_iterator = data_loading.load_dataset(
        dataset_name=self.config.dataset.name,
        root=config.dataset.root_dir,
        skip_examples=config.dataset.skip_examples,
        num_examples=config.dataset.num_examples,
        # UVG specific kwargs
        num_frames=config.dataset.num_frames,
        spatial_patch_size=config.dataset.get('spatial_patch_size', None),
        video_idx=config.dataset.video_idx,
    )

  def get_opt(
      self, use_cosine_schedule: bool, learning_rate: float | None = None
  ):
    """Returns optimizer."""
    if use_cosine_schedule:
      opt = experiment_utils.make_opt(
          transform_name='scale_by_adam',
          transform_kwargs={},
          global_max_norm=self.config.opt.grad_norm_clip,
          cosine_decay_schedule=self.config.opt.cosine_decay_schedule,
          cosine_decay_schedule_kwargs=self.config.opt.cosine_decay_schedule_kwargs,
      )
    else:
      opt = experiment_utils.make_opt(
          transform_name='scale_by_adam',
          transform_kwargs={},
          global_max_norm=self.config.opt.grad_norm_clip,
          learning_rate=learning_rate,
      )
    return opt

  @abc.abstractmethod
  def init_params(self, *args, **kwargs):
    raise NotImplementedError()

  def _get_upsampling_fn(self, input_res):
    if self.config.model.upsampling.type == 'image_resize':
      upsampling_fn = functools.partial(
          upsampling.jax_image_upsampling,
          input_res=input_res,
          **self.config.model.upsampling.kwargs,
      )
    else:
      raise ValueError(
          f'Unknown upsampling fn: {self.config.model.upsampling.type}'
      )
    return upsampling_fn

  def _num_pixels(self, input_res):
    """Returns number of pixels in the input."""
    return np.prod(input_res)

  def _get_latents(
      self,
      quant_type,
      input_res,
      soft_round_temp=None,
      kumaraswamy_a=None,
  ):
    """Returns the latents."""
    # grids: tuple of arrays of size ({T/2^i}, H/2^i, W/2^i) for i in
    # range(num_grids)
    latent_grids = latents.Latent(
        input_res=input_res,
        num_grids=self.config.model.latents.num_grids,
        add_gains=self.config.model.latents.add_gains,
        learnable_gains=self.config.model.latents.learnable_gains,
        gain_values=self.config.model.latents.gain_values,
        gain_factor=self.config.model.latents.gain_factor,
        q_step=self.config.model.latents.q_step,
        downsampling_factor=self.config.model.latents.downsampling_factor,
        downsampling_exponents=self.config.model.latents.downsampling_exponents,
    )(quant_type, soft_round_temp=soft_round_temp, kumaraswamy_a=kumaraswamy_a)
    return latent_grids

  def _upsample_latents(self, latent_grids, input_res):
    """Returns upsampled and stacked latent grids."""

    upsampling_fn = self._get_upsampling_fn(input_res)
    # Upsample all latent grids to ({T}, H, W) resolution and stack along last
    # dimension
    upsampled_latents = upsampling_fn(latent_grids)  # ({T}, H, W, n_grids)

    return upsampled_latents

  @abc.abstractmethod
  def _get_entropy_model(self, *args, **kwargs):
    """Returns entropy model."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _get_entropy_params(self, *args, **kwargs):
    """Returns parameters of autoregressive Laplace distribution."""
    raise NotImplementedError()

  def _get_synthesis_model(self, b_last_init_input_mean=None, is_video=False):
    """Returns synthesis model."""
    if not self.config.model.synthesis.b_last_init_input_mean:
      assert b_last_init_input_mean is None, (
          '`b_last_init_input_mean` is not None but `b_last_init_input_mean` is'
          ' `False`.'
      )
    out_channels = data_loading.DATASET_ATTRIBUTES[self.config.dataset.name][
        'num_channels'
    ]
    return synthesis.Synthesis(
        out_channels=out_channels,
        is_video=is_video,
        b_last_init_value=b_last_init_input_mean,
        **self.config.model.synthesis,
    )

  def _synthesize(self, upsampled_latents, b_last_init_input_mean=None,
                  is_video=False):
    """Synthesizes image or video from upsampled latents."""
    synthesis_model = self._get_synthesis_model(
        b_last_init_input_mean, is_video
    )
    return synthesis_model(upsampled_latents)

  @abc.abstractmethod
  def _forward_fn(self, *args, **kwargs):
    """Forward pass C3."""
    raise NotImplementedError()

  def _count_macs_per_pixel(self, input_shape):
    """Counts number of multiply accumulates per pixel."""
    num_dims = len(input_shape) - 1
    context_num_rows_cols = self.config.model.entropy.context_num_rows_cols
    if isinstance(context_num_rows_cols, int):
      ps = (2 * context_num_rows_cols + 1,) * num_dims
    else:
      assert len(context_num_rows_cols) == num_dims
      ps = tuple(2 * c + 1 for c in context_num_rows_cols)
    context_size = (np.prod(ps) - 1) // 2
    entropy_config = self.config.model.entropy
    synthesis_config = self.config.model.synthesis
    return macs.get_macs_per_pixel(
        input_shape=input_shape,
        layers_synthesis=synthesis_config.layers,
        layers_entropy=entropy_config.layers,
        context_size=context_size,
        num_grids=self.config.model.latents.num_grids,
        upsampling_type=self.config.model.upsampling.type,
        upsampling_kwargs=self.config.model.upsampling.kwargs,
        synthesis_num_residual_layers=synthesis_config.num_residual_layers,
        synthesis_residual_kernel_shape=synthesis_config.residual_kernel_shape,
        downsampling_factor=self.config.model.latents.downsampling_factor,
        downsampling_exponents=self.config.model.latents.downsampling_exponents,
        # Below we have arguments that are only defined in some configs
        entropy_use_prev_grid=entropy_config.conditional_spec.get(
            'use_prev_grid', False
        ),
        entropy_prev_kernel_shape=entropy_config.conditional_spec.get(
            'prev_kernel_shape', None
        ),
        entropy_mask_config=entropy_config.get(
            'mask_config', config_dict.ConfigDict()
        ),
        synthesis_per_frame_conv=synthesis_config.get('per_frame_conv', False),
    )

  @abc.abstractmethod
  def _loss_fn(self, *args, **kwargs):
    """Rate distortion loss: distortion + lambda * rate."""
    raise NotImplementedError()

  @abc.abstractmethod
  def single_train_step(self, *args, **kwargs):
    """Runs one batch forward + backward and run a single opt step."""
    raise NotImplementedError()

  @abc.abstractmethod
  def eval(self, params, inputs, blocked_rates=False):
    """Return reconstruction, PSNR and SSIM for given params and inputs."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _log_train_metrics(self, *args, **kwargs):
    raise NotImplementedError()

  @abc.abstractmethod
  def fit_datum(self, inputs, rng):
    """Optimize model to fit the given datum (inputs)."""
    raise NotImplementedError()

  def _quantize_network_params(
      self, q_step_weight: float, q_step_bias: float
  ) -> hk.Params:
    """Returns quantized network parameters."""
    raise NotImplementedError()

  def _get_network_params_bits(
      self, q_step_weight: float, q_step_bias: float
  ) -> Mapping[str, float]:
    """Returns a dictionary of numbers of bits for different model parts."""
    raise NotImplementedError()

  @abc.abstractmethod
  def quantization_step_search(self, *args, **kwargs):
    """Searches for best weight and bias quantization step sizes."""
    raise NotImplementedError()

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  @abc.abstractmethod
  def step(self, *, global_step, rng, writer):
    """One step accounts for fitting all images/videos in dataset."""
    raise NotImplementedError()

  # Dummy evaluation. Needed for jaxline exp, although we only run train mode.
  def evaluate(self, global_step, rng, writer):
    raise NotImplementedError()
